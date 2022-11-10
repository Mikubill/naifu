
# torchrun  trainer.py --model_path=/tmp/model --config test-run.yaml

import json
import math
import time
import torch
from pathlib import Path

import hivemind
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from PIL import Image
from tqdm.auto import tqdm
from omegaconf import OmegaConf

from lib.args import parse_args
from lib.buckets import AspectRatioBucket, AspectRatioDataset, AspectRatioSampler
from lib.utils import AverageMeter, get_gpu_ram
from lib.models import load_models
from lib.args import parse_args
from lib.ema import EMAModel


torch.backends.cudnn.benchmark = True

args = parse_args()
config = OmegaConf.load(args.config)
                   
def init_arb_buckets(args, config, world_size):
    arg_config = {
        "bsz": args.train_batch_size,
        "seed": config.trainer.seed,
        "world_size": world_size,
        "global_rank": args.local_rank,
        **config.arb
    }

    if config.arb.debug:
        print("BucketManager initialized using config:")
        print(json.dumps(arg_config, sort_keys=True, indent=4))
    else:
        print(f"BucketManager initialized with base_res = {arg_config['base_res']}, max_size = {arg_config['max_size']}")
        
    def get_id_size_dict(entries, hint):
        id_size_map = {}

        for entry in tqdm(entries, desc=f"Loading resolution from {hint} images", disable=args.local_rank not in [0, -1]):
            with Image.open(entry) as img:
                size = img.size
            id_size_map[entry] = size

        return id_size_map
    
    instance_entries = [x for x in Path(config.dataset.img_path).iterdir() if x.is_file() and x.suffix != ".txt"]
    instance_id_size_map = get_id_size_dict(instance_entries, "instance")
    instance_bucket_manager = AspectRatioBucket(instance_id_size_map, **arg_config)
    
    return instance_bucket_manager

def get_world_size() -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def main(args):
    world_size = get_world_size()
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda')
    torch.manual_seed(config.trainer.seed)
    print(get_gpu_ram())

    tokenizer, text_encoder, vae, unet, noise_scheduler, optimizer, scheduler = load_models(args.model_path, config)
    ema_unet = EMAModel(unet.parameters())
    
    weight_dtype = torch.float16 if config.trainer.precision == "fp16" else torch.float32
    vae = vae.to(device, dtype=weight_dtype)
    unet = unet.to(device, dtype=weight_dtype)
    text_encoder = text_encoder.to(device, dtype=weight_dtype)
    ema_unet.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    if config.trainer.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # setup fp16 stuff
    scaler = hivemind.GradScaler()
    dht = hivemind.DHT(
        host_maddrs=[
            # Casting: C
            "/ip4/0.0.0.0/tcp/0", 
            "/ip4/0.0.0.0/udp/0/quic"
        ],
        initial_peers=config.peers, 
        start=True
    )     
    print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
    print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))
    print("Important Note: This is a RELAY of the existing session. You can share this peer adress to peers close to you!")

    optimizer = hivemind.Optimizer(
        dht=dht,                 
        run_id=config.name,        # unique identifier of this collaborative run
        optimizer=optimizer,      # wrap the SGD optimizer defined above
        scheduler=scheduler,
        verbose=True,              # print logs incessently
        **config.hivemind
    )

    train_dataset = AspectRatioDataset(
        tokenizer=tokenizer,
        size=config.trainer.resolution,
        bsz=args.train_batch_size,
        seed=config.trainer.seed,
        **config.dataset
    )

    def collate_fn(examples):
        if len(examples) == 1:
            examples = examples[0]
    
        input_ids = [example["prompt_ids"] for example in examples]
        pixel_values = [example["images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch
    
    instance_bucket_manager = init_arb_buckets(args, config, world_size)
    sampler = AspectRatioSampler(instance_bucket_manager, world_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, sampler=sampler)

    # Train!
    total_batch_size = args.train_batch_size * world_size 
    
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / len(train_dataloader))

    if args.local_rank in [0, -1]:
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num batches each epoch = {len(train_dataloader)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Total optimization steps = {args.num_train_epochs * len(train_dataloader)}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.num_train_epochs * len(train_dataloader)), disable=not args.local_rank in [0, -1])
    progress_bar.set_description("Steps")
    local_step = 0
    loss_avg = AverageMeter()

    for _ in range(args.num_train_epochs):
        unet.train()

        for _, batch in enumerate(train_dataloader):
            b_start = time.perf_counter()
            
            # Convert images to latent space
            with torch.no_grad():
                latent_dist = vae.encode(batch["pixel_values"].to(device, dtype=weight_dtype)).latent_dist
                latents = latent_dist.sample() * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch['input_ids'].to(device), output_hidden_states=True)
            encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states['hidden_states'][-2])

            # Predict the noise residual
            with torch.autocast('cuda'):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")    
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            loss_avg.update(loss.detach_(), bsz)
            
            if config.trainer.use_ema:
                ema_unet.step(unet.parameters())
                
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            
            if args.local_rank in [0, -1]:
                samples_seen = local_step * args.train_batch_size * world_size

                logs = {
                    "loss": loss_avg.avg.item(),
                    "epoch": optimizer.local_epoch,
                    "samples_seen": samples_seen,
                }
                progress_bar.set_postfix(**logs)
                progress_bar.update(1)
                local_step += 1
                
    torch.distributed.barrier()

if __name__ == "__main__":
    torch.distributed.init_process_group("nccl", init_method="env://")
    try:
        args = parse_args()
        main(args)
    finally:
        torch.distributed.destroy_process_group()