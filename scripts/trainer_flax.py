import logging
import math
import os

import torch
from data.buckets import AspectRatioSampler
from data.store import AspectRatioDataset, ImageStore
from lib.args import parse_args
from omegaconf import OmegaConf

import jax
import jax.numpy as jnp
import optax
import torch
import torch.utils.checkpoint
import transformers
from diffusers import FlaxAutoencoderKL, FlaxDDPMScheduler
from diffusers import FlaxStableDiffusionPipeline, FlaxUNet2DConditionModel
from diffusers.utils import check_min_version
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import Repository
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, FlaxCLIPTextModel, set_seed

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0")

# python trainer.py --model_path=/tmp/model --config config/test.yaml
args = parse_args()
config = OmegaConf.load(args.config)

def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    set_seed(config.trainer.seed)
    
    # Handle the repository creation
    if jax.process_index() == 0:
        if config.monitor.huggingface_repo != "":
            repo = Repository(config.checkpoint.dirpath, clone_from=config.monitor.huggingface_repo, token=config.monitor.hf_auth_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif config.checkpoint.dirpath != "":
            os.makedirs(config.checkpoint.dirpath, exist_ok=True)
            
            
    local_rank = jax.process_index()
    world_size = jax.local_device_count()
    dataset_cls = AspectRatioDataset if config.arb.enabled else ImageStore
    total_train_batch_size = config.trainer.init_batch_size * jax.local_device_count()

    weight_dtype = jnp.float32
    if config.trainer.precision == "fp16":
        weight_dtype = jnp.float16
    elif config.trainer.precision == "bf16":
        weight_dtype = jnp.bfloat16

    # Load models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(config.trainer.model_path, subfolder="tokenizer")
    noise_scheduler = FlaxDDPMScheduler.from_pretrained(config.trainer.model_path, subfolder="scheduler")
    text_encoder = FlaxCLIPTextModel.from_pretrained(config.trainer.model_path, subfolder="text_encoder", dtype=weight_dtype, from_pt=True)
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(config.trainer.model_path, subfolder="vae", dtype=weight_dtype, from_pt=True)
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(config.trainer.model_path, subfolder="unet", dtype=weight_dtype, from_pt=True)

  
    # init Dataset
    dataset = dataset_cls(
        size=config.trainer.resolution,
        seed=config.trainer.seed,
        rank=local_rank,
        init=not config.arb.enabled,
        tokenizer=tokenizer,
        **config.dataset
    )
    
    def collate_fn(examples):
        input_ids, pixel_value = dataset.collate_fn(examples)
        batch = {
            "pixel_values": pixel_value,
            "input_ids": input_ids,
        }
        batch = {k: v.numpy() for k, v in batch.items()}
        return batch

        
    # init sampler
    data_sampler = AspectRatioSampler(
        bsz=total_train_batch_size,
        config=config, 
        rank=local_rank, 
        dataset=dataset, 
        world_size=world_size,
    ) if config.arb.enabled else None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        sampler=data_sampler,
        num_workers=config.dataset.num_workers,
        batch_size=1 if data_sampler else total_train_batch_size,
        persistent_workers=True,
    )
    
    # Optimization
    learning_rate = config.optimizer.params.lr * total_train_batch_size
    if config.trainer.lr_scale == "sqrt":
        learning_rate = math.sqrt(learning_rate)

    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-8,
        peak_value=learning_rate,
        warmup_steps=100,
        decay_steps=20000,
        end_value=5e-7
    )
    adamw = optax.adamw(learning_rate=lr_scheduler)
    optimizer = optax.chain(optax.clip_by_global_norm(config.lightning.gradient_clip_val), adamw)
    state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)
    
    # Initialize our training
    rng = jax.random.PRNGKey(config.trainer.seed)
    train_rngs = jax.random.split(rng, jax.local_device_count())
    noise_scheduler = noise_scheduler[0]

    def train_step(state, vae_params, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

        def compute_loss(params):
            # Convert images to latent space
            vae_outputs = vae.apply(
                {"params": vae_params}, batch["pixel_values"], deterministic=True, method=vae.encode
            )
            latents = vae_outputs.latent_dist.sample(sample_rng)
            
            # (NHWC) -> (NCHW)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            
            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(timestep_rng, (bsz,), 0, noise_scheduler.config.num_train_timesteps)

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(
                batch["input_ids"],
                params=text_encoder_params,
                train=False,
            )[0]

            # Predict the noise residual and compute loss
            unet_outputs = unet.apply({"params": params}, noisy_latents, timesteps, encoder_hidden_states, train=True)
            noise_pred = unet_outputs.sample
            
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            loss = (target - noise_pred) ** 2
            loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics, new_train_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = jax_utils.replicate(state)
    text_encoder_params = jax_utils.replicate(text_encoder.params)
    vae_params = jax_utils.replicate(vae_params)

    # Train!
    global_step = 0
    epochs = tqdm(range(config.lightning.max_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ======================== Training ================================

        train_metrics = []

        steps_per_epoch = len(dataset) // total_train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for batch in dataloader:
            batch = shard(batch)
            state, train_metric, train_rngs = p_train_step(state, vae_params, batch, train_rngs)
            print(train_metric)
            train_metrics.append(train_metric)
            train_step_progress_bar.update(1)
            global_step += 1

        train_metric = jax_utils.unreplicate(train_metrics)
        train_step_progress_bar.close()
        epochs.write(f"Epoch... ({epoch + 1}/{config.lightning.max_epochs} | Loss: {train_metrics})")

    # Create the pipeline using using the trained modules and save it.
    if jax.process_index() == 0:
        noise_scheduler = FlaxDDPMScheduler.from_pretrained(config.trainer.model_path, subfolder="scheduler")
        pipeline = FlaxStableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
        )

        pipeline.save_pretrained(
            config.trainer.model_path + "_trained",
            params={
                "text_encoder": get_params_to_save(text_encoder_params),
                "vae": get_params_to_save(vae_params),
                "unet": get_params_to_save(state.params),
            },
        )

        if config.monitor.huggingface_repo != "":
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)


if __name__ == "__main__":
    main()