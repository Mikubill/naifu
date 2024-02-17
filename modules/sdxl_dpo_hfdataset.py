import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from datasets import load_dataset
from torchvision import transforms

import io
import copy
import torch.nn.functional as F
from PIL import Image
from common.utils import rank_zero_print, get_class
from modules.sdxl_model_diffusers import StableDiffusionModel
from lightning.pytorch.utilities.model_summary import ModelSummary



def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = SupervisedFineTune(
        model_path=model_path, config=config, device=fabric.device
    )

    dataset, dataloader = setup_hf_dataloader(config)
    params_to_optim = [{"params": model.unet.parameters()}]
    # params_to_optim = [{'params': model.model.parameters()}]
    if config.advanced.get("train_text_encoder_1"):
        lr = config.advanced.get("text_encoder_1_lr", config.optimizer.params.lr)
        params_to_optim.append({"params": model.text_encoder_1.parameters(), "lr": lr})

    if config.advanced.get("train_text_encoder_2"):
        lr = config.advanced.get("text_encoder_2_lr", config.optimizer.params.lr)
        params_to_optim.append({"params": model.text_encoder_2.parameters(), "lr": lr})

    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(params_to_optim, **optim_param)
    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(
            optimizer, **config.scheduler.params
        )

    model.vae.to(torch.float32)
    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


def setup_hf_dataloader(config):
    dataset = load_dataset(
        path=config.dataset.name,
        name=None,
        cache_dir=config.dataset.get("cache_dir", None),
    )
    reso = config.trainer.resolution
    interp = transforms.InterpolationMode.BILINEAR
    train_transforms = transforms.Compose(
        [
            transforms.Resize(reso, interpolation=interp),
            transforms.CenterCrop(reso),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        all_pixel_values = []
        for col_name in ["jpg_0", "jpg_1"]:
            images = [
                Image.open(io.BytesIO(im_bytes)).convert("RGB")
                for im_bytes in examples[col_name]
            ]
            original_sizes = [(image.height, image.width) for image in images]
            pixel_values = [train_transforms(image) for image in images]
            all_pixel_values.append(pixel_values)

        # Double on channel dim, jpg_y then jpg_w
        im_tup_iterator = zip(*all_pixel_values)
        combined_pixel_values = []
        for im_tup, label_0 in zip(im_tup_iterator, examples["label_0"]):
            if label_0 == 0:
                im_tup = im_tup[::-1]
            combined_im = torch.cat(im_tup, dim=0)  # no batch dim
            combined_pixel_values.append(combined_im)

        examples.update(
            {
                "original_size_as_tuple": original_sizes,
                "crop_coords_top_left": [(0, 0)] * len(original_sizes),
                "target_size_as_tuple": [(reso, reso)] * len(original_sizes),
                "prompts": examples["caption"],
                "pixels": combined_pixel_values,
            }
        )
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([e["pixels"] for e in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        orig_tup = [torch.tensor(e["original_size_as_tuple"]) for e in examples]
        crop_tup = [torch.tensor(e["crop_coords_top_left"]) for e in examples]
        target_tup = [torch.tensor(e["target_size_as_tuple"]) for e in examples]
        return_d = {
            "pixels": pixel_values,
            "original_size_as_tuple": torch.stack(orig_tup),
            "crop_coords_top_left": torch.stack(crop_tup),
            "target_size_as_tuple": torch.stack(target_tup),
            "prompts": [e["prompts"] for e in examples],
        }
        return return_d

    train_dataset = (
        dataset[config.dataset.dataset_split]
        .shuffle(seed=config.trainer.seed)
        .with_transform(preprocess_train)
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=config.trainer.batch_size,
        num_workers=4,
        drop_last=True,
    )
    return train_dataset, dataloader


class SupervisedFineTune(StableDiffusionModel):
    def init_model(self):
        super().init_model()
        # clone frozen unet as ref
        self.unet_ref = copy.deepcopy(self.unet)
        self.unet_ref.eval().requires_grad_(False)
        # since we're in dpo, unet_ref in 16bit is ok
        self.unet_ref.to(torch.float16)
        
    def forward(self, batch):
        advanced = self.config.get("advanced", {})
        self.vae.to(self.target_device)

        feed_pixel_values = torch.cat(batch["pixels"].chunk(2, dim=1))
        latents = self.encode_pixels(feed_pixel_values)
        if torch.any(torch.isnan(latents)):
            rank_zero_print("NaN found in latents, replacing with zeros")
            latents = torch.where(
                torch.isnan(latents), torch.zeros_like(latents), latents
            )

        self.text_encoder_1.to(self.target_device)
        self.text_encoder_2.to(self.target_device)

        model_dtype = next(self.unet.parameters()).dtype
        text_embedding, pooled = self.encode_prompt(batch)
        text_embedding = text_embedding.to(model_dtype).repeat(2, 1, 1)
        pooled = pooled.to(model_dtype).repeat(2, 1)

        bs = batch["original_size_as_tuple"]
        bc = batch["crop_coords_top_left"]
        bt = batch["target_size_as_tuple"]
        add_time_ids = torch.cat(
            [self.compute_time_ids(s, c, t) for s, c, t in zip(bs, bc, bt)]
        ).repeat(2, 1)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, dtype=model_dtype).chunk(2)[0].repeat(2, 1, 1, 1)
        bsz = latents.shape[0] // 2

        # Sample a random timestep for each image
        timesteps = torch.randint(
            advanced.get("timestep_start", 0),
            advanced.get("timestep_end", 1000),
            (bsz,),
            dtype=torch.int64,
            device=latents.device,
        ).repeat(2)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=text_embedding,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
        ).sample

        # Get the target for loss depending on the prediction type
        is_v = advanced.get("v_parameterization", False)
        target = (
            noise
            if not is_v
            else self.noise_scheduler.get_velocity(latents, noise, timesteps)
        )

        # Compute losses.
        model_losses = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
        model_losses_w, model_losses_l = model_losses.chunk(2)

        # For logging
        raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
        model_diff = model_losses_w - model_losses_l  # These are both LBS (as is t)

        with torch.no_grad():
            ref_preds = self.unet_ref(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embedding,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
            ).sample

            ref_loss = F.mse_loss(ref_preds.float(), target.float(), reduction="none")
            ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))

            ref_losses_w, ref_losses_l = ref_loss.chunk(2)
            ref_diff = ref_losses_w - ref_losses_l
            raw_ref_loss = ref_loss.mean()

        scale_term = -0.5 * self.config.beta_dpo
        inside_term = scale_term * (model_diff - ref_diff)
        loss = -1 * F.logsigmoid(inside_term).mean()

        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
        implicit_acc += 0.5 * (inside_term == 0).sum().float() / inside_term.size(0)

        self.fabric.log_dict(
            {
                "loss": loss.detach().item(),
                "raw_model_loss": raw_model_loss.detach().item(),
                "ref_loss": raw_ref_loss.detach().item(),
                "implicit_acc": implicit_acc.detach().item(),
            }
        )
        return loss
