import torch
import copy
import os
import lightning as pl
import numpy as np
from omegaconf import OmegaConf
from common.utils import get_class
from common.logging import logger
from modules.sdxl_model_diffusers import StableDiffusionModel
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.utilities import rank_zero_only
from diffusers import UNet2DConditionModel, LCMScheduler, StableDiffusionXLPipeline


def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = SupervisedFineTune(
        model_path=model_path, config=config, device=fabric.device
    )
    dataset_class = get_class(config.dataset.get("name", "data.AspectRatioDataset"))
    dataset = dataset_class(
        batch_size=config.trainer.batch_size,
        rank=fabric.global_rank,
        dtype=torch.float32,
        **config.dataset,
    )
    dataloader = dataset.init_dataloader()

    params_to_optim = [{"params": model.lcm_unet.parameters()}]
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

    model.model, optimizer = fabric.setup(model.model, optimizer)
    if config.advanced.get("train_text_encoder_1"):
        model.text_encoder_1 = fabric.setup(model.text_encoder_1)
    if config.advanced.get("train_text_encoder_2"):
        model.text_encoder_2 = fabric.setup(model.text_encoder_2)
    
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# Compare LCMScheduler.step, Step 4
def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def get_w_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
    Args:
    timesteps: torch.Tensor: generate embedding vectors at these timesteps
    embedding_dim: int: dimension of the embeddings to generate
    dtype: data type of the generated embeddings
    Returns:
    embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w.cpu() * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (
            np.arange(1, ddim_timesteps + 1) * step_ratio
        ).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


class SupervisedFineTune(StableDiffusionModel):
    def init_model(self):
        super().init_model()
        # clone frozen unet as ref
        teacher_unet_config = copy.deepcopy(self.unet.config)
        teacher_unet_config["time_cond_proj_dim"] = 256
        self.lcm_unet = UNet2DConditionModel(**teacher_unet_config)
        self.lcm_unet.load_state_dict(self.unet.state_dict(), strict=False)
        self.lcm_unet.train().requires_grad_(True)

        self.unet.eval().requires_grad_(False)
        self.unet.to(torch.float16)
        # Initialize the DDIM ODE solver for distillation.
        self.solver = DDIMSolver(
            self.noise_scheduler.alphas_cumprod.numpy(),
            timesteps=self.noise_scheduler.config.num_train_timesteps,
            ddim_timesteps=50,
        )
        alpha_cumprods = self.noise_scheduler.alphas_cumprod
        self.alphas = torch.sqrt(alpha_cumprods).to(self.target_device)
        self.sigmas = torch.sqrt(1 - alpha_cumprods).to(self.target_device)

    def forward(self, batch):
        advanced = self.config.get("advanced", {})
        if not batch["is_latent"]:
            self.vae.to(self.target_device)
            latents = self.encode_pixels(batch["pixels"])
            if torch.any(torch.isnan(latents)):
                logger.info("NaN found in latents, replacing with zeros")
                latents = torch.where(
                    torch.isnan(latents), torch.zeros_like(latents), latents
                )
        else:
            self.vae.cpu()
            latents = self._normliaze(batch["pixels"])

        self.text_encoder_1.to(self.target_device)
        self.text_encoder_2.to(self.target_device)

        model_dtype = next(self.unet.parameters()).dtype
        text_embedding, pooled = self.encode_prompt(batch)
        text_embedding = text_embedding.to(model_dtype)
        pooled = pooled.to(model_dtype)

        bs = batch["original_size_as_tuple"]
        bc = batch["crop_coords_top_left"]
        bt = batch["target_size_as_tuple"]
        add_time_ids = torch.cat(
            [self.compute_time_ids(s, c, t) for s, c, t in zip(bs, bc, bt)]
        )

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, dtype=model_dtype)
        bsz = latents.shape[0]
        num_ddim_timesteps = 50

        self.solver.to(latents.device)
        topk = self.noise_scheduler.config.num_train_timesteps // num_ddim_timesteps
        index = torch.randint(
            0, num_ddim_timesteps, (bsz,), device=latents.device
        ).long()

        start_timesteps = self.solver.ddim_timesteps[index]
        timesteps = start_timesteps - topk
        timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

        c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
        c_skip_start, c_out_start = [
            append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]
        ]
        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        # w_max, w_min = 14, 2
        # w = (w_max - w_min) * torch.rand((bsz,)) + w_min
        w = torch.tensor([6.0] * bsz)

        # guidance_scale = torch.randint(2, 14, (bsz,), device=latents.device)
        w_embedding = get_w_embedding(w, embedding_dim=256).to(latents.device)
        w = w.reshape(bsz, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.lcm_unet(
            sample=noisy_latents,
            timestep=timesteps,
            timestep_cond=w_embedding,
            encoder_hidden_states=text_embedding,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
        ).sample

        pred_x_0 = predicted_origin(
            noise_pred,
            start_timesteps,
            noisy_latents,
            self.noise_scheduler.config.prediction_type,
            self.alphas,
            self.sigmas,
        )
        model_pred = c_skip_start * noisy_latents + c_out_start * pred_x_0

        with torch.no_grad(), torch.autocast("cuda"):
            # teacher unet (frozen)
            cond_teacher_output = self.unet(
                noisy_latents,
                start_timesteps,
                encoder_hidden_states=text_embedding,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
            ).sample
            cond_pred_x0 = predicted_origin(
                cond_teacher_output,
                start_timesteps,
                noisy_latents,
                self.noise_scheduler.config.prediction_type,
                self.alphas,
                self.sigmas,
            )
            # Get teacher model prediction on noisy_latents and unconditional embedding
            # teacher unet (frozen)
            uncond_teacher_output = self.unet(
                noisy_latents,
                start_timesteps,
                encoder_hidden_states=text_embedding,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
            ).sample
            uncond_pred_x0 = predicted_origin(
                uncond_teacher_output,
                start_timesteps,
                noisy_latents,
                self.noise_scheduler.config.prediction_type,
                self.alphas,
                self.sigmas,
            )

            # 20.4.11. Perform "CFG" to get x_prev estimate (using the LCM paper's CFG formulation)
            pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
            pred_noise = cond_teacher_output + w * (
                cond_teacher_output - uncond_teacher_output
            )
            x_prev = self.solver.ddim_step(pred_x0, pred_noise, index)

        with torch.no_grad(), torch.autocast("cuda"):
            # algo.1 step.3 alternative - same as above
            # with self.ema.average_parameters():
            target_noise_pred = self.unet(
                x_prev.float(),
                timesteps,
                encoder_hidden_states=text_embedding,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
            ).sample
            pred_x_0 = predicted_origin(
                target_noise_pred,
                timesteps,
                x_prev,
                self.noise_scheduler.config.prediction_type,
                self.alphas,
                self.sigmas,
            )
            target = c_skip * x_prev + c_out * pred_x_0

        loss = torch.nn.functional.mse_loss(
            model_pred.float(), target.float(), reduction="mean"
        )
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        return loss

    @rank_zero_only
    def save_checkpoint(self, model_path, metadata):
        lcm_scheduler = LCMScheduler.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler"
        )
        pipeline = StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            unet=self.lcm_unet,
            scheduler=lcm_scheduler,
        )
        pipeline.save_pretrained(model_path)
        logger.info(f"Saved model to {model_path}")
