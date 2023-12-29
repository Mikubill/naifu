import lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
from torch_ema import ExponentialMovingAverage

from lib.sgm import GeneralConditioner
from lib.wrappers import AutoencoderKLWrapper, UnetWrapper
from lib.utils import load_torch_file, rank_zero_print

from diffusers import EulerDiscreteScheduler, DDPMScheduler
from lightning.pytorch.utilities import rank_zero_only


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def prepare_scheduler_for_custom_training(noise_scheduler, device):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)

def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler):
    # fix beta: zero terminal SNR
    print(f"fix noise scheduler betas: https://arxiv.org/abs/2305.08891")

    def enforce_zero_terminal_snr(betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas

    betas = noise_scheduler.betas
    betas = enforce_zero_terminal_snr(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # print("original:", noise_scheduler.betas)
    # print("fixed:", betas)

    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alphas_cumprod = alphas_cumprod

def apply_snr_weight(loss, timesteps, noise_scheduler, gamma, v_prediction=False):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    loss = loss * snr_weight
    return loss

def scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler):
    scale = get_snr_scale(timesteps, noise_scheduler)
    loss = loss * scale
    return loss

def get_snr_scale(timesteps, noise_scheduler):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(
        snr_t, torch.ones_like(snr_t) * 1000
    )  # if timestep is 0, snr_t is inf, so limit it to 1000
    scale = snr_t / (snr_t + 1)
    # # show debug info
    # print(f"timesteps: {timesteps}, snr_t: {snr_t}, scale: {scale}")
    return scale

def add_v_prediction_like_loss(loss, timesteps, noise_scheduler, v_pred_like_loss):
    scale = get_snr_scale(timesteps, noise_scheduler)
    # print(f"add v-prediction like loss: {v_pred_like_loss}, scale: {scale}, loss: {loss}, time: {timesteps}")
    loss = loss + loss / scale * v_pred_like_loss
    return loss

def apply_debiased_estimation(loss, timesteps, noise_scheduler):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(
        snr_t, torch.ones_like(snr_t) * 1000
    )  # if timestep is 0, snr_t is inf, so limit it to 1000
    weight = 1 / torch.sqrt(snr_t)
    loss = weight * loss
    return loss


# define the LightningModule
class StableDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config, device):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.target_device = device
        self.init_model()

    def init_model(self):
        config = self.config
        sd = load_torch_file(self.model_path)
        key_name_sd_xl_refiner = ("conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias")

        config_file = Path(self.model_path).with_suffix(".yaml")
        if not config_file.is_file():
            config_file = "lib/model_configs/sd_xl_base.yaml"
            if key_name_sd_xl_refiner in sd:
                config_file = "lib/model_configs/sd_xl_refiner.yaml"

        self.model_config = OmegaConf.load(config_file)
        model_params = self.model_config.model.params

        for item in model_params.conditioner_config.params.emb_models:
            item["target"] = item["target"].replace("modules.", "")
            item["target"] = "lib." + item["target"]

        if self.config.trainer.use_xformers:
            model_params.network_config.params.spatial_transformer_attn_type = ("softmax-xformers")
            model_params.first_stage_config.params.ddconfig.attn_type = ("vanilla-xformers")

        encoder = AutoencoderKLWrapper(**model_params.first_stage_config.params).eval()
        encoder.train = disabled_train
        encoder.requires_grad_(False)
        for param in encoder.parameters():
            param.requires_grad = False

        self.first_stage_model = encoder
        self.scale_factor = model_params.scale_factor

        self.model = UnetWrapper(model_params.network_config.params)
        for conditioner in model_params.conditioner_config.params.emb_models:
            if "CLIPEmbedder" not in conditioner.target:
                continue
            conditioner.params["max_length"] = (self.config.dataset.get("max_token_length", 75) + 2)

        self.conditioner = GeneralConditioner(**model_params.conditioner_config.params)
        self.to(self.target_device)
        
        rank_zero_print(f"Loading model from {self.model_path}")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            rank_zero_print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            rank_zero_print(f"Unexpected Keys: {unexpected}")

        self.cast_dtype = torch.float32
        self.offset_noise_level = self.config.trainer.get("offset_noise_val")
        self.extra_config = self.config.get("extra", None)
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )
        prepare_scheduler_for_custom_training(self.noise_scheduler, self.device)
        
        advanced = self.config.get("advanced", {})
        if advanced.get("zero_terminal_snr", None):
            fix_noise_scheduler_betas_for_zero_terminal_snr(self.noise_scheduler)

        if config.trainer.use_ema:
            self.model_ema = ExponentialMovingAverage(
                self.model.parameters(), decay=0.9999
            )
            rank_zero_print(f"EMA is enabled with decay {self.model_ema.decay}")

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", enabled=False):
            out = self.first_stage_model.decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", enabled=False):
            z = self.first_stage_model.encode(x).sample()
        z = self.scale_factor * z
        return z

    @torch.inference_mode()
    @rank_zero_only
    def sample(
        self,
        prompt,
        negative_prompt,
        generator=None,
        size=(1024, 1024),
        steps=20,
        guidance_scale=7.5,
    ):
        self.model.eval()
        self.conditioner.cuda()
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

        # first construct batch
        model_dtype = next(self.model.parameters()).dtype
        prompts_batch = {
            "target_size_as_tuple": torch.stack([torch.asarray(size)]).cuda(),
            "original_size_as_tuple": torch.stack([torch.asarray(size)]).cuda(),
            "crop_coords_top_left": torch.stack([torch.asarray((0, 0))]).cuda(),
        }
        prompts_batch["prompts"] = prompt
        cond = self.conditioner(prompts_batch)
        prompts_batch["prompts"] = negative_prompt
        uncond = self.conditioner(prompts_batch)
        cond = {
            "crossattn": torch.cat([uncond["crossattn"], cond["crossattn"]], dim=0).cuda().to(model_dtype),
            "vector": torch.cat([uncond["vector"], cond["vector"]], dim=0).cuda().to(model_dtype),
        }

        latents_shape = (1, 4, size[0] // 8, size[1] // 8)
        latents = torch.randn(latents_shape, generator=generator, dtype=torch.float32)
        latents = latents * scheduler.init_noise_sigma

        scheduler.set_timesteps(steps)
        timesteps = scheduler.timesteps
        num_latent_input = 2
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.cuda().to(model_dtype)

            noise_pred = self.model(latent_model_input, torch.asarray([t]).cuda(), cond)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents.cuda()).prev_sample

        self.first_stage_model.cuda()
        image = (torch.clamp((self.decode_first_stage(latents) + 1.0) / 2.0, min=0.0, max=1.0).cpu().float())
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]

        self.model.train()
        return image

    def forward(self, batch):
        advanced = self.config.get("advanced", {})
        if not batch["is_latent"]:
            self.first_stage_model.to(self.target_device)
            latents = self.encode_first_stage(batch["pixels"].to(self.first_stage_model.dtype))
            if torch.any(torch.isnan(latents)):
                rank_zero_print("NaN found in latents, replacing with zeros")
                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)

            del batch["images"]
        else:
            self.first_stage_model.cpu()
            latents = batch["pixels"]

        if "conds" not in batch.keys():
            self.conditioner.to(self.target_device)
            cond = self.conditioner(batch)
        else:
            self.conditioner.cpu()
            cond = batch["conds"]

        model_dtype = next(self.model.parameters()).dtype
        cond = {k: v.to(model_dtype) for k, v in cond.items()}

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, dtype=model_dtype)
        if advanced.get("offset_noise"):
            offset = torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
            noise = torch.randn_like(latents) + float(advanced.get("offset_noise_val")) * offset

        # https://arxiv.org/abs/2301.11706
        if advanced.get("input_perturbation"):
            noise = noise + float(advanced.get("input_perturbation_val")) * torch.randn_like(noise)

        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            advanced.get("timestep_start", 0),
            advanced.get("timestep_end", 1000),
            (bsz,),
            dtype=torch.int64,
            device=latents.device,
        )

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.model(noisy_latents, timesteps, cond)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)

        min_snr_gamma = advanced.get("min_snr_val", None)
        scale_pred = advanced.get("scale_v_pred_loss_like_noise_pred", False)
        vl_loss = advanced.get("v_pred_like_loss", False)
        de_loss = advanced.get("debiased_estimation_loss", False)
        if min_snr_gamma or scale_pred or vl_loss or de_loss:
            # do not mean over batch dimension for snr weight or scale v-pred loss
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
            loss = loss.mean([1, 2, 3])
            is_v = self.noise_scheduler.config.prediction_type == "v_prediction"

            if min_snr_gamma:
                loss = apply_snr_weight(loss, timesteps, self.noise_scheduler, min_snr_gamma, is_v)
            if scale_pred:
                loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, self.noise_scheduler)
            if vl_loss:
                loss = add_v_prediction_like_loss(loss, timesteps, self.noise_scheduler, vl_loss)
            if de_loss:
                loss = apply_debiased_estimation(loss, timesteps, self.noise_scheduler)

            loss = loss.mean()  # mean over batch dimension
        else:
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        return loss
