import torch
import torch.utils.checkpoint
import lightning as pl

from pathlib import Path
from common.utils import rank_zero_print
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler, DDPMScheduler
from lightning.pytorch.utilities import rank_zero_only
from modules.utils import apply_zero_terminal_snr, cache_snr_values

# define the LightningModule
class StableDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config, device):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.target_device = device
        self.init_model()

    def init_model(self):
        trainer_cfg = self.config.trainer
        config = self.config
        advanced = config.get("advanced", {})
        
        rank_zero_print(f"Loading model from {self.model_path}")
        p = StableDiffusionPipeline
        if Path(self.model_path).is_dir():
            self.pipeline = pipeline = p.from_pretrained(self.model_path)
        else:
            self.pipeline = pipeline = p.from_single_file(self.model_path)
            
        self.vae, self.unet = pipeline.vae, pipeline.unet
        self.text_encoder, self.tokenizer = (
            pipeline.text_encoder,
            pipeline.tokenizer,
        )
        self.max_prompt_length = 225 + 2
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.scale_factor = 0.13025

        self.unet.enable_gradient_checkpointing()
        if trainer_cfg.use_xformers:
            self.unet.enable_xformers_memory_efficient_attention()

        if advanced.get("train_text_encoder"):
            self.text_encoder.requires_grad_(True)
            self.text_encoder.gradient_checkpointing_enable()

        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )
        if advanced.zero_terminal_snr:
            apply_zero_terminal_snr(self.noise_scheduler)
        cache_snr_values(self.noise_scheduler, self.target_device)

    def encode_prompts(self, prompts):            
        input_ids = self.tokenizer(prompts, padding="do_not_pad", truncation=True, max_length=225).input_ids 
        input_ids = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
            
        z = []
        if input_ids.shape[1] > 77:
            # todo: Handle end-of-sentence truncation
            while max(map(len, input_ids)) != 0:
                rem_tokens = [x[75:] for x in input_ids]
                tokens = []
                for j in range(len(input_ids)):
                    tokens.append(input_ids[j][:75] if len(input_ids[j]) > 0 else [self.tokenizer.eos_token_id] * 75)

                rebuild = [[self.tokenizer.bos_token_id] + list(x[:75]) + [self.tokenizer.eos_token_id] for x in tokens]
                if hasattr(torch, "asarray"):
                    z.append(torch.asarray(rebuild))
                else:
                    z.append(torch.IntTensor(rebuild))
                input_ids = rem_tokens
        else:
            z.append(input_ids)

        # Get the text embedding for conditioning
        encoder_hidden_states = None
        for tokens in z:
            state = self.text_encoder(tokens.to(self.device), output_hidden_states=True)
            state = self.text_encoder.text_model.final_layer_norm(state['hidden_states'][-self.config.trainer.clip_skip])
            encoder_hidden_states = state if encoder_hidden_states is None else torch.cat((encoder_hidden_states, state), axis=-2)

        return encoder_hidden_states

    @torch.inference_mode()
    @rank_zero_only
    def sample(
        self,
        prompt,
        negative_prompt="lowres, low quality, text, error, extra digit, cropped",
        generator=None,
        size=(512, 512),
        steps=20,
        guidance_scale=6.5,
    ):
        self.vae.to(self.target_device)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        pipeline = StableDiffusionPipeline(
            unet=self.unet,
            vae=self.vae,
            text_encoder_1=self.text_encoder_1,
            text_encoder_2=self.text_encoder_2,
            noise_scheduler=scheduler,
        )
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=size[0],
            height=size[1],
            steps=steps,
            generator=generator,
            guidance_scale=guidance_scale,
            return_dict=False,
        )[0]
        return image

    def save_checkpoint(self, model_path):
        self.pipeline.save_pretrained(model_path)
        rank_zero_print(f"Saved model to {model_path}")

    def forward(self, batch):
        raise NotImplementedError
