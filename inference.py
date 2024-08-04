import random
import torch
from diffusers import HunyuanDiTPipeline
from transformers import T5EncoderModel
import time
from loguru import logger
import gc
import sys

NEGATIVE_PROMPT = '错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺'
    
TEXT_ENCODER_CONF = {
    "negative_prompt": NEGATIVE_PROMPT,
    "prompt_embeds": None,
    "negative_prompt_embeds": None,
    "prompt_attention_mask": None,
    "negative_prompt_attention_mask": None,
    "max_sequence_length": 256,
    "text_encoder_index": 1,
}

def flush():
    gc.collect()
    torch.cuda.empty_cache()


class End2End(object):
    def __init__(self, model_id="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # ========================================================================
        self.default_negative_prompt = NEGATIVE_PROMPT
        logger.info("==================================================")
        logger.info(f"                Model is ready.                  ")
        logger.info("==================================================")

    def load_pipeline(self):
        self.pipeline= HunyuanDiTPipeline.from_pretrained(
            self.model_id,
            text_encoder=None,
            text_encoder_2=None,
            torch_dtype=torch.float16,
        ).to(self.device)

    
    def get_text_emb(self, prompts):
        with torch.no_grad():
            text_encoder_2 = T5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder_2",
            )
            encoder_pipeline = HunyuanDiTPipeline.from_pretrained(
                self.model_id, 
                text_encoder_2=text_encoder_2,
                transformer=None,
                vae=None,
                torch_dtype=torch.float16,
            )
            TEXT_ENCODER_CONF["negative_prompt"]=self.default_negative_prompt
            prompt_emb1 = encoder_pipeline.encode_prompt(prompts, negative_prompt=self.default_negative_prompt)
            prompt_emb2 = encoder_pipeline.encode_prompt(prompts, **TEXT_ENCODER_CONF)
            del text_encoder_2
            del encoder_pipeline
        flush()
        return prompt_emb1, prompt_emb2

    def predict(self,
                user_prompt,
                seed=None,
                enhanced_prompt=None,
                negative_prompt=None,
                infer_steps=50,
                guidance_scale=6,
                batch_size=1,
                ):
        # ========================================================================
        # Arguments: seed
        # ========================================================================
        if seed is None:
            seed = random.randint(0, 1_000_000)
        if not isinstance(seed, int):
            raise TypeError(f"`seed` must be an integer, but got {type(seed)}")
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if not isinstance(user_prompt, str):
            raise TypeError(f"`user_prompt` must be a string, but got {type(user_prompt)}")
        user_prompt = user_prompt.strip()
        prompt = user_prompt

        if enhanced_prompt is not None:
            if not isinstance(enhanced_prompt, str):
                raise TypeError(f"`enhanced_prompt` must be a string, but got {type(enhanced_prompt)}")
            enhanced_prompt = enhanced_prompt.strip()
            prompt = enhanced_prompt

        # negative prompt
        if negative_prompt is not None and negative_prompt != '':
            self.default_negative_prompt = negative_prompt
        if not isinstance(self.default_negative_prompt, str):
            raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")


        # ========================================================================
        
        logger.debug(f"""
                       prompt: {user_prompt}
              enhanced prompt: {enhanced_prompt}
                         seed: {seed}
              negative_prompt: {negative_prompt}
                   batch_size: {batch_size}
               guidance_scale: {guidance_scale}
                  infer_steps: {infer_steps}
        """)

        
        # get text embeding 
        flush()
        prompt_emb1, prompt_emb2 = self.get_text_emb(prompt)
        prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask = prompt_emb1
        prompt_embeds_2,negative_prompt_embeds_2,prompt_attention_mask_2,negative_prompt_attention_mask_2 = prompt_emb2
        # print(prompt_emb1, prompt_emb2)
        del prompt_emb1
        del prompt_emb2
        # get pipeline
        self.load_pipeline()
        
        print(prompt_embeds, prompt_embeds_2)
        samples = self.pipeline(
            prompt_embeds=prompt_embeds,
            prompt_embeds_2=prompt_embeds_2,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_2=negative_prompt_embeds_2,
            prompt_attention_mask=prompt_attention_mask,
            prompt_attention_mask_2=prompt_attention_mask_2,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            negative_prompt_attention_mask_2=negative_prompt_attention_mask_2,
            num_images_per_prompt=batch_size,
            guidance_scale=guidance_scale,
            num_inference_steps=infer_steps,
            generator=generator, 
            height=1280,
            width=768
        ).images[0]
        
        return {
            'images': samples,
            'seed': seed,
        }


if __name__ == "__main__":
    gen = End2End()
    seed = 42
    results = gen.predict(
        "best quality, 1girl, solo, loli, cat girl, silver hair ,blue eyes, flat chest, solo, beautiful detailed background, messy hair, long hair",
        seed=1,
        infer_steps=40,
        guidance_scale=6,
    )
    results['images'].save('./lite_image.png')
