# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math, os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import functools
import xformers.ops

from einops import rearrange
from tqdm.auto import tqdm
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import DPMSolverMultistepScheduler, AutoencoderKL

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        input_size,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        flatten=True,
        bias=True,
        interp_scale=2,
    ):
        super().__init__()
        height = width = input_size
        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interp_scale
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, int(num_patches**0.5), base_size=self.base_size, interpolation_scale=self.interpolation_scale
        )
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)
        
    @functools.cache
    def get_pos_embed(self, latent_height, latent_width):
        height, width = latent_height // self.patch_size, latent_width // self.patch_size
        if self.height != height or self.width != width:
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=self.pos_embed.shape[-1],
                grid_size=(height, width),
                base_size=self.base_size,
                interpolation_scale=self.interpolation_scale,
            )
            pos_embed = torch.from_numpy(pos_embed)
            pos_embed = pos_embed.float().unsqueeze(0)
        else:
            pos_embed = self.pos_embed
        return pos_embed

    def forward(self, latent, pos_embed):
        # height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC

        # Interpolate positional embeddings if needed.
        # if precomputed_pos_embed is not None:
        #     pos_embed = precomputed_pos_embed
        # elif self.height != height or self.width != width:
        #     pos_embed = get_2d_sincos_pos_embed(
        #         embed_dim=self.pos_embed.shape[-1],
        #         grid_size=(height, width),
        #         base_size=self.base_size,
        #         interpolation_scale=self.interpolation_scale,
        #     )
        #     pos_embed = torch.from_numpy(pos_embed)
        #     pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
        # else:
        #     pos_embed = self.pos_embed
        return (latent + pos_embed).to(latent.dtype)
    

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)
        x = xformers.ops.memory_efficient_attention(q, k, v)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self, 
        d_model, 
        num_heads
    ):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model*2)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape
        
        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, -1, C)
        x = self.proj(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000.0):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(dtype))
        return t_emb


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate='tanh'), token_num=120):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer)
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def forward(self, caption):
        caption = self.y_proj(caption)
        return caption
    
    
class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs, dtype):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs//s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size)
        s_emb = self.mlp(s_freq.to(dtype))
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=lambda: nn.GELU(approximate="tanh"))
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        # self.xattn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, t, y, mask):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
            
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C)
        x = x + self.cross_attn(x, y, mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
def ckpt_wrapper(module, *inputs):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs
    return torch.utils.checkpoint.checkpoint(ckpt_forward, *inputs, use_reentrant=False)


#############################################################################
#                                 Core DiT Model                                #
#################################################################################
class DiTModel(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        caption_channels=4096,
        learn_sigma=True,
        interpolation_scale=2.,
        max_token_length=120,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.base_size = input_size // self.patch_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, interp_scale=interpolation_scale)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.csize_embedder = SizeEmbedder(hidden_size//3)  # c_size embed
        self.ar_embedder = SizeEmbedder(hidden_size//3)     # aspect ratio embed
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        
        # adaln single fn (= adaln_single.linear)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels, 
            hidden_size=hidden_size, 
            uncond_prob=class_dropout_prob, 
            act_layer=approx_gelu,
            token_num=max_token_length,
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size
        # h = w = int(x.shape[1] ** 0.5)
        
        h, w = self.h // p, self.w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def forward(
        self, x, t, y, 
        mask=None,
        c_size=None,
        ar=None,
        pos_embed=None,
    ):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """ 
        if pos_embed is None or c_size is None or ar is None:
            kw = get_model_kwargs(x, self)
            c_size, ar, pos_embed = kw["c_size"], kw["ar"], kw["pos_embed"]
        
        input_dtype = x.dtype
        bsz, _, self.h, self.w = x.shape
        
        x = self.x_embedder(x, pos_embed=pos_embed)  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t, dtype=input_dtype)  # (N, D)
        csize = self.csize_embedder(c_size, bsz, dtype=input_dtype)  # (N, D)
        ar = self.ar_embedder(ar, bsz, dtype=input_dtype)  # (N, D)
        
        temb = t + torch.cat([csize, ar], dim=1)
        t = self.t_block(temb)  # (N, 6, D) for adaLN-single
        y = self.y_embedder(y)  # (N, 1, L, D)
        
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).long().tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        for block in self.blocks:
            if self.training:
                x = ckpt_wrapper(block, x, t, y, y_lens)  # (N, T, D)
            else:
                x = block(x, t, y, y_lens)
                          
        x = self.final_layer(x, temb)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

# Sine/Cosine Positional Embedding Functions 
def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

@functools.cache
def get_cached_micro_condition(h, w):
    c_size = torch.tensor([8 * h, 8 * w])
    ar = torch.tensor([c_size[0] / c_size[1]]) # aspect ratio
    c_size = c_size[None]
    return c_size, ar

@torch.no_grad()
def get_model_kwargs(latents, model):
    h, w = latents.shape[-2:]
    c_size, ar = get_cached_micro_condition(h, w)
    pos_embed = model.x_embedder.get_pos_embed(h, w)
    return {
        "c_size": c_size.clone().to(latents.device),
        "ar": ar.clone().to(latents.device),
        "pos_embed": pos_embed.clone().to(latents.device),
    }

#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    print(f"Building DiT-XL-2 alpha model with kwargs: {kwargs}")
    model = DiTModel(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
    return model

@torch.no_grad()    
def sample(model, vae, prompt, tokenizer: T5Tokenizer, text_encoder: T5EncoderModel, \
    negative_prompt, size=(1024,1024), steps=20, guidance_scale=7.5, generator=None, \
    max_token_length=120, device="cuda", scheduler=None):
    
    assert type(prompt) == list, "Prompt must be a list of strings"
    if len(negative_prompt) != len(prompt):
        negative_prompt = [negative_prompt] * len(prompt)
        
    prompt = [prompt.replace("_", " ").lower() for prompt in prompt]
    negative_prompt = [negative_prompt.replace("_", " ").lower() for negative_prompt in negative_prompt]
    
    # encode prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_token_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ) 
    prompt_attention_mask = text_inputs.attention_mask.to(device)
    prompt_embeds = text_encoder(
        input_ids=text_inputs.input_ids.to(device), 
        attention_mask=prompt_attention_mask,
        return_dict=True
    )['last_hidden_state'] 
    
    # encode negative prompt
    uncond_input = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=max_token_length,
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    negative_prompt_attention_mask = uncond_input.attention_mask.to(device)
    negative_prompt_embeds = text_encoder(
        input_ids=uncond_input.input_ids.to(device), 
        attention_mask=negative_prompt_attention_mask,
        return_dict=True
    )['last_hidden_state']
    
    scheduler = DPMSolverMultistepScheduler() if scheduler is None else scheduler
    model_dtype = next(model.parameters()).dtype
    vae.to(model_dtype)
    
    latents_shape = (len(prompt), 4, size[0] // 8, size[1] // 8)
    latents = torch.randn(latents_shape, generator=generator, dtype=torch.float16).to(device)
    latents = latents * scheduler.init_noise_sigma
        
    scheduler.set_timesteps(steps)
    timesteps = scheduler.timesteps
    num_latent_input = 2
    
    extra_kwargs = {
        "y": torch.cat([negative_prompt_embeds, prompt_embeds]).to(device).to(model_dtype), 
        "mask": torch.cat([negative_prompt_attention_mask, prompt_attention_mask]).to(device).to(model_dtype),
        **get_model_kwargs(latents, model),
    }        
    for i, t in tqdm(enumerate(timesteps), total=steps, desc="Sampling", leave=False):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = model(
            x=latent_model_input, 
            t=torch.asarray([t] * latent_model_input.shape[0]).to(device), 
            **extra_kwargs
        )
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # compute the previous noisy sample x_t -> x_t-1
        noise_pred  = noise_pred.chunk(2, dim=1)[0]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    with torch.autocast('cuda', enabled=False):
        decoded = vae.decode(latents / vae.config.scaling_factor).sample
        
    image = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0).cpu().float()
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()

    image = (image * 255).round().astype("uint8")
    image = [Image.fromarray(im) for im in image]
    return image


if __name__ == '__main__':    
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # torch.set_float32_matmul_precision('medium')

    seed = int(184371982347)
    generator = torch.Generator().manual_seed(seed)
    
    text_encoder = T5EncoderModel.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        use_safetensors=True, 
        subfolder="text_encoder"
    )
    tokenizer = T5Tokenizer.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", legacy=False, subfolder="tokenizer")
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").cuda()
    model = DiT_XL_2(input_size=1024//8, interpolation_scale=2.)
    model.to("cuda:0", dtype=torch.float16, memory_format=torch.channels_last).eval()
    state_dict = torch.hub.load_state_dict_from_url("https://huggingface.co/datasets/nyanko7/tmp-public/resolve/main/PixArt-XL-2-1024-MS.pth", map_location="cuda:0")["state_dict"]            
    
    print("Loading weights from DiT-XL-2-1024-MS.pth")
    if 'pos_embed' in state_dict:
        del state_dict['pos_embed']
    result = model.load_state_dict(state_dict)
    
    # model = torch.compile(model, mode="max-autotune", dynamic=True)
    img = sample(
        model=model, 
        vae=vae, 
        prompt=["flower, rose, solo, blue_flower, blue_rose, blue_eyes, long_hair, looking_at_viewer, petals, sitting, white_hair, bangs, bow, hair_bow, dress, blush, long_sleeves, sailor_collar, water, white_sailor_collar, blunt_bangs, window"],
        negative_prompt="", 
        tokenizer=tokenizer, 
        text_encoder=text_encoder, 
        generator=generator,
        size=(1024, 1024), 
        steps=16, 
        guidance_scale=5
    )
    img[0].save("sample.png")
