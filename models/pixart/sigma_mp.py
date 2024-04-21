# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------


import torch
import torch.distributed as dist

from typing import Optional
from torch.distributed import ProcessGroup

from .alpha import *
from .dist_ops import *


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

    
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, sr_ratio=1, sampling='conv', **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        kv_compress_config = {} # {'sr_ratio': sr_ratio, 'sampling': sampling}
        self.attn = Attention(dim=hidden_size, num_heads=num_heads, qkv_bias=True, **kv_compress_config)
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
        sequence_parallel_size: int = 1,
        sequence_parallel_group: Optional[ProcessGroup] = None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.base_size = input_size // self.patch_size
        
        self.sequence_parallel_size = sequence_parallel_size
        self.sequence_parallel_group = sequence_parallel_group

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, interp_scale=interpolation_scale)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
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
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                # sr_ratio=kvconfig['scale_factor'] if i in kvconfig['kv_compress_layer'] else 1,
                # sampling=kvconfig['sampling'],
            ) for i in range(depth)
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
        pos_embed=None,
        **kwargs,
    ):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """ 
        if pos_embed is None:
            kw = get_model_kwargs(x, self)
            pos_embed = kw["pos_embed"]
        
        input_dtype = x.dtype
        bsz, _, self.h, self.w = x.shape
        
        x = self.x_embedder(x, pos_embed=pos_embed)  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t, dtype=input_dtype)  # (N, D)
        
        t0 = self.t_block(t)  # (N, 6, D) for adaLN-single
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
            
        if self.sequence_parallel_size > 1:
            x = x.chunk(self.sequence_parallel_size, dim=1)[dist.get_rank(self.sequence_parallel_group)]

        for block in self.blocks:
            if self.training:
                x = ckpt_wrapper(block, x, t0, y, y_lens)  # (N, T, D)
            else:
                x = block(x, t0, y, y_lens)
                
        if self.sequence_parallel_size > 1:
            x = gather_forward_split_backward(x, dim=1, process_group=self.sequence_parallel_group)
                          
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x
    

def DiT_XL_2(**kwargs):
    print(f"Building DiT-XL-2 Sigma model with kwargs: {kwargs}")
    model = DiTModel(
        depth=28, 
        hidden_size=1152, 
        patch_size=2, 
        num_heads=16, 
        **kwargs
    )
    return model


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
    
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").cuda()
    model = DiT_XL_2(input_size=1024//8, interpolation_scale=2., max_token_length=300)
    model.to("cuda:0", dtype=torch.float16, memory_format=torch.channels_last).eval()
    state_dict = torch.hub.load_state_dict_from_url("https://huggingface.co/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-1024-MS.pth", map_location="cuda:0")["state_dict"]            
    
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
        guidance_scale=5,
        max_token_length=300
    )
    img[0].save("sample.png")