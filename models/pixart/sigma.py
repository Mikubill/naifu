# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from .alpha import *

class AttentionKVCompress(nn.Module):
    """Multi-head Attention block with KV token compression and qk norm."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        sampling='conv',
        sr_ratio=1,
        qk_norm=False,
        **block_kwargs,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.sampling = sampling    # ['conv', 'ave', 'uniform', 'uniform_every']
        self.sr_ratio = sr_ratio
        
        if sr_ratio > 1 and sampling == 'conv':
            # Avg Conv Init.
            self.sr = nn.Conv2d(dim, dim, groups=dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr.weight.data.fill_(1/sr_ratio**2)
            self.sr.bias.data.zero_()
            self.norm = nn.LayerNorm(dim)
            
        if qk_norm:
            self.q_norm = nn.LayerNorm(dim)
            self.k_norm = nn.LayerNorm(dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def downsample_2d(self, tensor, H, W, scale_factor, sampling=None):
        if sampling is None or scale_factor == 1:
            return tensor
        B, N, C = tensor.shape

        if sampling == 'uniform_every':
            return tensor[:, ::scale_factor], int(N // scale_factor)

        tensor = tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
        new_H, new_W = int(H / scale_factor), int(W / scale_factor)
        new_N = new_H * new_W

        if sampling == 'ave':
            tensor = F.interpolate(
                tensor, scale_factor=1 / scale_factor, mode='nearest'
            ).permute(0, 2, 3, 1)
        elif sampling == 'uniform':
            tensor = tensor[:, :, ::scale_factor, ::scale_factor].permute(0, 2, 3, 1)
        elif sampling == 'conv':
            tensor = self.sr(tensor).reshape(B, C, -1).permute(0, 2, 1)
            tensor = self.norm(tensor)
        else:
            raise ValueError

        return tensor.reshape(B, new_N, C).contiguous(), new_N

    def forward(self, x, mask=None, HW=None, block_id=None):
        B, N, C = x.shape
        new_N = N
        if HW is None:
            H = W = int(N ** 0.5)
        else:
            H, W = HW
        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype
        q = self.q_norm(q)
        k = self.k_norm(k)

        # KV compression
        if self.sr_ratio > 1:
            k, new_N = self.downsample_2d(k, H, W, self.sr_ratio, sampling=self.sampling)
            v, new_N = self.downsample_2d(v, H, W, self.sr_ratio, sampling=self.sampling)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)

        use_fp32_attention = getattr(self, 'fp32_attention', False)     # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        attn_bias = None
        if mask is not None:
            attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float('-inf'))
        x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)

        x = x.view(B, N, C)
        x = self.proj(x)
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
        kvconfig={},
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

        for block in self.blocks:
            if self.training:
                x = ckpt_wrapper(block, x, t0, y, y_lens)  # (N, T, D)
            else:
                x = block(x, t0, y, y_lens)
                          
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
        # kvconfig={
        #     'sampling': 'conv',
        #     'scale_factor': 2,
        #     'kv_compress_layer': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        # }, 
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