import functools
from typing import Union
import torch
import torch.utils.checkpoint
import numpy as np
from common.utils import load_torch_file
from common.logging import logger

def cache_snr_values(noise_scheduler, device):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)

def apply_zero_terminal_snr(noise_scheduler):
    # fix beta: zero terminal SNR
    # print(f"fix noise scheduler betas: https://arxiv.org/abs/2305.08891")

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
    

@torch.no_grad()
def calc_snr_weight(loss, timesteps, noise_scheduler, gamma, v_prediction=False):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    return snr_weight

def load_file_from_path(path, model, device, extra_keys={}):
    sd = load_torch_file(path, device)
    sd.update(extra_keys)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0:
        logger.info(f"Missing Keys: {missing}")
    if len(unexpected) > 0:
        logger.info(f"Unexpected Keys: {unexpected}")
        
def clip_process_input_ids(input_ids, tokenizer, tokenizer_max_length=225):
    if tokenizer_max_length > tokenizer.model_max_length:
        input_ids = input_ids.squeeze(0)
        iids_list = []
        for i in range(1, tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2): 
            ids_chunk = (
                input_ids[0].unsqueeze(0),  # BOS
                input_ids[i : i + tokenizer.model_max_length - 2],
                input_ids[-1].unsqueeze(0),
            )  # PAD or EOS
            ids_chunk = torch.cat(ids_chunk)

            # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
            # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
            if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                ids_chunk[-1] = tokenizer.eos_token_id
            # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
            if ids_chunk[1] == tokenizer.pad_token_id:
                ids_chunk[1] = tokenizer.eos_token_id

            iids_list.append(ids_chunk)

        input_ids = torch.stack(iids_list)  # 3,77
    return input_ids

def clip_get_hidden_states(
    input_ids,
    tokenizer,
    text_encoder,
    max_token_length=225,
    weight_dtype=None,
):
    # with no_token_padding, the length is not max length, return result immediately
    if input_ids.size()[-1] != tokenizer.model_max_length:
        return text_encoder(input_ids)[0]

    # input_ids: b,n,77
    b_size = input_ids.size(0)
    input_ids = input_ids.reshape((-1, tokenizer.model_max_length))  # batch_size*3, 77
    masks = (input_ids != tokenizer.pad_token_id).long()

    encoder_hidden_states = text_encoder(input_ids, attention_mask=masks)[0]

    # bs*3, 77, 768 or 1024
    encoder_hidden_states = encoder_hidden_states.reshape(
        (b_size, -1, encoder_hidden_states.shape[-1])
    )
    masks = masks.reshape((b_size, -1))
    input_ids = input_ids.reshape((b_size, -1))

    if max_token_length is not None:
        states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
        mask_list = [masks[:, 0].unsqueeze(1)]
        for i in range(1, max_token_length, tokenizer.model_max_length):
            states_list.append(
                encoder_hidden_states[:, i : i + tokenizer.model_max_length - 2]
            )  # <BOS> の後から <EOS> の前まで
            ids = input_ids[:, i : i + tokenizer.model_max_length - 2]
            mask_list.append(
                masks[:, i : i + tokenizer.model_max_length - 2]
                * (ids[:, :1] != tokenizer.eos_token_id)
            )
        states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
        mask_list.append(masks[:, -1].unsqueeze(1))

        masks = torch.cat(mask_list, dim=1)
        encoder_hidden_states = torch.cat(states_list, dim=1)

    if weight_dtype is not None:
        # this is required for additional network training
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return (
        encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1])),
        masks,
    )
    
@functools.cache
def calc_rope(height, width, patch_size=2, head_size=64):
    th = height // 8 // patch_size
    tw = width // 8 // patch_size
    base_size = 512 // 8 // patch_size
    start, stop = get_fill_resize_and_crop((th, tw), base_size)
    sub_args = [start, stop, (th, tw)]
    rope = get_2d_rotary_pos_embed(head_size, *sub_args)
    return rope


def _to_tuple(x):
    if isinstance(x, int):
        return x, x
    else:
        return x


def get_fill_resize_and_crop(src, tgt):
    th, tw = _to_tuple(tgt)
    h, w = _to_tuple(src)

    tr = th / tw  # base resolution
    r = h / w  # target resolution

    # resize
    if r > tr:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(
            round(tw / w * h)
        )  # resize the target resolution down based on the base resolution

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def get_meshgrid(start, *args):
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start)
        start = (0, 0)
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start)
        stop = _to_tuple(args[0])
        num = (stop[0] - start[0], stop[1] - start[1])
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start)
        stop = _to_tuple(args[0])
        num = _to_tuple(args[1])
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    grid_h = np.linspace(start[0], stop[0], num[0], endpoint=False, dtype=np.float32)
    grid_w = np.linspace(start[1], stop[1], num[1], endpoint=False, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)  # [2, W, H]
    return grid


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, start, *args, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = get_meshgrid(start, *args)  # [2, H, w]
    # grid_h = np.arange(grid_size, dtype=np.float32)
    # grid_w = np.arange(grid_size, dtype=np.float32)
    # grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # grid = np.stack(grid, axis=0)   # [2, W, H]

    grid = grid.reshape([2, 1, *grid.shape[1:]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (W,H)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                   Rotary Positional Embedding Functions                       #
#################################################################################
# https://github.com/facebookresearch/llama/blob/main/llama/model.py#L443


def get_2d_rotary_pos_embed(embed_dim, start, *args, use_real=True):
    """
    This is a 2d version of precompute_freqs_cis, which is a RoPE for image tokens with 2d structure.

    Parameters
    ----------
    embed_dim: int
        embedding dimension size
    start: int or tuple of int
        If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop, step is 1;
        If len(args) == 2, start is start, args[0] is stop, args[1] is num.
    use_real: bool
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns
    -------
    pos_embed: torch.Tensor
        [HW, D/2]
    """
    grid = get_meshgrid(start, *args)  # [2, H, w]
    grid = grid.reshape(
        [2, 1, *grid.shape[1:]]
    )  # Returns a sampling matrix with the same resolution as the target resolution
    pos_embed = get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=use_real)
    return pos_embed


def get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=False):
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[0].reshape(-1), use_real=use_real
    )  # (H*W, D/4)
    emb_w = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[1].reshape(-1), use_real=use_real
    )  # (H*W, D/4)

    if use_real:
        cos = torch.cat([emb_h[0], emb_w[0]], dim=1)  # (H*W, D/2)
        sin = torch.cat([emb_h[1], emb_w[1]], dim=1)  # (H*W, D/2)
        return cos, sin
    else:
        emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D/2)
        return emb


def get_1d_rotary_pos_embed(
    dim: int, pos: Union[np.ndarray, int], theta: float = 10000.0, use_real=False
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (np.ndarray, int): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional): If True, return real part and imaginary part separately.
                                   Otherwise, return complex numbers.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials. [S, D/2]

    """
    if isinstance(pos, int):
        pos = np.arange(pos)
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )  # [D/2]
    t = torch.from_numpy(pos).to(freqs.device)  # type: ignore  # [S]
    freqs = torch.outer(t, freqs).float()  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )  # complex64     # [S, D/2]
        return freqs_cis
