# from https://github.com/KohakuBlueleaf/LyCORIS/blob/main/lycoris/kohya/model_utils.py

import argparse
from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.linalg as linalg
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", help="The model which use it to train the dreambooth model",
        default='', type=str
    )
    parser.add_argument(
        "--target", help="the dreambooth model you want to extract the locon",
        default='', type=str
    )
    parser.add_argument(
        "--output", help="the output model",
        default='./out.pt', type=str
    )
    parser.add_argument(
        "--device", help="Which device you want to use to extract the locon",
        default='cuda', type=str
    )
    parser.add_argument(
        "--mode", 
        help=(
            'extraction mode, can be "fixed", "threshold", "ratio", "quantile". '
            'If not "fixed", network_dim and conv_dim will be ignored'
        ),
        default='fixed', type=str
    )
    parser.add_argument(
        "--linear_dim", help="network dim for linear layer in fixed mode",
        default=128, type=int
    )
    parser.add_argument(
        "--conv_dim", help="network dim for conv layer in fixed mode",
        default=128, type=int
    )
    parser.add_argument(
        "--linear_threshold", help="singular value threshold for linear layer in threshold mode",
        default=0., type=float
    )
    parser.add_argument(
        "--conv_threshold", help="singular value threshold for conv layer in threshold mode",
        default=0., type=float
    )
    parser.add_argument(
        "--linear_ratio", help="singular ratio for linear layer in ratio mode",
        default=0., type=float
    )
    parser.add_argument(
        "--conv_ratio", help="singular ratio for conv layer in ratio mode",
        default=0., type=float
    )
    parser.add_argument(
        "--linear_quantile", help="singular value quantile for linear layer quantile mode",
        default=1., type=float
    )
    parser.add_argument(
        "--conv_quantile", help="singular value quantile for conv layer quantile mode",
        default=1., type=float
    )
    parser.add_argument(
        "--use_sparse_bias", help="enable sparse bias",
        default=False, action="store_true"
    )
    parser.add_argument(
        "--sparsity", help="sparsity for sparse bias",
        default=0.98, type=float
    )
    parser.add_argument(
        "--disable_cp", help="don't use cp decomposition",
        default=False, action="store_true"
    )
    return parser.parse_args()

ARGS = get_args()

def make_sparse(t: torch.Tensor, sparsity=0.95):
    abs_t = torch.abs(t)
    np_array = abs_t.detach().cpu().numpy()
    quan = float(np.quantile(np_array, sparsity))
    sparse_t = t.masked_fill(abs_t < quan, 0)
    return sparse_t


def extract_conv(
    weight: Union[torch.Tensor, nn.Parameter],
    mode = 'fixed',
    mode_param = 0,
    device = 'cpu',
    is_cp = False,
) -> Tuple[nn.Parameter, nn.Parameter]:
    weight = weight.to(device)
    out_ch, in_ch, kernel_size, _ = weight.shape
    
    U, S, Vh = linalg.svd(weight.reshape(out_ch, -1))
    
    if mode=='fixed':
        lora_rank = mode_param
    elif mode=='threshold':
        assert mode_param>=0
        lora_rank = torch.sum(S>mode_param)
    elif mode=='ratio':
        assert 1>=mode_param>=0
        min_s = torch.max(S)*mode_param
        lora_rank = torch.sum(S>min_s)
    elif mode=='quantile' or mode=='percentile':
        assert 1>=mode_param>=0
        s_cum = torch.cumsum(S, dim=0)
        min_cum_sum = mode_param * torch.sum(S)
        lora_rank = torch.sum(s_cum<min_cum_sum)
    else:
        raise NotImplementedError('Extract mode should be "fixed", "threshold", "ratio" or "quantile"')
    lora_rank = max(1, lora_rank)
    lora_rank = min(out_ch, in_ch, lora_rank)
    if lora_rank>=out_ch/2 and not is_cp:
        return weight, 'full'
    
    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]
    
    diff = (weight - (U @ Vh).reshape(out_ch, in_ch, kernel_size, kernel_size)).detach()
    extract_weight_A = Vh.reshape(lora_rank, in_ch, kernel_size, kernel_size).detach()
    extract_weight_B = U.reshape(out_ch, lora_rank, 1, 1).detach()
    del U, S, Vh, weight
    return (extract_weight_A, extract_weight_B, diff), 'low rank'


def extract_linear(
    weight: Union[torch.Tensor, nn.Parameter],
    mode = 'fixed',
    mode_param = 0,
    device = 'cpu',
) -> Tuple[nn.Parameter, nn.Parameter]:
    weight = weight.to(device)
    out_ch, in_ch = weight.shape
    
    U, S, Vh = linalg.svd(weight)
    
    if mode=='fixed':
        lora_rank = mode_param
    elif mode=='threshold':
        assert mode_param>=0
        lora_rank = torch.sum(S>mode_param)
    elif mode=='ratio':
        assert 1>=mode_param>=0
        min_s = torch.max(S)*mode_param
        lora_rank = torch.sum(S>min_s)
    elif mode=='quantile' or mode=='percentile':
        assert 1>=mode_param>=0
        s_cum = torch.cumsum(S, dim=0)
        min_cum_sum = mode_param * torch.sum(S)
        lora_rank = torch.sum(s_cum<min_cum_sum)
    else:
        raise NotImplementedError('Extract mode should be "fixed", "threshold", "ratio" or "quantile"')
    lora_rank = max(1, lora_rank)
    lora_rank = min(out_ch, in_ch, lora_rank)
    if lora_rank>=out_ch/2:
        return weight, 'full'
    
    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]
    
    diff = (weight - U @ Vh).detach()
    extract_weight_A = Vh.reshape(lora_rank, in_ch).detach()
    extract_weight_B = U.reshape(out_ch, lora_rank).detach()
    del U, S, Vh, weight
    return (extract_weight_A, extract_weight_B, diff), 'low rank'


def extract_diff(
    base_model,
    db_model,
    mode = 'fixed',
    linear_mode_param = 0,
    conv_mode_param = 0,
    extract_device = 'cpu',
    use_bias = False,
    sparsity = 0.98,
    small_conv = True
):
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel", 
        "Attention", 
        "ResnetBlock2D", 
        "Downsample2D", 
        "Upsample2D"
    ]
    UNET_TARGET_REPLACE_NAME = [
        "conv_in",
        "conv_out",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'
    def make_state_dict(
        prefix, 
        root_module: torch.nn.Module,
        target_module: torch.nn.Module,
        target_replace_modules,
        target_replace_names = []
    ):
        loras = {}
        temp = {}
        temp_name = {}
        
        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                temp[name] = {}
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ not in {'Linear', 'Conv2d', "LoRACompatibleLinear", "LoRACompatibleConv"}:
                        continue
                    temp[name][child_name] = child_module.weight
            elif name in target_replace_names:
                temp_name[name] = module.weight
        
        for name, module in tqdm(list(target_module.named_modules())):
            if name in temp:
                weights = temp[name]
                for child_name, child_module in module.named_modules():
                    lora_name = prefix + '.' + name + '.' + child_name
                    lora_name = lora_name.replace('.', '_')
                    layer = child_module.__class__.__name__
                    if layer in {'Linear', 'Conv2d', "LoRACompatibleLinear", "LoRACompatibleConv"}:
                        root_weight = child_module.weight
                        if torch.allclose(root_weight, weights[child_name]):
                            continue
                    
                    if layer in ['Linear', "LoRACompatibleLinear"]:
                        weight, decompose_mode = extract_linear(
                            (child_module.weight - weights[child_name]),
                            mode,
                            linear_mode_param,
                            device = extract_device,
                        )
                        if decompose_mode == 'low rank':
                            extract_a, extract_b, diff = weight
                    elif layer in ['Conv2d', "LoRACompatibleConv"]:
                        is_linear = (child_module.weight.shape[2] == 1
                                     and child_module.weight.shape[3] == 1)
                        weight, decompose_mode = extract_conv(
                            (child_module.weight - weights[child_name]), 
                            mode,
                            linear_mode_param if is_linear else conv_mode_param,
                            device = extract_device,
                        )
                        if decompose_mode == 'low rank':
                            extract_a, extract_b, diff = weight
                        if small_conv and not is_linear and decompose_mode == 'low rank':
                            dim = extract_a.size(0)
                            (extract_c, extract_a, _), _ = extract_conv(
                                extract_a.transpose(0, 1), 
                                'fixed', dim, 
                                extract_device, True
                            )
                            extract_a = extract_a.transpose(0, 1)
                            extract_c = extract_c.transpose(0, 1)
                            loras[f'{lora_name}.lora_mid.weight'] = extract_c.detach().cpu().contiguous().half()
                            diff = child_module.weight - torch.einsum(
                                'i j k l, j r, p i -> p r k l', 
                                extract_c, extract_a.flatten(1, -1), extract_b.flatten(1, -1)
                            ).detach().cpu().contiguous()
                            del extract_c
                    else:
                        continue
                    if decompose_mode == 'low rank':
                        loras[f'{lora_name}.lora_down.weight'] = extract_a.detach().cpu().contiguous().half()
                        loras[f'{lora_name}.lora_up.weight'] = extract_b.detach().cpu().contiguous().half()
                        loras[f'{lora_name}.alpha'] = torch.Tensor([extract_a.shape[0]]).half()
                        if use_bias:
                            diff = diff.detach().cpu().reshape(extract_b.size(0), -1)
                            sparse_diff = make_sparse(diff, sparsity).to_sparse().coalesce()
                            
                            indices = sparse_diff.indices().to(torch.int16)
                            values = sparse_diff.values().half()
                            loras[f'{lora_name}.bias_indices'] = indices
                            loras[f'{lora_name}.bias_values'] = values
                            loras[f'{lora_name}.bias_size'] = torch.tensor(diff.shape).to(torch.int16)
                        del extract_a, extract_b, diff
                    elif decompose_mode == 'full':
                        loras[f'{lora_name}.diff'] = weight.detach().cpu().contiguous().half()
                    else:
                        raise NotImplementedError
            elif name in temp_name:
                weights = temp_name[name]
                lora_name = prefix + '.' + name
                lora_name = lora_name.replace('.', '_')
                layer = module.__class__.__name__
                
                if layer in {'Linear', 'Conv2d', "LoRACompatibleLinear", "LoRACompatibleConv"}:
                    root_weight = module.weight
                    if torch.allclose(root_weight, weights):
                        continue
                
                if layer in ['Linear', "LoRACompatibleLinear"]:
                    weight, decompose_mode = extract_linear(
                        (root_weight - weights),
                        mode,
                        linear_mode_param,
                        device = extract_device,
                    )
                    if decompose_mode == 'low rank':
                        extract_a, extract_b, diff = weight
                elif layer in ['Conv2d', "LoRACompatibleConv"]:
                    is_linear = (
                        root_weight.shape[2] == 1
                        and root_weight.shape[3] == 1
                    )
                    weight, decompose_mode = extract_conv(
                        (root_weight - weights), 
                        mode,
                        linear_mode_param if is_linear else conv_mode_param,
                        device = extract_device,
                    )
                    if decompose_mode == 'low rank':
                        extract_a, extract_b, diff = weight
                    if small_conv and not is_linear and decompose_mode == 'low rank':
                        dim = extract_a.size(0)
                        (extract_c, extract_a, _), _ = extract_conv(
                            extract_a.transpose(0, 1), 
                            'fixed', dim, 
                            extract_device, True
                        )
                        extract_a = extract_a.transpose(0, 1)
                        extract_c = extract_c.transpose(0, 1)
                        loras[f'{lora_name}.lora_mid.weight'] = extract_c.detach().cpu().contiguous().half()
                        diff = root_weight - torch.einsum(
                            'i j k l, j r, p i -> p r k l', 
                            extract_c, extract_a.flatten(1, -1), extract_b.flatten(1, -1)
                        ).detach().cpu().contiguous()
                        del extract_c
                else:
                    continue
                if decompose_mode == 'low rank':
                    loras[f'{lora_name}.lora_down.weight'] = extract_a.detach().cpu().contiguous().half()
                    loras[f'{lora_name}.lora_up.weight'] = extract_b.detach().cpu().contiguous().half()
                    loras[f'{lora_name}.alpha'] = torch.Tensor([extract_a.shape[0]]).half()
                    if use_bias:
                        diff = diff.detach().cpu().reshape(extract_b.size(0), -1)
                        sparse_diff = make_sparse(diff, sparsity).to_sparse().coalesce()
                        
                        indices = sparse_diff.indices().to(torch.int16)
                        values = sparse_diff.values().half()
                        loras[f'{lora_name}.bias_indices'] = indices
                        loras[f'{lora_name}.bias_values'] = values
                        loras[f'{lora_name}.bias_size'] = torch.tensor(diff.shape).to(torch.int16)
                    del extract_a, extract_b, diff
                elif decompose_mode == 'full':
                    loras[f'{lora_name}.diff'] = weight.detach().cpu().contiguous().half()
                else:
                    raise NotImplementedError
        return loras
    
    unet_loras = make_state_dict(
        LORA_PREFIX_UNET,
        base_model, db_model, 
        UNET_TARGET_REPLACE_MODULE,
        UNET_TARGET_REPLACE_NAME
    )
    return unet_loras


import torch
from safetensors.torch import save_file
from diffusers import StableDiffusionXLPipeline


def main():
    args = ARGS
    base = StableDiffusionXLPipeline.from_single_file(args.base).unet
    db = StableDiffusionXLPipeline.from_single_file(args.target).unet
    
    linear_mode_param = {
        'fixed': args.linear_dim,
        'threshold': args.linear_threshold,
        'ratio': args.linear_ratio,
        'quantile': args.linear_quantile,
    }[args.mode]
    conv_mode_param = {
        'fixed': args.conv_dim,
        'threshold': args.conv_threshold,
        'ratio': args.conv_ratio,
        'quantile': args.conv_quantile,
    }[args.mode]
    
    state_dict = extract_diff(
        base, db,
        args.mode,
        linear_mode_param, conv_mode_param,
        args.device, 
        args.use_sparse_bias, 
        args.sparsity,
        not args.disable_cp
    )
    save_file(state_dict, args.output)


if __name__ == '__main__':
    main()