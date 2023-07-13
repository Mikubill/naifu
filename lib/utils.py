import os
import re
import torch

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
    
def min_snr_weighted_loss(eps_pred:torch.Tensor, eps:torch.Tensor, timesteps, noise_scheduler, gamma:float):
    alphas = noise_scheduler.alphas_cumprod[timesteps.to("cpu")]
    prediction_type = noise_scheduler.config.prediction_type
    snrs = alphas / (1 - alphas)
    comp = torch.tensor(gamma) / snrs if prediction_type == "epsilon" else torch.tensor(gamma) / (snrs+1)
    weights = torch.minimum(comp, torch.ones_like(comp)).to(eps_pred.device)
    losses = torch.nn.functional.mse_loss(eps_pred, eps, reduction="none").mean(dim=tuple(range(1, eps.ndim)))
    loss = (losses * weights).mean()
    return loss

def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count

# Modified from https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_detection.py

def detect_unet_config(state_dict, key_prefix):
    state_dict_keys = list(state_dict.keys())

    unet_config = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False
    }

    y_input = '{}label_emb.0.0.weight'.format(key_prefix)
    if y_input in state_dict_keys:
        unet_config["num_classes"] = "sequential"
        unet_config["adm_in_channels"] = state_dict[y_input].shape[1]
    else:
        unet_config["adm_in_channels"] = None

    model_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[0]
    in_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[1]

    num_res_blocks = []
    channel_mult = []
    attention_resolutions = []
    transformer_depth = []
    context_dim = None
    use_linear_in_transformer = False


    current_res = 1
    count = 0

    last_res_blocks = 0
    last_transformer_depth = 0
    last_channel_mult = 0

    while True:
        prefix = '{}input_blocks.{}.'.format(key_prefix, count)
        block_keys = sorted(list(filter(lambda a: a.startswith(prefix), state_dict_keys)))
        if len(block_keys) == 0:
            break

        if "{}0.op.weight".format(prefix) in block_keys: #new layer
            if last_transformer_depth > 0:
                attention_resolutions.append(current_res)
            transformer_depth.append(last_transformer_depth)
            num_res_blocks.append(last_res_blocks)
            channel_mult.append(last_channel_mult)

            current_res *= 2
            last_res_blocks = 0
            last_transformer_depth = 0
            last_channel_mult = 0
        else:
            res_block_prefix = "{}0.in_layers.0.weight".format(prefix)
            if res_block_prefix in block_keys:
                last_res_blocks += 1
                last_channel_mult = state_dict["{}0.out_layers.3.weight".format(prefix)].shape[0] // model_channels

            transformer_prefix = prefix + "1.transformer_blocks."
            transformer_keys = sorted(list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys)))
            if len(transformer_keys) > 0:
                last_transformer_depth = count_blocks(state_dict_keys, transformer_prefix + '{}')
                if context_dim is None:
                    context_dim = state_dict['{}0.attn2.to_k.weight'.format(transformer_prefix)].shape[1]
                    use_linear_in_transformer = len(state_dict['{}1.proj_in.weight'.format(prefix)].shape) == 2

        count += 1

    if last_transformer_depth > 0:
        attention_resolutions.append(current_res)
    transformer_depth.append(last_transformer_depth)
    num_res_blocks.append(last_res_blocks)
    channel_mult.append(last_channel_mult)
    transformer_depth_middle = count_blocks(state_dict_keys, '{}middle_block.1.transformer_blocks.'.format(key_prefix) + '{}')

    if len(set(num_res_blocks)) == 1:
        num_res_blocks = num_res_blocks[0]

    if len(set(transformer_depth)) == 1:
        transformer_depth = transformer_depth[0]

    unet_config["in_channels"] = in_channels
    unet_config["model_channels"] = model_channels
    unet_config["num_res_blocks"] = num_res_blocks
    unet_config["attention_resolutions"] = attention_resolutions
    unet_config["transformer_depth"] = transformer_depth
    unet_config["channel_mult"] = channel_mult
    unet_config["transformer_depth_middle"] = transformer_depth_middle
    unet_config['use_linear_in_transformer'] = use_linear_in_transformer
    unet_config["context_dim"] = context_dim
    return unet_config

import safetensors.torch

def load_torch_file(ckpt, safe_load=False):
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device="cpu")
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd

def save_torch_file(sd, ckpt, metadata=None):
    if metadata is not None:
        safetensors.torch.save_file(sd, ckpt, metadata=metadata)
    else:
        safetensors.torch.save_file(sd, ckpt)

def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from*x:shape_from*(x + 1)]
    return sd

def load_model_weights(model, sd):
    m, u = model.load_state_dict(sd, strict=False)
    m = set(m)
    unexpected_keys = set(u)

    k = list(sd.keys())
    for x in k:
        if x not in unexpected_keys:
            w = sd.pop(x)
            del w
    if len(m) > 0:
        print("missing", m)
    return model

def load_clip_weights(model, sd):
    k = list(sd.keys())
    for x in k:
        if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
            y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
            sd[y] = sd.pop(x)

    if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in sd:
        ids = sd['cond_stage_model.transformer.text_model.embeddings.position_ids']
        if ids.dtype == torch.float32:
            sd['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

    sd = transformers_convert(sd, "cond_stage_model.model.", "cond_stage_model.transformer.text_model.", 24)
    return load_model_weights(model, sd)

def get_free_memory(dev=None, torch_free_too=False):
    stats = torch.cuda.memory_stats(dev)
    mem_active = stats['active_bytes.all.current']
    mem_reserved = stats['reserved_bytes.all.current']
    mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total
