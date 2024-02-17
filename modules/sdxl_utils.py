import torch
import math
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from models.sgm import UNetModel, AutoencoderKL


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class AutoencoderKLWrapper(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class UnetWrapper(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.diffusion_model = UNetModel(**config)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,
        )


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_timestep_embedding(x, outdim):
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb


def get_size_embeddings(orig_size, crop_size, target_size, device):
    emb1 = get_timestep_embedding(orig_size, 256)
    emb2 = get_timestep_embedding(crop_size, 256)
    emb3 = get_timestep_embedding(target_size, 256)
    vector = torch.cat([emb1, emb2, emb3], dim=1).to(device)
    return vector


def convert_sdxl_text_encoder_2_checkpoint(checkpoint, max_length=77):
    prefix = "conditioner.embedders.1.model."
    conv_mapping = {
        ".positional_embedding": ".embeddings.position_embedding.weight",
        "text_model.text_projection": "text_projection.weight",
        ".token_embedding.weight": ".embeddings.token_embedding.weight",
        ".ln_final": ".final_layer_norm",
    }

    def convert_key(key):
        key = key.replace(prefix + "transformer.", "text_model.encoder.")
        key = key.replace(prefix, "text_model.")
        if (
            "attn.in_proj" in key
            or "logit_scale" in key
            or "embeddings.position_ids" in key
        ):
            return None
        if ".resblocks." in key:
            key = key.replace(".resblocks.", ".layers.")
            if ".ln_" in key:
                key = key.replace(".ln_", ".layer_norm")
            elif ".mlp." in key:
                key = key.replace(".c_fc.", ".fc1.").replace(".c_proj.", ".fc2.")
            elif ".attn.out_proj" in key:
                key = key.replace(".attn.out_proj.", ".self_attn.out_proj.")
            elif ".attn.in_proj" in key:
                return None
            else:
                raise ValueError(f"unexpected key in SD: {key}")
        else:
            for old, new in conv_mapping.items():
                key = key.replace(old, new)
        return key

    new_sd = {convert_key(k): v for k, v in checkpoint.items() if convert_key(k)}
    for key in checkpoint.keys():
        if ".resblocks" in key and ".attn.in_proj_" in key:
            values = torch.chunk(checkpoint[key], 3)

            key_suffix = ".weight" if "weight" in key else ".bias"
            key_pfx = key.replace(
                prefix + "transformer.resblocks.", "text_model.encoder.layers."
            )
            key_pfx = key_pfx.replace("_weight", "")
            key_pfx = key_pfx.replace("_bias", "")
            key_pfx = key_pfx.replace(".attn.in_proj", ".self_attn.")
            new_sd[key_pfx + "q_proj" + key_suffix] = values[0]
            new_sd[key_pfx + "k_proj" + key_suffix] = values[1]
            new_sd[key_pfx + "v_proj" + key_suffix] = values[2]

    return new_sd


def process_input_ids(input_ids, tokenizer, max_length):
    if max_length > tokenizer.model_max_length:
        input_ids = input_ids.squeeze(0)
        iids_list = []
        for i in range(
            1,
            max_length - tokenizer.model_max_length + 2,
            tokenizer.model_max_length - 2,
        ):
            ids_chunk = (
                input_ids[0].unsqueeze(0),  # BOS
                input_ids[i : i + tokenizer.model_max_length - 2],
                input_ids[-1].unsqueeze(0),  # PAD or EOS
            )
            ids_chunk = torch.cat(ids_chunk)

            if ids_chunk[-2] not in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                ids_chunk[-1] = tokenizer.eos_token_id
            if ids_chunk[1] == tokenizer.pad_token_id:
                ids_chunk[1] = tokenizer.eos_token_id

            iids_list.append(ids_chunk)

        input_ids = torch.stack(iids_list)
    return input_ids


def get_hidden_states_sdxl(
    prompt: str,
    max_token_length: int,
    tokenizer1: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    text_encoder1: CLIPTextModel,
    text_encoder2: CLIPTextModelWithProjection,
):
    device = next(text_encoder1.parameters()).device
    input_ids1 = tokenizer1(
        prompt,
        truncation=True,
        max_length=max_token_length,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",
    ).input_ids
    input_ids2 = tokenizer2(
        prompt,
        truncation=True,
        max_length=max_token_length,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",
    ).input_ids

    input_ids1 = torch.stack(
        [process_input_ids(inp, tokenizer1, max_token_length) for inp in input_ids1]
    ).to(device)
    input_ids2 = torch.stack(
        [process_input_ids(inp, tokenizer2, max_token_length) for inp in input_ids2]
    ).to(device)

    # input_ids: b,n,77 -> b*n, 77
    b_size = input_ids1.size()[0]
    input_ids1 = input_ids1.reshape(
        (-1, tokenizer1.model_max_length)
    )  # batch_size*n, 77
    input_ids2 = input_ids2.reshape(
        (-1, tokenizer2.model_max_length)
    )  # batch_size*n, 77

    # text_encoder1
    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    # text_encoder2
    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]

    pool2 = enc_out["text_embeds"]

    # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
    n_size = 1 if max_token_length is None else max_token_length // 75
    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

    if max_token_length is not None:
        states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer1.model_max_length):
            states_list.append(
                hidden_states1[:, i : i + tokenizer1.model_max_length - 2]
            )
        states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
        hidden_states1 = torch.cat(states_list, dim=1)

        states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer2.model_max_length):
            chunk = hidden_states2[:, i : i + tokenizer2.model_max_length - 2]
            states_list.append(chunk)

        states_list.append(hidden_states2[:, -1].unsqueeze(1))
        hidden_states2 = torch.cat(states_list, dim=1)
        pool2 = pool2[::n_size]

    return hidden_states1, hidden_states2, pool2
