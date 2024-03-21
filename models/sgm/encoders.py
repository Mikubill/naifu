from contextlib import nullcontext
from typing import Dict, List, Optional, Union

import math

import numpy as np
import open_clip
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import ListConfig
from torch.utils.checkpoint import checkpoint
from .model_util import EmptyInitWrapper
from transformers import (
    CLIPTextModel,
    CLIPTextConfig,
    CLIPTokenizer,
    modeling_utils,
)
from .encoder_util import (
    count_params,
    default,
    disabled_train,
    expand_dims_like,
    instantiate_from_config,
)

import logging
logger = logging.getLogger("Trainer")

def process_input_ids(input_ids, tokenizer, max_length):
    if max_length > tokenizer.model_max_length:
        input_ids = input_ids.squeeze(0)
        iids_list = []
        if tokenizer.pad_token_id == tokenizer.eos_token_id: # sdv1
            for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):  # (1, 152, 75)
                ids_chunk = (
                    input_ids[0].unsqueeze(0),
                    input_ids[i : i + tokenizer.model_max_length - 2],
                    input_ids[-1].unsqueeze(0),
                )
                ids_chunk = torch.cat(ids_chunk)
                iids_list.append(ids_chunk)
        else: # v2 or SDXL
            # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
            for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
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


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


class Timestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key


class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

    def __init__(self, emb_models: Union[List, ListConfig]):
        super().__init__()
        embedders = []
        for n, embconfig in enumerate(emb_models):
            embedder = instantiate_from_config(embconfig)
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                if hasattr(embedder, "freeze"):
                    embedder.freeze()
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            logger.info(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}")

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

    def forward(self, batch: Dict, force_zero_embeddings=None):
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
            
        for embedder in self.embedders:
            # print(embedder.__class__.__name__, embedder.input_key if hasattr(embedder, "input_key") else None)
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
                    
            assert isinstance(emb_out, (torch.Tensor, list, tuple)) \
                , f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
                
            for emb in emb_out:
                # print(emb.shape, emb.dim(), self.OUTPUT_DIM2KEYS[emb.dim()], self.KEY2CATDIM[self.OUTPUT_DIM2KEYS[emb.dim()]])
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    emb = expand_dims_like(torch.bernoulli((1.0 - embedder.ucg_rate) * torch.ones(emb.shape[0], device=emb.device)),emb,) * emb
                    
                if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
                    emb = torch.zeros_like(emb)
                    
                if out_key in output:
                    output[out_key] = torch.cat((output[out_key], emb), self.KEY2CATDIM[out_key])
                else:
                    output[out_key] = emb
                    
        return output

    def get_unconditional_conditioning(
        self, batch_c, batch_uc=None, force_uc_zero_embeddings=None
    ):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class IdentityEncoder(AbstractEmbModel):
    def encode(self, x):
        return x

    def forward(self, x):
        return x


class ClassEmbedder(AbstractEmbModel):
    def __init__(self, embed_dim, n_classes=1000, add_sequence_dim=False):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.add_sequence_dim = add_sequence_dim

    def forward(self, c):
        c = self.embedding(c)
        if self.add_sequence_dim:
            c = c[:, None, :]
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = (
            self.n_classes - 1
        )  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc.long()}
        return uc


class ClassEmbedderForMultiCond(ClassEmbedder):
    def forward(self, batch, key=None, disable_dropout=False):
        out = batch
        key = default(key, self.key)
        islist = isinstance(batch[key], list)
        if islist:
            batch[key] = batch[key][0]
        c_out = super().forward(batch, key, disable_dropout)
        out[key] = [c_out] if islist else c_out
        return out


class FrozenCLIPEmbedder(AbstractEmbModel):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=227,
        layer="last",
        layer_idx=None,
        always_return_pooled=False,
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        with modeling_utils.no_init_weights():
            config = CLIPTextConfig.from_pretrained(version)
            self.transformer = CLIPTextModel(config)
            # self.transformer = CLIPTextModel.from_pretrained(version)
                
        self.device = device
        self.max_length = max_length
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = always_return_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12
            
            # Freeze the layers after the unused layer to avoid grad issues
            actual_layer_idx = layer_idx if layer_idx >= 0 else 12 + layer_idx
            text_model = self.transformer.text_model
            for layer in text_model.encoder.layers[actual_layer_idx:]:
                layer.requires_grad_(False)
            text_model.final_layer_norm.requires_grad_(False)

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        # not support pooled output at the moment
        assert not self.return_pooled and not self.layer == "pooled"
        
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        ).input_ids
        
        # b,n,77
        input_ids = torch.stack([process_input_ids(batch, self.tokenizer, self.max_length) for batch in batch_encoding]).to(self.device)
        batch_size = input_ids.size()[0]
        
        # b,n,77 -> b*n, 77
        input_ids = input_ids.reshape((-1, self.tokenizer.model_max_length))
        outputs = self.transformer(input_ids=input_ids, output_hidden_states=self.layer == "hidden")
        
        if self.layer == "last":
            z = outputs.last_hidden_state
        else:
            z = outputs.hidden_states[self.layer_idx]
            
        # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
        hidden_states = z.reshape((batch_size, -1, z.shape[-1]))
        states_list = [hidden_states[:, 0].unsqueeze(1)]  # <BOS>
        
        # <BOS> の後から <EOS> の前まで
        for i in range(1, self.max_length, self.tokenizer.model_max_length):
            states_list.append(hidden_states[:, i : i + self.tokenizer.model_max_length - 2])  
            
        states_list.append(hidden_states[:, -1].unsqueeze(1))  # <EOS>
        z = torch.cat(states_list, dim=1)       
        return z

    def encode(self, text):
        return self(text)


class OpenClipTextModel(open_clip.CLIP):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim,
            vision_cfg,
            text_cfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        nn.Module.__init__(self)
        self.output_dict = output_dict
        text = open_clip.model._build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
            
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale, requires_grad=False)
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)
            

class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=227,
        layer="last",
        always_return_pooled=False,
        legacy=True,
    ):
        super().__init__()
        assert layer in self.LAYERS
        with EmptyInitWrapper(device):
            model = OpenClipTextModel(**open_clip.get_model_config(arch))
            
        self.model = model
        self.device = device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = legacy
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", 
            pad_token="!"
        ) # same as open clip tokenizer

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def process_hidden_state(self, z, batch_size):
        # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
        hidden_states = z.reshape((batch_size, -1, z.shape[-1]))
        states_list = [hidden_states[:, 0].unsqueeze(1)]  # <BOS>
        
        # <BOS> の後から <EOS> の前まで
        for i in range(1, self.max_length, self.clip_tokenizer.model_max_length):
            states_list.append(hidden_states[:, i : i + self.clip_tokenizer.model_max_length - 2])  
            
        states_list.append(hidden_states[:, -1].unsqueeze(1))  # <EOS>
        z = torch.cat(states_list, dim=1)    
        return z

    def forward(self, text):
        tokens = open_clip.tokenize(text, context_length=self.max_length)
        input_ids = torch.stack([process_input_ids(batch, self.clip_tokenizer, self.max_length) for batch in tokens]).to(self.device)
        batch_size = input_ids.size()[0]
        
        input_ids = input_ids.reshape((-1, self.clip_tokenizer.model_max_length))
        z = self.encode_with_transformer(input_ids)
        if not self.return_pooled and self.legacy:
            return self.process_hidden_state(z, batch_size)
        
        hidden_state = z[self.layer]
        hidden_state = self.process_hidden_state(hidden_state, batch_size)
        if self.return_pooled:
            assert not self.legacy
            pooled = z["pooled"][::self.max_length // 75]
            return hidden_state, pooled
        
        return hidden_state

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        else:
            # x is a dict and will stay a dict
            o = x["last"]
            o = self.model.ln_final(o)
            pooled = self.pool(o, text)
            x["pooled"] = pooled
            return x

    def pool(self, x, text):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            @ self.model.text_projection
        )
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD
        return outputs

    def encode(self, text):
        return self(text)
    
class ConcatTimestepEmbedderND(AbstractEmbModel):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, outdim):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return emb
