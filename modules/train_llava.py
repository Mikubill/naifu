import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from common.utils import (
    get_class,
    rank_zero_print,
    rank_zero_only,
)
import torch.nn as nn
from lightning.pytorch.utilities.model_summary import ModelSummary
from transformers import AutoTokenizer, AutoConfig
from transformers import BitsAndBytesConfig
from models.llava.llava_llama import LlavaLlamaForCausalLM
from models.llava.llava_mpt import LlavaMptForCausalLM


ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def maybe_zero_3(param, ignore_status=False, name=None):
    try:
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    except ImportError:
        return param
    
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = LLaVAModel(model_path, config, fabric.device)
    dataset, dataloader = model.prepare_dataset(config)

    opt_model = model
    optim_param = config.optimizer.params
    mm_projector_lr = config.vision_model_params.mm_projector_lr
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    if mm_projector_lr is not None:
        projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                ],
                "weight_decay": optim_param.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                ],
                "weight_decay": optim_param.weight_decay,
                "lr": mm_projector_lr,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
                "lr": mm_projector_lr,
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": optim_param.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

    optimizer = get_class(config.optimizer.name)(optimizer_grouped_parameters, **optim_param)
    scheduler = get_class(config.scheduler.name)(optimizer, **config.scheduler.params)

    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    fabric.barrier()
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


# define the LightningModule
class LLaVAModel(pl.LightningModule):
    def __init__(self, model_path, config, device):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.target_device = device
        use_q_lora = config.get("q_lora", False)
        use_lora = config.get("use_lora", False) or use_q_lora
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ) if use_q_lora else None
        
        if 'mpt' in model_path:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = config.model_params.get('attn_implementation', 'sdpa')
            model = LlavaMptForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                quantization_config=q_config,
                **config.model_params
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                quantization_config=q_config,
                **config.model_params
            )

        mm_config = config.vision_model_params
        if isinstance(mm_config.pretrain_mm_mlp_adapter, str) and \
            mm_config.pretrain_mm_mlp_adapter.startswith('https://'):
            # download the adapter
            torch.hub.download_url_to_file(mm_config.pretrain_mm_mlp_adapter, "pretrain_mm_projector.pt")
            mm_config.pretrain_mm_mlp_adapter = "pretrain_mm_projector.pt"
        
        model.get_model().initialize_vision_modules(model_args=mm_config)
        vision_tower = model.get_vision_tower().to(dtype=next(model.parameters()).dtype)
        self.image_processor = vision_tower.image_processor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)     
        
        # update model config?
        model.config.image_aspect_ratio = mm_config.image_aspect_ratio
        model.config.tune_mm_mlp_adapter = mm_config.tune_mm_mlp_adapter
        model.config.tokenizer_padding_side = self.tokenizer.padding_side
        model.config.tokenizer_model_max_length = self.tokenizer.model_max_length
        model.config.freeze_mm_mlp_adapter = not mm_config.tune_mm_mlp_adapter
        
        if mm_config.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        else:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
                
        model.config.mm_use_im_start_end = mm_config.mm_use_im_start_end
        model.config.mm_projector_lr = mm_config.mm_projector_lr
        model.config.mm_use_im_patch_token = mm_config.mm_use_im_patch_token
        config.dataset.update({"model_version": mm_config.version})
        
        model.gradient_checkpointing_enable()
        model.train() 
        self.logger_samples = []
        if use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            rank_zero_print(f"Using Lora with q_lora={use_q_lora}")
            config.lora_params.update({"target_modules": find_all_linear_names(model)})
            lora_config = LoraConfig(**config.lora_params)
            if use_q_lora:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

            self.model = get_peft_model(model, lora_config)
            self.model.enable_input_require_grads()
            self.model.print_trainable_parameters()
            self.tokenizer.padding_side  = 'left'
        else:
            self.model = model
        
    def prepare_dataset(self, config):
        dataset_class = get_class(config.dataset.name)
        train_dataset = dataset_class(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            mm_config=self.config.vision_model_params,
            **config.dataset,
        )
        bsz = config.trainer.batch_size
        self.dataset = train_dataloader = train_dataset.build_dataloader(batch_size=bsz)
        return train_dataset, train_dataloader

    def forward(self, batch):
        for k, v in batch.items():
            batch[k] = v.to(self.target_device)

        out = self.model(**batch)
        loss = out["loss"]
        return loss

    @rank_zero_only
    def save_checkpoint(self, model_path):
        cfg = self.config.trainer
        if self.model.config.tune_mm_mlp_adapter:
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            named_params = list(self.model.named_parameters())
            weight_to_save = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
            weight_to_save = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in weight_to_save.items()}
            self.model.config.save_pretrained(model_path)
            torch.save(weight_to_save, os.path.join(model_path, f'mm_projector.bin'))
        else:
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            rank_zero_print(f"Saved model to {model_path}")
