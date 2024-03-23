from collections import defaultdict
import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from common.utils import get_class
from common.logging import logger

import torch.nn as nn
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.model_summary import ModelSummary, get_human_readable_count
from transformers import AutoTokenizer, AutoConfig
from transformers import BitsAndBytesConfig
from models.llava.llava_llama import LlavaLlamaForCausalLM

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

def get_optimizer_parameters(opt_model, config):
    # Get the names of the parameters that should decay
    optim_param = config.optimizer.params
    mm_projector_lr = config.model_config.mm_projector_lr
    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    # Get the names of the projector parameters, if applicable
    projector_parameters = []
    if mm_projector_lr is not None:
        projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name] 

    # Helper function to get the parameters in a group
    def get_params_in_group(decay, projector):
        params_in_group = []
        for n, p in opt_model.named_parameters():
            if (n in decay_parameters) == decay and (n in projector_parameters) == projector and p.requires_grad:
                params_in_group.append(p)
        return params_in_group

    # Create the parameter groups for the optimizer
    optimizer_grouped_parameters = [
        {"params": get_params_in_group(True, False), "weight_decay": optim_param.weight_decay},
        {"params": get_params_in_group(False, False), "weight_decay": 0.0},
    ]

    # Add the projector parameters to the parameter groups, if applicable
    if mm_projector_lr is not None:
        optimizer_grouped_parameters.extend([
            {"params": get_params_in_group(True, True), "weight_decay": optim_param.weight_decay, "lr": mm_projector_lr},
            {"params": get_params_in_group(False, True), "weight_decay": 0.0, "lr": mm_projector_lr},
        ])

    return optimizer_grouped_parameters

def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = LLaVAModel(model_path, config, fabric.device)
    dataset, dataloader = model.prepare_dataset(config)

    # Prepare the optimizer
    optim_param = config.optimizer.params
    optimizer_grouped_parameters = get_optimizer_parameters(model, config)
    optimizer = get_class(config.optimizer.name)(optimizer_grouped_parameters, **optim_param)
    scheduler = get_class(config.scheduler.name)(optimizer, **config.scheduler.params)

    # Print the model summary, if applicable
    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    fabric.barrier()
    model.model, optimizer = fabric.setup(model.model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler

# define the LightningModule
class LLaVAModel(pl.LightningModule):
    def __init__(self, model_path, config, device):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.target_device = device
        mm_config = OmegaConf.to_container(config.model_config)
        use_q_lora = config.get("q_lora", False)
        use_lora = config.get("use_lora", False) or use_q_lora
        config.dataset.update({"model_version": mm_config.pop("version")})
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ) if use_q_lora else None
        
        cfg_pretrained = None
        if not mm_config.get("vision_tower"):
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            cfg_pretrained.update(mm_config)
            # logger.info(f"Using config: {cfg_pretrained}")
        
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=q_config,
            config=cfg_pretrained,
            **config.model_params
        )
        mm_config = config.model_config
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.train() 

        if use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            logger.info(f"Using Lora with q_lora={use_q_lora}")
            config.lora_params.update({"target_modules": find_all_linear_names(model)})
            lora_config = LoraConfig(**config.lora_params)
            if use_q_lora:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

            model = get_peft_model(model, lora_config)
            model.enable_input_require_grads()
            model.print_trainable_parameters()
            
        # load adapter
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)     
        self.tokenizer.pad_token = self.tokenizer.unk_token
        model.get_model().initialize_vision_modules(model_args=mm_config)
        model.config.tokenizer_padding_side = self.tokenizer.padding_side
        model.config.tokenizer_model_max_length = self.tokenizer.model_max_length  
         
        if mm_config.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        else:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
                
        param_counts = defaultdict(int)
        for name, param in model.model.named_parameters():
            if param.requires_grad:
                # Increment the count for this type of parameter
                param_counts[".".join(name.split('.')[:2])] += param.numel()

        # Log the count of each type of parameter
        for param_type, count in param_counts.items():
            logger.info(f"Trainable: {param_type} - {get_human_readable_count(count)} parameters")
                
        model.initialize_vision_tokenizer(mm_config, tokenizer=self.tokenizer)    
        self.image_processor = model.get_vision_tower().image_processor
        self.logger_samples = []
        self.model = model

    def prepare_dataset(self, config):
        dataset_class = get_class(config.dataset.name) 
        train_dataset = dataset_class(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            mm_config=self.config.model_config,
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
    def save_checkpoint(self, model_path, metadata):
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
            logger.info(f"Saved model to {model_path}")
