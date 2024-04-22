from collections import defaultdict
import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from common.utils import get_class
from common.logging import logger

import bitsandbytes
import torch.nn as nn
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.model_summary import ModelSummary, get_human_readable_count
from transformers import AutoTokenizer, BitsAndBytesConfig
from models.llava.llava_llama import LlavaLlamaForCausalLM, LlavaConfig

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

def get_optimizer_parameters(opt_model, config):
    # Get the names of the parameters that should decay
    optim_param = config.optimizer.params
    mm_projector_lr = config.model_config.mm_projector_lr
    mm_vision_tower_lr = config.model_config.mm_vision_tower_lr
    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name or "image_newline" in name]
    vision_tower_parameters = [name for name, _ in opt_model.named_parameters() if "vision_tower" in name] 
    added_params = set()

    # Helper function to get the parameters in a group
    def get_params_in_group(decay, projector=False, vision_tower=False):
        params_in_group = []
        for n, p in opt_model.named_parameters():
            if (
                p.requires_grad and \
                (n not in added_params) and \
                (n in decay_parameters) == decay and \
                (n in projector_parameters) == projector and \
                (n in vision_tower_parameters) == vision_tower
            ):
                params_in_group.append(p)
                added_params.add(n)
        return params_in_group
    
    optimizer_grouped_parameters = []

    # Add the projector parameters to the parameter groups, if applicable
    if mm_projector_lr is not None:
        optimizer_grouped_parameters.extend([
            {"params": get_params_in_group(True, projector=True), "weight_decay": optim_param.weight_decay, "lr": mm_projector_lr},
            {"params": get_params_in_group(False, projector=True), "weight_decay": 0.0, "lr": mm_projector_lr},
        ])
        
    if mm_vision_tower_lr is not None:
        optimizer_grouped_parameters.extend([
            {"params": get_params_in_group(True, vision_tower=True), "weight_decay": optim_param.weight_decay, "lr": mm_vision_tower_lr},
            {"params": get_params_in_group(False, vision_tower=True), "weight_decay": 0.0, "lr": mm_vision_tower_lr},
        ])
        
    # Create the parameter groups for the optimizer
    optimizer_grouped_parameters.extend([
        {"params": get_params_in_group(True), "weight_decay": optim_param.weight_decay},
        {"params": get_params_in_group(False), "weight_decay": 0.0},
    ])
    
    # scan and remove empty groups
    optimizer_grouped_parameters = [group for group in optimizer_grouped_parameters if len(group["params"]) > 0]
    return optimizer_grouped_parameters

def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = LLaVAModel(model_path, config, fabric.device)
    dataset, dataloader = model.prepare_dataset(config)

    # Prepare the optimizer
    optim_param = config.optimizer.params
    optim_cls = config.optimizer.name
    optimizer_grouped_parameters = get_optimizer_parameters(model, config)
    optimizer = get_class(optim_cls)(optimizer_grouped_parameters, **optim_param)
    scheduler = get_class(config.scheduler.name)(optimizer, **config.scheduler.params)

    if "bitsandbytes" in optim_cls and "8bit" in optim_cls:
        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
        skipped = 0
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"skipped: {skipped/2**20}M params")
                
    # Print the model summary, if applicable
    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    fabric.barrier()
    model.model, optimizer = fabric.setup(model.model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    if hasattr(fabric.strategy, "_deepspeed_engine"):
        model._deepspeed_engine = fabric.strategy._deepspeed_engine
    if hasattr(fabric.strategy, "_fsdp_kwargs"):
        model._fsdp_engine = fabric.strategy
    
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
        config.dataset.update({"version": mm_config.pop("version")})
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ) if use_q_lora else None
        
        cfg_pretrained = None
        if not mm_config.get("vision_tower"):
            cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            cfg_pretrained.update(mm_config)
            # logger.info(f"Using config: {cfg_pretrained}")
        
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=q_config,
            config=cfg_pretrained,
            **config.model_params
        )
        mm_config = OmegaConf.create(cfg_pretrained.to_dict())
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.train() 

        if use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            logger.info(f"Using Lora with q_lora={use_q_lora}")
            config.lora_params.update({"target_modules": find_all_linear_names(model)})
            lora_config = LoraConfig(**OmegaConf.to_container(config.lora_params))
            if use_q_lora:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

            model = get_peft_model(model, lora_config)
            model.enable_input_require_grads()
            model.print_trainable_parameters()
            
        # load adapter
        tokenizer_path = config.trainer.get("tokenizer_path", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)     
        self.tokenizer.pad_token = self.tokenizer.unk_token
        model.config.tokenizer_padding_side = self.tokenizer.padding_side
        model.config.tokenizer_model_max_length = self.tokenizer.model_max_length  

        if mm_config.freeze_backbone:
            model.requires_grad_(False)
        
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = mm_config.tune_mm_mlp_adapter
        
        if hasattr(model, "image_newline"):
            model.get_model().image_newline.requires_grad = True
                
        if mm_config.tune_mm_vision_tower:
            model.get_vision_tower().requires_grad_(True)
        else:
            model.get_vision_tower().requires_grad_(False)
                
        param_counts = defaultdict(int)
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Increment the count for this type of parameter
                param_counts[".".join(name.split('.')[:2])] += param.numel()

        # Log the count of each type of parameter
        for param_type, count in param_counts.items():
            logger.info(f"Trainable: {param_type} - {get_human_readable_count(count)} parameters")
                
        model.get_vision_tower().to(next(model.parameters()).dtype) 
        self.image_processor = model.get_vision_tower().image_processor
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
            if isinstance(batch[k], torch.Tensor):
                batch[k] = v.to(self.target_device)

        out = self.model(**batch)
        loss = out["loss"]
        return loss

    def save_checkpoint(self, model_path, metadata):
        if self.model.config.freeze_backbone:
            keys_to_match = ['mm_projector', 'vision_resampler', 'image_newline']
            if getattr(self.config.model_config, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            named_params = list(self.model.named_parameters())
            weight_to_save = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
            param = next(self.model.parameters())
            if hasattr(param, "ds_id") and hasattr(self, "_deepspeed_engine"):
                from deepspeed import zero
    
                with zero.GatheredParameters([param]):
                    for k, v in weight_to_save.items():
                        v = v.data.detach().cpu().clone()
                        weight_to_save[k] = v.cpu()
            elif hasattr(self, "_fsdp_engine"):
                from lightning.fabric.strategies.fsdp import _get_full_state_dict_context
                
                world_size = self._fsdp_engine.world_size
                with _get_full_state_dict_context(self.model._forward_module, world_size=world_size):
                    for k, v in weight_to_save.items():
                        v = v.data.detach().cpu().clone()
                        weight_to_save[k] = v.cpu()
        else:
            # save full model
            # test param if its zero3
            weight_to_save = None
            param = next(self.model.parameters())
            if hasattr(param, "ds_id") and hasattr(self, "_deepspeed_engine"):
                try:
                    weight_to_save = self._deepspeed_engine._zero3_consolidated_16bit_state_dict()
                except ValueError:
                    pass
            elif hasattr(self, "_fsdp_engine"):
                from lightning.fabric.strategies.fsdp import _get_full_state_dict_context
                
                world_size = self._fsdp_engine.world_size
                with _get_full_state_dict_context(self.model._forward_module, world_size=world_size):
                    weight_to_save = self.model._forward_module.state_dict()
            else:
                weight_to_save = self.model.state_dict()
                
        self._save_checkpoint(model_path, weight_to_save)

    @rank_zero_only
    def _save_checkpoint(self, model_path, weight_to_save):
        # remove all '_forward_module.module' prefix
        for k in list(weight_to_save.keys()):
            if k.startswith("_forward_module.module."):
                weight_to_save[k.replace("_forward_module.module.", "")] = weight_to_save.pop(k)
        
        if self.model.config.freeze_backbone:
            self.model.config.save_pretrained(model_path)
            torch.save(weight_to_save, os.path.join(model_path, f'mm_projector.bin'))
        else:
            self.tokenizer.save_pretrained(model_path)
            self.model.save_pretrained(model_path, state_dict=weight_to_save)
            
        logger.info(f"Saved model to {model_path}")