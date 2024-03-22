import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from common.utils import get_class
from common.logging import logger
from lightning.pytorch.utilities.model_summary import ModelSummary
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from lightning.pytorch.utilities import rank_zero_only
from lightning.fabric.wrappers import _unwrap_objects

def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = LLMModel(model_path, config, fabric.device)
    dataset, dataloader = model.prepare_dataset(config)

    params_to_optim = [{"params": model.parameters()}]
    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(params_to_optim, **optim_param)
    scheduler = get_class(config.scheduler.name)(optimizer, **config.scheduler.params)

    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    fabric.barrier()
    model.model, optimizer = fabric.setup(model.model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


# define the LightningModule
class LLMModel(pl.LightningModule):
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
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            quantization_config=q_config,
            **config.model_params
        )
        model.gradient_checkpointing_enable()
        model.train()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)      
        self.logger_samples = []
        if use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            logger.info(f"Using Lora with q_lora={use_q_lora}")
            lora_config = LoraConfig(**OmegaConf.to_container(config.lora_params))
            if use_q_lora:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

            self.model = get_peft_model(model, lora_config)
            self.model.enable_input_require_grads()
            self.model.print_trainable_parameters()
        else:
            self.model = model
            self.model = torch.compile(self.model) if config.use_compile else self.model
            self.model.use_neftune = config.use_neftune
            self.model.neft_alpha = config.neft_alpha  
        
    def prepare_dataset(self, config):
        dataset_class = get_class(config.dataset.name)
        train_dataset = dataset_class(
            dataset_args=config.dataset.train_dataset,
            tokenizer=self.tokenizer,
            **config.dataset,
        )
        val_dataset = dataset_class(
            dataset_args=config.dataset.val_dataset,
            tokenizer=self.tokenizer,
            **config.dataset,
        )
        bsz = config.trainer.batch_size
        train_dataloader = train_dataset.build_dataloader(batch_size=bsz)
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataset.build_dataloader(batch_size=bsz)
        return train_dataset, train_dataloader

    def forward(self, batch):
        for k, v in batch.items():
            batch[k] = v.to(self.target_device)

        out = self.model(**batch)
        loss = out["loss"]
        return loss

    @torch.no_grad()
    def eval_model(self, logger, current_epoch, global_step):
        self.model.eval()
        val_loss = 0
        total_val_steps = 0
        val_steps = self.config.trainer.get("eval_samples", 100)
        for batch in self.val_dataloader:
            for k, v in batch.items():
                batch[k] = v.to(self.target_device)
                
            loss = self.model(**batch)["loss"]
            val_loss += loss.item()
            if val_steps < 0:
                break
            val_steps -= 1
            total_val_steps += 1

        val_loss /= total_val_steps
        logger.log_metrics({"val_loss": val_loss}, step=global_step)
        self.model.train()

    @rank_zero_only
    def generate_samples(self, logger, current_epoch, global_step):
        config = self.config.sampling
        prompts = list(config.prompts)
        self.model.eval()
        for curr_prompt in prompts:
            ret = self.val_dataset.preprocess(curr_prompt)
            for k, v in ret.items():
                ret[k] = v.to(self.target_device)
                
            generated_output = _unwrap_objects(self.model).generate(
                input_ids=ret["input_ids"][0],
                attention_mask=ret["attention_mask"][0],
                max_length=config.max_length,
            )
            generated_text = self.tokenizer.decode(
                generated_output[0], skip_special_tokens=True
            )
            self.logger_samples.append([global_step, curr_prompt, generated_text])

        columns = ["global_step", "inputs", "predictions"]
        logger.log_text(key="generated_samples", columns=columns, data=self.logger_samples)
        self.model.train()

    @rank_zero_only
    def save_checkpoint(self, model_path, metadata):
        cfg = self.config.trainer
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        logger.info(f"Saved model to {model_path}")
