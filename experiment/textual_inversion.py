import re
from lib.model import StableDiffusionModel, get_class
import torch
from pathlib import Path
from lightning.pytorch import Callback
from transformers import CLIPTokenizer, CLIPTextModel
from lightning.pytorch.utilities import rank_zero_only
from lib.utils import rank_zero_print

class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        torch.save(embedding_data, filename)

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum
    
# ref: ao
class CustomEmbeddingsCallback(Callback):
    def __init__(self, config):
        self.embs = {}
        self.trainer = config.trainer
        self.config = config
        
        assert Path(config.weights_path).is_dir(), f"No such file or directory: {config.weights_path}"
        self.save_path = Path(config.trainer.save_path) if config.trainer.save_path != None else Path(config.weights_path) / "checkpoints" 
        self.save_path.mkdir(exist_ok=True)
      
    def setup_embs(self, model):
        vec_match = re.compile(r":(\d+)v$")
        concepts = self.config.concepts.freeze + self.config.concepts.trainable
        trainable_concepts = [vec_match.sub("", v) for v in self.config.concepts.trainable]   
        weights_path = Path(self.config.weights_path)
        if self.config.load_all:
            concepts.extend([k.stem for k in weights_path.glob("*.pt")])
            
        for n, v in enumerate(concepts):
            entry = vec_match.sub("", v)
            if self.embs.get(entry) is not None:
                continue
            fp = Path(weights_path / f"{entry}.pt")
            if not fp.is_file():
                size = vec_match.findall(v)
                size = int(size[0]) if len(size) > 0 else 2
                name, vec = self.create_emb(entry, size, model)
            else:
                name, vec = self.load_emb(fp, model)
            self.embs[name] = vec              
                    
        self.trainable_concepts = trainable_concepts
        self.clip_keywords = [' '.join(s) for s in self.make_token_names(self.embs)]
        self.reg_match = [re.compile(fr"(?:^|(?<=\s|,)){k}(?=,|\s|$)") for k in self.embs.keys()]
        self.use_wandb = False
        self.init_wandb()
    
    @rank_zero_only
    def init_wandb(self):
        if self.config.get("use_wandb"):
            import wandb
            self.use_wandb = True
            self.wandb_run = wandb.init(reinit=False)
    
    def preliminary_check(self, model):
        counter = {}
        for entry in model.dataset.entries:
            for name in self.embs.keys():
                if name in entry[1]: # prompt
                    counter[name] = 1
        
        notused = []
        for n, v in counter.items():
            if counter.get(n) == None and n in self.trainable_concepts:
                notused.append(n)
                
        if len(notused) > 0:        
            rank_zero_print(f"Some embeddings will not be trained due to a lack of corresponding prompt entries: {notused}")
                    
    def parse_prompt(self, prompt: str):
        """Parse a prompt string into a list of embedding names and a list of tokens.
        """
        for m, v in zip(self.reg_match, self.clip_keywords):
            prompt = m.sub(v, prompt)
        return prompt
    
    def create_emb(self, emb_name, nvpt, model):
        embedding_layer = model.text_encoder.get_input_embeddings()
        ids = model.tokenizer(emb_name, max_length=nvpt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        embedded = embedding_layer(ids.to(model.device)).squeeze(0)
        vec = torch.zeros((nvpt, embedded.shape[1]), device=model.device)

        for i in range(nvpt):
            vec[i] = embedded[i * int(embedded.shape[0]) // nvpt]
        
        emb_path = self.save_path / f"{emb_name}_s0.pt"
        embedding = Embedding(vec, emb_name, step=0)
        embedding.save(emb_path)
        rank_zero_print(f"Created: {emb_path} \t[{vec.shape[0]},{vec.shape[1]}]")
        return emb_name, vec

    @staticmethod
    def load_emb(path, model):
        filename = path.stem 
        data = torch.load(path, map_location='cpu')
        
        # textual inversion embeddings
        if 'string_to_param' in data:
            param_dict = data['string_to_param']
            if hasattr(param_dict, '_parameters'):
                param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
            assert len(param_dict) == 1, 'embedding file has multiple terms in it'
            emb = next(iter(param_dict.items()))[1]
        # diffuser concepts
        elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
            assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

            emb = next(iter(data.values()))
            if len(emb.shape) == 1:
                emb = emb.unsqueeze(0)
        else:
            raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")

        vec = emb.detach().to(model.device, dtype=torch.float32)
        rank_zero_print(f"Loaded: {path} \t[{vec.shape[0]},{vec.shape[1]}]")
        return filename, vec

    @staticmethod
    def make_token_names(embs):
        all_tokens = []
        for name, vec in embs.items():
            tokens = [f'emb-{name}-{i}' for i in range(len(vec))]
            all_tokens.append(tokens)
        return all_tokens
    
    @rank_zero_only 
    def save_emb(self, step, epoch, model):
        if self.use_wandb:
            import wandb
            model_artifact = wandb.Artifact(f"embeddings-{self.wandb_run.id}", type='model', metadata={
                'epochs_trained': epoch,
                'steps_trained': step
            })
        lengths = [(k, v.shape[0]) for k, v in self.embs.items()]
        params = model.text_encoder.get_input_embeddings().weight.data
        params = params[-sum([p[1] for p in lengths]):].detach().clone().cpu()
        for item in lengths:
            entry, length = item
            if entry in self.trainable_concepts or self.config.train_all:
                embedding = Embedding(params[:length].clone(), entry, step=step)
                embedding.save(self.save_path / f"{entry}_s{step}.pt")
                if self.use_wandb:
                    model_artifact.add_file(self.save_path / f"{entry}_s{step}.pt")
            params = params[length:]
            
        if self.use_wandb:
            wandb.log_artifact(model_artifact, aliases=['latest', 'last', f'epoch {epoch}', f'step {step}'])
            
    def setup_clip(self, clip, tokenizer):
        """Adds custom embeddings to a CLIPTextModel. CLIPTokenizer is hooked to replace the custom embedding tokens with their corresponding CLIP tokens."""
        token_names = self.make_token_names(self.embs)
        token_names = [t for sublist in token_names for t in sublist]  # flatten nested list

        # add emb tokens to tokenizer
        n_added = tokenizer.add_tokens(token_names)
        assert n_added == sum([len(item) for item in self.embs.values()]), f"Unexpected number of tokens added: {n_added}. Try make the emb names are less nasty."

        # append TI embeddings to CLIP embedding table
        clip.resize_token_embeddings(len(tokenizer))
        
    def hook_clip(self, clip: CLIPTextModel, tokenizer: CLIPTokenizer, init_weight=True):
        n_added = sum([len(item) for item in self.embs.values()])
        emb_layer = clip.get_input_embeddings()
        
        if init_weight:
            delta_embeddings = torch.cat([item for item in self.embs.values()], dim=0)
            emb_layer.weight.data[-n_added:] = delta_embeddings
        
        if self.trainable_concepts or self.config.train_all:
            mask = torch.zeros_like(emb_layer.weight.data, dtype=torch.bool)
            offset = len(mask) - n_added
            for name, vec in self.embs.items():
                if name in self.trainable_concepts or self.config.train_all:
                    mask[offset:offset+len(vec)] = True
                offset += len(vec)
            
            mask.to(emb_layer.weight.device)
            emb_layer.weight.register_hook(lambda grad: grad.where(mask, mask))
            emb_layer.requires_grad_(True)
        else:
            emb_layer.requires_grad_(False)

        # hook tokenizer to replace emb tokens with their corresponding CLIP tokens
        original_prepare_for_tokenization = tokenizer.prepare_for_tokenization
        outer = self

        def prepare_for_tokenization(self, text: str, is_split_into_words: bool = False, **kwargs):
            text = outer.parse_prompt(text)
            r = original_prepare_for_tokenization(text, is_split_into_words, **kwargs)
            return r
        tokenizer.prepare_for_tokenization = prepare_for_tokenization.__get__(tokenizer, CLIPTokenizer)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):    
        if self.config.trainer.max_epochs > 0 and trainer.current_epoch > self.config.trainer.max_epochs:
            return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
            
        if self.trainer.every_n_steps == -1:
            return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        
        if trainer.global_step % self.config.trainer.every_n_steps == 0 and trainer.global_step > 0:
            self.save_emb(trainer.global_step, trainer.current_epoch, pl_module)
        
    def on_train_epoch_end(self, trainer, pl_module):
        if self.config.trainer.max_epochs > 0:
            if trainer.current_epoch == self.config.trainer.max_epochs:
                pl_module.text_encoder.get_input_embeddings().weight.register_hook(lambda grad: torch.zeros_like(grad))
            elif trainer.current_epoch > self.config.trainer.max_epochs:
                return super().on_train_epoch_end(trainer, pl_module)
        
        if self.trainer.every_n_epochs == -1:
            return super().on_train_epoch_end(trainer, pl_module)
        
        if trainer.current_epoch % self.config.trainer.every_n_epochs == 0 and trainer.current_epoch > 0:
            self.save_emb(trainer.global_step, trainer.current_epoch, pl_module)
            
    def setup(self, trainer, pl_module, stage: str):
        self.setup_embs(pl_module)
        self.setup_clip(pl_module.text_encoder, pl_module.tokenizer)
        self.init_weight = True
        original_configure_optimizers = pl_module.configure_optimizers
        
        outer = self
        def configure_optimizers(self):
            optimizers, _ = original_configure_optimizers()
            return outer.hook_optimizers(self, optimizers)
        
        if self.trainable_concepts or self.config.train_all:
            pl_module.configure_optimizers = configure_optimizers.__get__(pl_module, StableDiffusionModel)
        else:
            self.hook_clip(pl_module.text_encoder, pl_module.tokenizer, self.init_weight)
        
    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        self.init_weight = False

    def hook_optimizers(self, pl_module, optimizers):
        self.hook_clip(pl_module.text_encoder, pl_module.tokenizer, self.init_weight)
        self.preliminary_check(pl_module)

        if self.config.freeze_unet:
            pl_module.unet.requires_grad_(False)
        
        new_lr, scaled = pl_module.get_scaled_lr(self.config.trainer.lr)
        if scaled:
            self.config.trainer.lr = new_lr
            rank_zero_print(f"Using scaled embeddings LR: {self.config.trainer.lr}")
        
        assert len(optimizers) == 1, "Expect only one optimizer in StableDiffusionModel"
        if not self.config.freeze_unet:
            param_to_optimize = optimizers[0].param_groups
        else:
            param_to_optimize = []
            del optimizers[0]

        param_to_optimize.append(
            {'params': pl_module.text_encoder.get_input_embeddings().parameters(), 'lr': self.config.trainer.lr}
        )
        optimizer = get_class(pl_module.config.optimizer.name)(
            param_to_optimize, **pl_module.config.optimizer.params
        )
        scheduler = get_class(pl_module.config.lr_scheduler.name)(
            optimizer=optimizer, **pl_module.config.lr_scheduler.params
        )

        return [[optimizer], [scheduler]]