import tempfile
from huggingface_hub import Repository
from huggingface_hub.constants import ENDPOINT
from pytorch_lightning import Callback

# Modified: https://github.com/nateraw/hf-hub-lightning/blob/main/hf_hub_lightning/callback.py

class HuggingFaceHubCallback(Callback):
    def __init__(
        self,
        repo_name,
        use_auth_token=True,
        git_user=None,
        git_email=None,
        private=True,
        every_n_steps=None,
        every_n_epochs=1,
        **kwargs
    ):
        self.repo_owner, self.repo_name = repo_name.rstrip("/").split("/")[-2:]
        self.repo_namespace = f"{self.repo_owner}/{self.repo_name}"
        self.repo_url = f"{ENDPOINT}/{self.repo_namespace}"
        self.use_auth_token = use_auth_token if use_auth_token != "" else True
        self.git_user = git_user
        self.git_email = git_email
        self.private = private
        self.repo = None
        self.every_n_steps=every_n_steps
        self.every_n_epochs=every_n_epochs

    def on_init_end(self, trainer):
        self.repo = Repository(
            tempfile.TemporaryDirectory().name,
            clone_from=self.repo_url,
            use_auth_token=self.use_auth_token,
            git_user=self.git_user,
            git_email=self.git_email,
            revision=None, 
            private=self.private,
        )
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.every_n_steps:
            return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        
        if trainer.global_step % self.every_n_steps == 0 and trainer.global_step > 0:
            with self.repo.commit(f"Add/Update Model: Step {trainer.global_step}", blocking=False, auto_lfs_prune=True):
                trainer.save_checkpoint(f"model-e{trainer.current_epoch}-s{trainer.global_step}.ckpt")  

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.every_n_epochs:
            return super().on_train_epoch_end(self, trainer, pl_module)
        
        if trainer.current_epoch % self.every_n_epochs == 0:
            with self.repo.commit(f"Add/Update Model: epoch {trainer.current_epoch}", blocking=False, auto_lfs_prune=True):
                trainer.save_checkpoint(f"model-e{trainer.current_epoch}.ckpt")