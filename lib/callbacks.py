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
    ):
        self.repo_owner, self.repo_name = repo_name.rstrip("/").split("/")[-2:]
        self.repo_namespace = f"{self.repo_owner}/{self.repo_name}"
        self.repo_url = f"{ENDPOINT}/{self.repo_namespace}"
        self.use_auth_token = use_auth_token if use_auth_token != "" else True
        self.git_user = git_user
        self.git_email = git_email
        self.private = private
        self.repo = None

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

    def on_train_epoch_end(self, trainer, pl_module):
        with self.repo.commit("Add/Update Model"):
            trainer.save_checkpoint(f"model-e{trainer.current_epoch}.ckpt")