import functools
import sys
import tarfile
import requests
import torch

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from torchvision import transforms
from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Code ref: https://github.com/7eu7d7/pixiv_AI_crawler
def download_file(url, fname="model"):
    print(f'Downloading: "{url}" to {fname}\n')

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fname, "wb") as f:
            pbar = tqdm(total=int(r.headers["Content-Length"]))
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


class ConvNextClassifier:
    def __init__(self, model_path, threshold=[2.0, 0.4, 0.2]):
        model_fs = Path(model_path) / "checkpoint-best_t5.pth"
        if not model_fs.is_file():
            # pretrained: cls=['others', 'high quality', 'yuri']
            model_url = "http://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/checkpoint-best_t5.pth"
            download_file(model_url, fname=model_fs)

        self.net = create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=3,
            drop_path_rate=0,
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
        self.load_state_dict(
            self.net, torch.load(model_fs, map_location="cpu"), prefix=""
        )
        self.device = torch.device("cuda")
        self.net.to(self.device).eval()
        self.threshold = threshold

    def check(self, img) -> bool:
        img = self.transform(img).to(self.device).unsqueeze(0)
        pred = self.net(img)
        cls = pred.view(-1).argmax().item()

        conf = torch.softmax(pred, dim=-1)[0, cls]
        if conf.detach() < self.threshold[cls]:
            print(cls, conf)
            return False

        return True

    def load_state_dict(
        self, model, state_dict, prefix="", ignore_missing="relative_position_index"
    ):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix=prefix)
        warn_missing_keys = []
        ignore_missing_keys = []
        for key in missing_keys:
            keep_flag = True
            for ignore_key in ignore_missing.split("|"):
                if ignore_key in key:
                    keep_flag = False
                    break
            if keep_flag:
                warn_missing_keys.append(key)
            else:
                ignore_missing_keys.append(key)

        missing_keys = warn_missing_keys

        if len(missing_keys) > 0:
            print(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            print(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(ignore_missing_keys) > 0:
            print(
                "Ignored weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, ignore_missing_keys
                )
            )
        if len(error_msgs) > 0:
            print("\n".join(error_msgs))


if __name__ == "__main__":
    cls = ConvNextClassifier(model_path="../animesfw")
    for _, img in tqdm(enumerate(sys.argv[1:])):
        im = Image.open(img)
        print(img, cls.check(im))
