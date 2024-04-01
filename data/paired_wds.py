import io
from PIL import Image
import torch
from datasets import load_dataset
from torchvision import transforms


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.dataset = load_dataset(
            path=config.dataset.name,
            name=None,
            cache_dir=config.dataset.get("cache_dir", None),
        )[config.dataset.dataset_split]
        self.dataset = self.dataset.with_transform(self.preprocess_train)
        self.reso = config.dataset.resolution

        interp = transforms.InterpolationMode.BILINEAR
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(self.reso, interpolation=interp),
                transforms.CenterCrop(self.reso),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def preprocess_train(self, examples):
        all_pixel_values = []
        for col_name in ["jpg_0", "jpg_1"]:
            images = [
                Image.open(io.BytesIO(im_bytes)).convert("RGB")
                for im_bytes in examples[col_name]
            ]
            original_sizes = [(image.height, image.width) for image in images]
            pixel_values = [self.train_transforms(image) for image in images]
            all_pixel_values.append(pixel_values)

        # Double on channel dim, jpg_y then jpg_w
        im_tup_iterator = zip(*all_pixel_values)
        combined_pixel_values = []
        for im_tup, label_0 in zip(im_tup_iterator, examples["label_0"]):
            if label_0 == 0:
                im_tup = im_tup[::-1]
            combined_im = torch.cat(im_tup, dim=0)  # no batch dim
            combined_pixel_values.append(combined_im)

        examples.update(
            {
                "original_size_as_tuple": original_sizes,
                "crop_coords_top_left": [(0, 0)] * len(original_sizes),
                "target_size_as_tuple": [(self.reso, self.reso)] * len(original_sizes),
                "prompts": examples["caption"],
                "pixels": combined_pixel_values,
            }
        )
        return examples

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return example


def collate_fn(examples):
    pixel_values = torch.stack([e["pixels"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    orig_tup = [torch.tensor(e["original_size_as_tuple"]) for e in examples]
    crop_tup = [torch.tensor(e["crop_coords_top_left"]) for e in examples]
    target_tup = [torch.tensor(e["target_size_as_tuple"]) for e in examples]
    return_d = {
        "pixels": pixel_values,
        "original_size_as_tuple": torch.stack(orig_tup),
        "crop_coords_top_left": torch.stack(crop_tup),
        "target_size_as_tuple": torch.stack(target_tup),
        "prompts": [e["prompts"] for e in examples],
    }
    return return_d


def setup_hf_dataloader(config):
    train_dataset = PairedDataset(config)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=config.trainer.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )
    return train_dataset, dataloader
