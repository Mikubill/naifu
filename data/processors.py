from dataclasses import dataclass
import torch
import random


@dataclass
class Entry:
    """
    This class represents an entry in a batch of image data. Each entry contains information about an image and its associated prompt.

    Attributes:
        is_latent (bool): A flag indicating whether the image is in latent space.
        pixel (torch.Tensor): The pixel data of the image.
        prompt (str): The prompt associated with the image.
        extras (dict): A dictionary to store any extra information associated with the image.
        
    """
    is_latent: bool
    pixel: torch.Tensor
    prompt: str
    extras: dict = None


def identical(inputs: Entry):
    """
    This method processes an `Entry` object.

    Args:
        inputs (Entry): An `Entry` objects. Each `Entry` object has the following attributes:
            is_latent (bool): A flag indicating whether the image is in latent space.
            pixel (torch.Tensor): The pixel data of the image.
            prompt (str): The prompt associated with the image.
            original_size (tuple[int, int]): The original height and width of the image.
            extras (dict): A dictionary to store any extra information associated with the image.

    Returns:
        inputs (Entry): The processed list of `Entry` objects.

    This method allows users to alter any item in the batch, such as changing the prompt order or enhancing the image.
    """
    return inputs

def shuffle_prompts(e: Entry):
    e.prompt = e.prompt.split(", ")
    random.shuffle(e.prompt)
    e.prompt = ", ".join(e.prompt)
    return e