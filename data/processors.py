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
    return inputs

def shuffle_prompts(e: Entry):
    e.prompt = e.prompt.split(", ")
    random.shuffle(e.prompt)
    e.prompt = ", ".join(e.prompt)
    return e

def dropout_tags(tokens, dropout=0):
    if dropout <= 0:
        return tokens
    l = []
    for token in tokens:
        if random.random() >= dropout:
            l.append(token)
    return l

def shuffle_prompts_sdstyle(e: Entry):
    # constrants
    shuffle_caption = True
    token_warmup_step = 0 # unsupported
    caption_tag_dropout_rate = 0
    caption_separator = ","
    keep_tokens_separator = "|||"
    replacements = {}
    
    if keep_tokens_separator not in e.prompt:
        return e
    
    caption = e.prompt
    fixed_part, flex_part = caption.split(keep_tokens_separator, 1)
    fixed_tokens = [t.strip() for t in fixed_part.split(caption_separator) if t.strip()]
    flex_tokens = [t.strip() for t in flex_part.split(caption_separator) if t.strip()]

    if shuffle_caption:
        random.shuffle(flex_tokens)
        
    # dropout flex tags by rate
    flex_tokens = dropout_tags(flex_tokens, caption_tag_dropout_rate)
    caption = ", ".join(fixed_tokens + flex_tokens)

    for str_from, str_to in replacements.items():
        if str_from == "":
            # replace all
            if type(str_to) == list:
                caption = random.choice(str_to)
            else:
                caption = str_to
        else:
            caption = caption.replace(str_from, str_to)
    
    e.prompt = caption
    return e