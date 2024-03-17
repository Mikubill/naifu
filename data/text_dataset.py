import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from common.utils import get_class
from typing import List, Optional, Callable, Dict, Any

def padding_collate_fn(batch):
    input_ids = pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([x['attention_mask'] for x in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([x['labels'] for x in batch], batch_first=True, padding_value=-100)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

class SimpleTextDataset(Dataset):
    """A custom dataset that serves 1024-token blocks as input_ids == labels"""
    def __init__(
        self, 
        dataset_path: str, 
        tokenizer: str,
        batch_size: int = 16, 
        block_len = 300,
        eot_token_id = 50256, # gpt2
        **kwargs
    ):
        self.data = load_dataset('text', data_files=dataset_path)
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.eot_token_id = eot_token_id
        self.batch_size = batch_size
        self.block_len = block_len
        self.data = self.data.shuffle()['train']

    def __getitem__(self, idx):
        line = self.data[idx]["text"]
        input_ids = self.tokenizer.encode(line.strip())
        input_ids = input_ids[:self.block_len-1] if self.block_len > 0 else input_ids
        input_ids.append(self.eot_token_id)  # Ensure the end token is added
                
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids}
        
    def __len__(self):
        return len(self.data)
    
    def build_dataloader(self, batch_size: int, shuffle: bool = False):
        dataloader = torch.utils.data.DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=padding_collate_fn
        )
        return dataloader
    
class TextDataset(Dataset):
    def __init__(
        self, 
        dataset_args: Dict[str, Any],
        tokenizer: object, 
        prompt_style: Dict[str, Any],
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        ignore_index: int = -100,
        cutoff_len: int = 2048,
        transform: Optional[Callable[[Any], Any]] = None,
        **kwargs,
    ):
        self.data = load_dataset(**dataset_args)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.ignore_index = ignore_index
        self.transform = transform
        self.cutoff_len = cutoff_len
        self.prompt_style = get_class(prompt_style["name"])(**prompt_style)
        
    def tokenize(self, prompt: str, add_eos: bool = False):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
        )

        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and add_eos
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result
    
    def build_dataloader(self, batch_size: int, shuffle: bool = False):
        dataloader = torch.utils.data.DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=padding_collate_fn
        )
        return dataloader
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        example = self.data[idx]
        if self.transform is not None:
            example = self.transform(example)
            
        prompt, outputs = self.prompt_style.apply(example)
        prompt_and_response = prompt + outputs
        encoded_prompt = self.tokenize(prompt)
        encoded_prompt_and_response = self.tokenize(prompt_and_response, add_eos=True)

        # The labels are the full prompt with response, but with the prompt masked out
        if self.mask_prompt:
            prompt_len = len(encoded_prompt["input_ids"])
            encoded_prompt_and_response["labels"] = \
                [-100] * prompt_len + encoded_prompt_and_response["labels"][prompt_len:]
        
        for k in encoded_prompt_and_response.keys():
            encoded_prompt_and_response[k] = torch.LongTensor(encoded_prompt_and_response[k])
        
        return encoded_prompt_and_response
