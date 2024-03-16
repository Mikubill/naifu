import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

class StreamingTextDataset(IterableDataset):
    """A custom dataset that serves 1024-token blocks as input_ids == labels"""
    def __init__(
        self, 
        dataset_path: str, 
        tokenizer: str,
        batch_size: int = 16, 
        block_len = 300,
        eot_token_id = 50256, # gpt2
        rank: int = 0,
        **kwargs
    ):
        self.data = []
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.eot_token_id = eot_token_id
        self.batch_size = batch_size
        self.block_len = block_len

    def __iter__(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            buffer = []
            for line in file:
                input_ids = self.tokenizer.encode(line.strip())
                input_ids = input_ids[:self.block_len-1] if self.block_len > 0 else input_ids
                input_ids.append(self.eot_token_id)  # Ensure the end token is added
                
                input_ids = torch.tensor(input_ids, dtype=torch.long)
                attention_mask = torch.ones(len(input_ids), dtype=torch.long)
                yield {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids}
        
    def __len__(self):
        # get line count
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)
                
    def collate_fn(self, batch):
        input_ids = pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=0)
        attention_mask = pad_sequence([x['attention_mask'] for x in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([x['labels'] for x in batch], batch_first=True, padding_value=-100)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    
    def build_dataloader(self, batch_size: int, shuffle: bool = False):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)