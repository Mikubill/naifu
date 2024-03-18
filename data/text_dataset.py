import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from common.utils import get_class
from typing import Optional, Callable, Dict, Any

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
        mask_prompt: bool = True,
        ignore_index: int = -100,
        cutoff_len: int = 2048,
        transform: Optional[Callable[[Any], Any]] = None,
        **kwargs,
    ):
        self.data = load_dataset(**dataset_args)
        self.tokenizer = tokenizer
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

# https://github.com/OpenAccess-AI-Collective/axolotl/raw/d485a083938e995979d1a566bb6a2f876075c667/src/axolotl/utils/chat_templates.py
def build_chat_template(user_choice: str):
    """
    Finds the correct chat_template for the tokenizer_config.

    Args:
        user_choice (str): The user's choice of template.

    Returns:
        str: The chosen template string.

    Raises:
        ValueError: If the user_choice is not found in the templates.
    """

    templates = {
        "alpaca": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Instruction: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '### Response: ' + message['content'] + eos_token}}{% endif %}{% endfor %}",
        "inst": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",  # I don't know what this one is called. Used by Mistral/Mixtral.
        "chatml": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'You are a helpful assistant.' %}{% endif %}{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{{'<|im_start|>system\n' + system_message + '<|im_end|>\n'}}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        "gemma": "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}",
    }

    if user_choice in templates:
        return templates[user_choice]

    raise ValueError(f"Template '{user_choice}' not found.")

class ChatMLDataset(Dataset):
    def __init__(
        self, 
        dataset_args: Dict[str, Any],
        tokenizer: object, 
        max_seq_length: int = -1,
        ignore_index: int = -100,
        mask_inputs: str = True,
        cache_prompts: bool = True,
        **kwargs,
    ):
        self.data = load_dataset(**dataset_args)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ignore_index = ignore_index
        self.cached_data_dict = {}
        self.cache_prompts = cache_prompts
        self.mask_inputs = mask_inputs
        
    def get_conversation_thread(self, prompt):
        conversations = prompt[0]["conversations"]
        # remap roles - allow for assistant turn
        role_map = {
            "human": "user",
            "user": "user",
            "assistant": "assistant",
            "gpt": "assistant",
            "system": "system",
            "function_response": "function_response",
        }
        turns = [
            {"role": role_map[t["from"]], "content": t["value"]} for t in conversations
        ]
        return turns

    def preprocess(self, prompts) -> Dict:
        """Preprocesses the data for supervised fine-tuning."""
        build_prompt = lambda turn, **kwargs: self.tokenizer.apply_chat_template(
            turn,
            truncation=True,
            max_length=self.max_seq_length,
            **kwargs
        )
        turns = self.get_conversation_thread(prompts)
        prompt_ids = build_prompt([turns[0]], add_generation_prompt=True)
        input_ids = build_prompt(turns)
        
        if self.mask_inputs:
            user_prompt_len = len(prompt_ids)
            labels = [-100] * user_prompt_len + input_ids[user_prompt_len:]
        else:
            labels = input_ids
            
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        return dict(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            attention_mask=attention_mask
        )

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
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = self.preprocess([self.data[i]])
        if self.cache_prompts:
            self.cached_data_dict[i] = ret
        return ret
