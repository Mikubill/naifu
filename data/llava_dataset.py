# copy from https://github.com/haotian-liu/LLaVA/blob/7440ec9ee37b0374c6b5548818e89878e38f3353/llava/train/train.py

import copy
from models.llava.mm_utils import process_anyres_image, tokenizer_image_token
from models.llava.constants import *
from typing import Dict
from torch.utils.data import Dataset
import os
import torch
from PIL import Image

import json
import data.conversation as conv_lib
from data.conversation import conv_templates
from packaging import version
import tokenizers
from torch.nn.utils.rnn import pad_sequence
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, 
        data_path, 
        image_folder, 
        tokenizer, 
        mm_config, 
        image_processor, 
        version="v1", 
        model_max_length: int = 512,
        is_multimodel: bool = True,
        **kwargs
    ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.loads(open(data_path, "r").read())
        self.default_conv = conv_templates[version] if version in conv_templates else conv_templates["vicuna_v1"]
        self.tokenizer = tokenizer
        self.mm_config = mm_config
        if hasattr(self.mm_config, "image_grid_pinpoints"):
             self.mm_config.image_grid_pinpoints = list(mm_config.image_grid_pinpoints)
             
        self.list_data_dict = list_data_dict
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.is_multimodal = is_multimodel
        self.model_max_length = model_max_length

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.image_folder
            processor = self.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image_size = image.size
            if self.mm_config.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.mm_config.image_aspect_ratio == 'anyres':
                image = process_anyres_image(image, processor, self.mm_config.image_grid_pinpoints)
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                
            sources = self.preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            
        data_dict = self.preprocess(sources, has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0]
            )

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
        elif self.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image_size'] = (crop_size['height'], crop_size['width'])
        
        return data_dict
    
    def preprocess_multimodal(self, sources) -> Dict:
        if not self.is_multimodal:
            return sources

        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    if "mmtag" in self.default_conv.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
                replace_token = DEFAULT_IMAGE_TOKEN
                if self.mm_config.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

        return sources
    
    def _tokenize_fn(self, strings):
        """Tokenize a list of strings."""
        tokenized_list = [
            self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def _mask_targets(self, target, tokenized_lens, speakers):
        # cur_idx = 0
        cur_idx = tokenized_lens[0]
        tokenized_lens = tokenized_lens[1:]
        target[:cur_idx] = IGNORE_INDEX
        for tokenized_len, speaker in zip(tokenized_lens, speakers):
            if speaker == "human":
                target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
            cur_idx += tokenized_len

    def _add_speaker_and_signal(self, header, source, get_conversation=True):
        """Add speaker and start/end signal on each round."""
        BEGIN_SIGNAL = "### "
        END_SIGNAL = "\n"
        conversation = header
        for sentence in source:
            from_str = sentence["from"]
            if from_str.lower() == "human":
                from_str = self.default_conv.roles[0]
            elif from_str.lower() == "gpt":
                from_str = self.default_conv.roles[1]
            else:
                from_str = 'unknown'
            sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                                sentence["value"] + END_SIGNAL)
            if get_conversation:
                conversation += sentence["value"]
        conversation += BEGIN_SIGNAL
        return conversation

    def preprocess_plain(self, sources):
        tokenizer = self.tokenizer
        # add end signal and concatenate together
        conversations = []
        for source in sources:
            assert len(source) == 2
            assert DEFAULT_IMAGE_TOKEN in source[0]['value']
            source[0]['value'] = DEFAULT_IMAGE_TOKEN
            conversation = source[0]['value'] + source[1]['value'] + self.default_conv.sep
            conversations.append(conversation)
        # tokenize conversations
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
        targets = copy.deepcopy(input_ids)
        for target, source in zip(targets, sources):
            tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
            target[:tokenized_len] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets)


    def preprocess(self, sources, has_image: bool = False):
        tokenizer = self.tokenizer
        """
        Given a list of sources, each is a conversation list. This transform:
        1. Add signal '### ' at the beginning each sentence, with end signal '\n';
        2. Concatenate conversations together;
        3. Tokenize the concatenated conversation;
        4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
        """
        if self.default_conv.sep_style == conv_lib.SeparatorStyle.PLAIN:
            return self.preprocess_plain(sources)
        if self.default_conv.sep_style == conv_lib.SeparatorStyle.LLAMA_2:
            return self.preprocess_llama_2(sources, has_image=has_image)
        if self.default_conv.sep_style == conv_lib.SeparatorStyle.MPT:
            return self.preprocess_mpt(sources)
        if self.default_conv.version.startswith("v1"):
            return self.preprocess_v1(sources, has_image=has_image)
        # add end signal and concatenate together
        conversations = []
        for source in sources:
            header = f"{self.default_conv.system}\n\n"
            conversation = self._add_speaker_and_signal(header, source)
            conversations.append(conversation)
        # tokenize conversations
        def get_tokenize_len(prompts):
            return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

        if has_image:
            input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
        else:
            conversations_tokenized = self._tokenize_fn(conversations)
            input_ids = conversations_tokenized["input_ids"]

        targets = copy.deepcopy(input_ids)
        for target, source in zip(targets, sources):
            if has_image:
                tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
            else:
                tokenized_lens = self._tokenize_fn([header] + [s["value"] for s in source])["input_ids_lens"]
            speakers = [sentence["from"] for sentence in source]
            self._mask_targets(target, tokenized_lens, speakers)

        return dict(input_ids=input_ids, labels=targets)

    def preprocess_llama_2(self, sources, has_image: bool = False):        
        tokenizer = self.tokenizer
        conv = self.default_conv.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations

        if has_image:
            input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        else:
            input_ids = tokenizer(
                conversations,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids

        targets = input_ids.clone()

        assert conv.sep_style == conv_lib.SeparatorStyle.LLAMA_2
        # Mask targets
        sep = "[/INST] "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )
    
    def preprocess_v1(self, sources, has_image: bool = False):        
        tokenizer = self.tokenizer
        conv = self.default_conv.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations
        if has_image:
            input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        else:
            input_ids = tokenizer(
                conversations,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids

        targets = input_ids.clone()

        assert conv.sep_style == conv_lib.SeparatorStyle.TWO

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                    round_len -= 1
                    instruction_len -= 1

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )
        
    def preprocess_mpt(self, sources) -> Dict:
        conv = self.default_conv.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations
        input_ids = torch.stack([tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        targets = input_ids.clone()
        assert conv.sep_style == conv_lib.SeparatorStyle.MPT

        # Mask targets
        sep = conv.sep + conv.roles[1]
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
            for conv_idx in range(3, len(rounds), 2):
                re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
            cur_len = 0
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(re_rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(tokenizer_image_token(rou, self.tokenizer)) + len(tokenizer_image_token(conv.sep, self.tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer))
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < self.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )
        
    def padding_collate_fn(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['image_sizes'] = [instance['image_size'] for instance in instances]
            
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        return batch
                
    def build_dataloader(self, batch_size: int, shuffle: bool = False):
        dataloader = torch.utils.data.DataLoader(
            self, 
            batch_size=batch_size, 
            num_workers=8,
            shuffle=shuffle, 
            collate_fn=self.padding_collate_fn
        )
        return dataloader