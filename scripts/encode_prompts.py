import argparse
import hashlib
from typing import Optional
import h5py as h5
import json
import torch

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPTextModel


def sha1sum(txt):
    return hashlib.sha1(txt.encode()).hexdigest()


class PromptDataset(Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.is_json_index = False
        if self.root.is_file():
            assert self.root.suffix == ".json"
            js = json.loads(self.root.read_text())
            self.root = [p["train_caption"] for p in js.values()]
            self.is_json_index = True
        else:
            assert self.root.is_dir()
            # select all txt
            self.root = list(self.root.glob("**/*.txt"))
        print(f"Found {len(self.root)} prompts")

    def __getitem__(self, index):
        if self.is_json_index:
            prompt = self.root[index]
        else:
            path = self.root[index]
            prompt = path.read_text().strip()
        return prompt, sha1sum(prompt)

    def __len__(self):
        return len(self.root)
    

def process_input_ids(input_ids, tokenizer, max_length):
    if max_length > tokenizer.model_max_length:
        input_ids = input_ids.squeeze(0)
        iids_list = []
        # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
        for i in range(1, max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
            ids_chunk = (
                input_ids[0].unsqueeze(0),  # BOS
                input_ids[i : i + tokenizer.model_max_length - 2],
                input_ids[-1].unsqueeze(0),
            )  # PAD or EOS
            ids_chunk = torch.cat(ids_chunk)
                
            # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
            # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
            if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                ids_chunk[-1] = tokenizer.eos_token_id
            # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
            if ids_chunk[1] == tokenizer.pad_token_id:
                ids_chunk[1] = tokenizer.eos_token_id

            iids_list.append(ids_chunk)

        input_ids = torch.stack(iids_list)  # 3,77
    return input_ids
    

def get_hidden_states_sdxl(
    prompt: str,
    max_token_length: int,
    tokenizer1: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    text_encoder1: CLIPTextModel,
    text_encoder2: CLIPTextModelWithProjection,
):
    device = next(text_encoder1.parameters()).device
    # self.tokenizer[0](captions, padding=True, truncation=True, return_tensors="pt").input_ids
    input_ids1 = tokenizer1(prompt, truncation=True, max_length=max_token_length,
        return_overflowing_tokens=False, padding="max_length", return_tensors="pt").input_ids
    input_ids2 = tokenizer2(prompt, truncation=True, max_length=max_token_length,
        return_overflowing_tokens=False, padding="max_length", return_tensors="pt").input_ids
    input_ids1 = process_input_ids(input_ids1, tokenizer1, max_token_length).to(device).unsqueeze(0)
    input_ids2 = process_input_ids(input_ids2, tokenizer2, max_token_length).to(device).unsqueeze(0)
    
    # input_ids: b,n,77 -> b*n, 77
    b_size = input_ids1.size()[0]
    input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))  # batch_size*n, 77
    input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))  # batch_size*n, 77

    # text_encoder1
    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    # text_encoder2
    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer

    pool2 = enc_out["text_embeds"]

    # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
    n_size = 1 if max_token_length is None else max_token_length // 75
    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

    if max_token_length is not None:
        # bs*3, 77, 768 or 1024
        # encoder1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
        states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer1.model_max_length):
            states_list.append(hidden_states1[:, i : i + tokenizer1.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
        hidden_states1 = torch.cat(states_list, dim=1)

        states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer2.model_max_length):
            chunk = hidden_states2[:, i : i + tokenizer2.model_max_length - 2] 
            states_list.append(chunk)
        states_list.append(hidden_states2[:, -1].unsqueeze(1)) 
        hidden_states2 = torch.cat(states_list, dim=1)
        pool2 = pool2[::n_size]      
    
    return hidden_states1, hidden_states2, pool2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="root directory of images")
    parser.add_argument("--output", "-o", type=str, required=True, help="output folder")
    parser.add_argument("--device", "-D", type=str, default="cuda:0", help="device")
    parser.add_argument("--model", "-m", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="model name")
    parser.add_argument("--num_workers", "-n", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--max_length", "-l", type=int, default=225, help="max length of prompt")
    parser.add_argument(
        "--compress",
        "-c",
        type=str,
        default='gzip',
        help="compression algorithm for output hdf5 file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    root = args.input
    opt = Path(args.output)
    device = args.device
    num_workers = args.num_workers
    max_length = args.max_length + 2
    
    p = StableDiffusionXLPipeline.from_pretrained(
        args.model, 
        unet=UNet2DConditionModel(), 
        vae=AutoencoderKL(), 
        image_encoder=None,
        feature_extractor=None, 
        torch_dtype=torch.float16,
    )
    text_encoder1, text_encoder2 = p.text_encoder.to(device), p.text_encoder_2.to(device)
    tokenizer1, tokenizer2 = p.tokenizer, p.tokenizer_2

    dataset = PromptDataset(root)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=num_workers)
    
    print(f"Starting encoding...")
    if not opt.exists():
        opt.mkdir()
        
    assert opt.is_dir(), f"{opt} is not a directory"
    h5_cache_file = opt / "prompt_cache.h5"
    file_mode = "w" if not h5_cache_file.exists() else "r+"
    with h5.File(h5_cache_file, file_mode, libver="latest") as f:
        with torch.no_grad():
            for i, (prompt, digest) in enumerate(tqdm(dataloader)):
                if f"{digest}.emb1" in f:
                    print(f"\033[33mWarning: {digest} is already cached. Skipping... \033[0m")
                    continue
                hidden_states1, hidden_states2, pool2 = get_hidden_states_sdxl(prompt, max_length, tokenizer1, tokenizer2, text_encoder1, text_encoder2)
                f.create_dataset(f"{digest}.emb1", data=hidden_states1.half().cpu().numpy(), compression=args.compress,)
                f.create_dataset(f"{digest}.emb2", data=hidden_states2.half().cpu().numpy(), compression=args.compress,)
                f.create_dataset(f"{digest}.pool2", data=pool2.half().cpu().numpy(), compression=args.compress,)