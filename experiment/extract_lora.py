import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file

def process_file(src, dst, file_type):
    state_dict = torch.load(src, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    lorasd = {k.replace("lora.", ""): v for k, v in state_dict.items()}
    
    print(f"Saving to {dst}")
    if file_type == "pt":
        torch.save(lorasd, dst)
    elif file_type == "safetensors":
        save_file(lorasd, dst)
    else:
        raise ValueError("Invalid file_type. It must be 'pt' or 'safetensors'")

def process_directory(src, dst=None, file_type="pt"):
    src_path = Path(src)
    src_dir = src_path if src_path.is_dir() else src_path.parent
    dst_path = Path(dst) if dst else None
    
    if src_path.is_file() and src_path.suffix == '.ckpt':
        dst_file = dst_path / (src_path.stem + '.' + file_type) if dst_path.is_dir() else dst_path
        process_file(src_path, dst_file, file_type)
    else:
        for pt_file in src_path.rglob('*.ckpt'):
            if dst_path:
                relative = pt_file.relative_to(src_dir)
                target_dir = dst_path / relative.parent
                target_dir.mkdir(parents=True, exist_ok=True)
            else:
                target_dir = pt_file.parent

            dst_file = target_dir / (pt_file.stem + '.' + file_type)
            process_file(pt_file, dst_file, file_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, type=str, required=True, help="Path to the file or directory to process.")
    parser.add_argument("--dst", default=None, type=str, help="Path to the destination file or directory.")
    parser.add_argument("--type", default="pt", type=str, choices=["pt", "safetensors"], help="File type to save as.")
    args = parser.parse_args()

    assert args.src is not None, "Must provide a source path!"

    process_directory(args.src, args.dst, args.type)