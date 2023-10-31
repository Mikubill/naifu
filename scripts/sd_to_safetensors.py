import argparse
import torch
from safetensors.torch import save_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--dst", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--half", action="store_true", help="Convert to half precision.")
    args = parser.parse_args()

    assert args.src is not None, "Must provide a model path!"
    assert args.dst is not None, "Must provide a checkpoint path!"

    state_dict = torch.load(args.src, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    
    if args.half:
        state_dict = {k: v.half().contiguous().to_dense() for k, v in state_dict.items()}
    else:
        state_dict = {k: v.contiguous().to_dense() for k, v in state_dict.items()}
    
    save_file(state_dict, args.dst)
