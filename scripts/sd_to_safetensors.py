import argparse
import torch
from safetensors.torch import save_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--dst", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()

    assert args.src is not None, "Must provide a model path!"
    assert args.dst is not None, "Must provide a checkpoint path!"

    state_dict = torch.load(args.src, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    
    from safetensors.torch import save_file
    save_file({k: v.contiguous().to_dense() for k, v in state_dict.items()}, args.dst)
