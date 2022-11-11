import torch


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def get_gpu_ram() -> str:
    """
    Returns memory usage statistics for the CPU, GPU, and Torch.
    :return:
    """
    devid = torch.cuda.current_device()
    return f"GPU.{devid} {torch.cuda.get_device_name(devid)}"


def get_world_size() -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()
