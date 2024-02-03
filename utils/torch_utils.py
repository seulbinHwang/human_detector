import platform

import torch


def get_device(use_gpu: bool = False) -> torch.device:
    """
    Returns the torch device to use for model inference.

    Args:
        use_gpu (bool): Whether to use GPU for model inference.

    Returns:
        torch.device: Torch device to use for model inference.
    """
    if use_gpu:
        device = get_torch_gpu_device()
    else:
        device = torch.device("cpu")
    return device


def get_torch_gpu_device(gpu_idx: int = 0) -> torch.device:
    if platform.system() == "Darwin" and platform.uname().processor == "arm":
        assert torch.backends.mps.is_available(
        ), "MPS is not available on this device."
        device = torch.device(f"mps:{gpu_idx}")
    else:
        assert torch.cuda.is_available(
        ), "CUDA is not available on this device."
        device = torch.device(f"cuda:{gpu_idx}")
    return device
