"""
Miscellaneous utilities
"""

import torch


def infer_device() -> torch.device:
    """
    Infer computation device based on the available hardware

    :return: `CUDA:0` PyTorch device if at least one GPU is available; `CPU` otherwise
    """

    return torch.device(
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )
