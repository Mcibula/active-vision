from typing import Literal

import torch

def infer_device() -> torch.device:
    return torch.device(
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )


def infer_device() -> Literal['cpu', 'cuda']:
    return 'cuda' if torch.cuda.is_available() else 'cpu'
