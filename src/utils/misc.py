from typing import Literal

import torch


def infer_device() -> Literal['cpu', 'cuda']:
    return 'cuda' if torch.cuda.is_available() else 'cpu'
