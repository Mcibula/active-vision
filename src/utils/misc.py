import torch


def infer_device() -> torch.device:
    return torch.device(
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )
