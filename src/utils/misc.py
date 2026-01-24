from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from types import TracebackType


def infer_device() -> torch.device:
    return torch.device(
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )


class timer:
    def __init__(self, name: str = '') -> None:
        self.name: str = name

    def __enter__(self) -> timer:
        self.start: float = time.perf_counter()
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType
    ) -> None:
        self.time = time.perf_counter() - self.start
        self.readout = f'Operation{" " + self.name if self.name else ""}: {self.time * 1000:.2f} ms'

        print(self.readout)
