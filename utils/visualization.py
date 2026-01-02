"""
Visualization utilities
"""

import time
from types import TracebackType

import cv2
import numpy as np


class FPSAnnotator:
    """
    Context manager annotating frames to be displayed with FPS
    based on the measured duration of the body's execution
    """

    def __init__(self, frames_per_cycle: int = 1) -> None:
        """
        Initialize the annotator

        :param frames_per_cycle: Number of frames processed in the body of this context manager before exiting
        """

        self._fpc: int = frames_per_cycle

    def __enter__(self) -> 'FPSAnnotator':
        self._t0: float = time.time()
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType
    ) -> None:
        self._dt: float = time.time() - self._t0
        self._fps: float = self._fpc / self._dt

    def put_fps(self, frame: np.ndarray) -> None:
        """
        Annotate a frame in-place with the calculated FPS

        :param frame: Frame to annotate
        """

        cv2.putText(
            img=frame,
            text=f'FPS: {self._fps:.2f}',
            org=(10, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 0),
            thickness=2
        )


class Color(tuple):
    def __new__(cls, r: int, g: int, b: int) -> 'Color':
        return super().__new__(cls, (r, g, b))

    @property
    def rgb(self) -> tuple[int, int, int]:
        return self

    @property
    def bgr(self) -> tuple[int, int, int]:
        return self[::-1]

    def __add__(self, other: 'Color') -> 'Color':
        return Color(
            r=min(self[0] + other[0], 255),
            g=min(self[1] + other[1], 255),
            b=min(self[2] + other[2], 255)
        )


BLACK = Color(0, 0, 0)
RED = Color(255, 0, 0)
GREEN = Color(0, 255, 0)
BLUE = Color(0, 0, 255)
YELLOW = GREEN + RED
CYAN = GREEN + BLUE
WHITE = RED + GREEN + BLUE

