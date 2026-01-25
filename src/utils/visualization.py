"""
Visualization utilities
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar

import cv2

if TYPE_CHECKING:
    from types import TracebackType

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

    def __enter__(self) -> FPSAnnotator:
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
    BLACK: ClassVar[Color]
    WHITE: ClassVar[Color]
    GREY: ClassVar[Color]
    RED: ClassVar[Color]
    GREEN: ClassVar[Color]
    BLUE: ClassVar[Color]
    YELLOW: ClassVar[Color]
    CYAN: ClassVar[Color]
    MAGENTA: ClassVar[Color]
    WARNING: ClassVar[Color]
    ERROR: ClassVar[Color]
    SUCCESS: ClassVar[Color]
    INFO: ClassVar[Color]

    def __new__(cls, r: int, g: int, b: int) -> Color:
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))
        return super().__new__(cls, (r, g, b))

    def __repr__(self) -> str:
        swatch = self.ansi('■■')
        return f'<{swatch} Color({self.r}, {self.g}, {self.b})>'

    def __add__(self, other: Color) -> Color:
        if not isinstance(other, Color):
            raise NotImplementedError

        return Color(self.r + other.r, self.g + other.g, self.b + other.b)

    def __sub__(self, other: Color) -> Color:
        if not isinstance(other, Color):
            raise NotImplementedError

        return Color(self.r - other.r, self.g - other.g, self.b - other.b)

    def __mul__(self, factor: float | int) -> Color:
        return Color(int(self.r * factor), int(self.g * factor), int(self.b * factor))

    @property
    def r(self) -> int:
        return self[0]

    @property
    def g(self) -> int:
        return self[1]

    @property
    def b(self) -> int:
        return self[2]

    @property
    def rgb(self) -> tuple[int, int, int]:
        return self

    @property
    def bgr(self) -> tuple[int, int, int]:
        return self.rgb[::-1]

    @property
    def hex(self) -> str:
        return f'#{self.r:02x}{self.g:02x}{self.b:02x}'

    @property
    def luminance(self) -> float:
        return 0.299 * self.r + 0.587 * self.g + 0.114 * self.b

    def ansi(self, text: str, bg: bool = False) -> str:
        mode = 48 if bg else 38
        return f'\033[{mode};2;{self.r};{self.g};{self.b}m{text}\033[0m'

    @classmethod
    def from_hex(cls, hex_code: str) -> Color:
        hex_code = hex_code.lstrip('#')

        if len(hex_code) != 6:
            raise ValueError

        return cls(
            *(
                int(hex_code[i:i + 2], 16)
                for i in (0, 2, 4)
            )
        )

    @classmethod
    def by_name(cls, name: str) -> Color:
        name = name.upper()

        if hasattr(cls, name):
            return getattr(cls, name)

        return cls.WHITE


Color.BLACK = Color(0, 0, 0)
Color.WHITE = Color(255, 255, 255)
Color.GREY = Color(128, 128, 128)

Color.RED = Color(255, 0, 0)
Color.GREEN = Color(0, 255, 0)
Color.BLUE = Color(0, 0, 255)

Color.YELLOW = Color.RED + Color.GREEN
Color.CYAN = Color.GREEN + Color.BLUE
Color.MAGENTA = Color.RED + Color.BLUE

Color.WARNING = Color.YELLOW
Color.ERROR = Color.RED * 0.9
Color.SUCCESS = Color.GREEN
Color.INFO = Color.CYAN
