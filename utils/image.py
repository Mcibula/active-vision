import numpy as np
from ultralytics.data.augment import LetterBox


def crop_zeros(src: np.ndarray) -> np.ndarray:
    if src.ndim < 2:
        raise ValueError

    y_nonzero, x_nonzero, _ = np.nonzero(src)
    y_up, y_down = np.min(y_nonzero), np.max(y_nonzero)
    x_left, x_right = np.min(x_nonzero), np.max(x_nonzero)

    return src[
        y_up:y_down,
        x_left:x_right
    ]


def resize_img(src: np.ndarray, size: tuple[int, ...]) -> np.ndarray:
    if src.ndim < 2:
        raise ValueError

    return LetterBox(
        new_shape=size,
        scale_fill=True,
        scaleup=True
    )(image=src)
