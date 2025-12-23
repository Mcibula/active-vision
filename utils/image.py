"""
Image manipulation utilities
"""

import cv2
import numpy as np
from ultralytics.data.augment import LetterBox


def crop_zeros(src: np.ndarray) -> np.ndarray:
    """
    Crop zeros around the center

    :param src: An input image
    :return: Cropped image
    """

    if src.ndim < 2:
        raise ValueError

    y_nonzero, x_nonzero, _ = np.nonzero(src)
    y_up, y_down = np.min(y_nonzero), np.max(y_nonzero)
    x_left, x_right = np.min(x_nonzero), np.max(x_nonzero)

    return src[
        y_up:y_down,
        x_left:x_right
    ]


def resize_img(src: np.ndarray, to_shape: tuple[int, ...]) -> np.ndarray:
    """
    Resize an image without padding or cropping

    :param src: An input image
    :param to_shape: New shape
    :return: Stretched image
    """

    if src.ndim < 2:
        raise ValueError

    return LetterBox(
        new_shape=to_shape,
        scale_fill=True,
        scaleup=True
    )(image=src)


def pad_img(src: np.ndarray, to_shape: tuple[int, int], value: int = 0) -> np.ndarray:
    """
    Center-pad an image to the desired shape with a constant value

    :param src: An input image
    :param to_shape: New shape
    :param value: Padding value
    :return: Padded image
    """

    if src.ndim != 2:
        raise NotImplementedError

    if src.ndim != len(to_shape):
        raise ValueError

    y, x = src.shape
    y_new, x_new = to_shape

    if y_new < y or x_new < x:
        raise ValueError

    y_pad = y_new - y
    x_pad = x_new - x

    return np.pad(
        src,
        pad_width=(
            (y_pad // 2, y_pad // 2 + y_pad % 2),
            (x_pad // 2, x_pad // 2 + x_pad % 2)
        ),
        mode='constant',
        constant_values=value
    )


def is_duplicate(
        img1: np.ndarray,
        img2: np.ndarray,
        thresh: float = 27.0
) -> bool:
    """
    Detect if two images are approximately identical by calculating
    the average difference across HSV channels between the images.
    A simplified version of the PySceneDetect's Content-Aware Detector

    :param img1: An input image
    :param img2: An input image
    :param thresh:
    :return:
    """

    if img1.ndim != 3 or img2.ndim != 3:
        raise ValueError

    hsv1 = cv2.split(cv2.cvtColor(img1, cv2.COLOR_RGB2HSV))
    hsv2 = cv2.split(cv2.cvtColor(img2, cv2.COLOR_RGB2HSV))

    y1, x1 = img1.shape
    y2, x2 = img2.shape
    new_shape = (max(y1, y2), max(x1, x2))

    d_hsv = sum(
        mpd(
            img1=pad_img(mat1, new_shape),
            img2=pad_img(mat2, new_shape)
        )
        for mat1, mat2 in zip(hsv1, hsv2)
    )

    similarity = d_hsv / 3
    return similarity >= thresh


def mpd(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate mean pixel distance between two images

    :param img1: A single-channel image
    :param img2: A single-channel image of the same shape as `img1`
    :return: Mean pixel distance
    """

    if (
            img1.ndim != 2 or img2.ndim != 2
            or img1.shape != img2.shape
    ):
        raise ValueError

    diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))
    return np.sum(diff) / img1.size
