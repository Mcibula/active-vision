"""
Image manipulation utilities
"""

from typing import Literal

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvf
from torch import Tensor
from torchvision.transforms import InterpolationMode


def crop_zeros(src: Tensor) -> Tensor | None:
    """
    Crop zeros around the center

    :param src: An input image of shape (H, W) or (H, W, C)
    :return: Cropped image or `None` if `src` is zeros only
    """

    if src.ndim < 2:
        raise ValueError

    src_h, src_w, *_ = src.shape
    nonzero = torch.nonzero(src, as_tuple=False)

    if nonzero.numel() == 0:
        return None

    y_nonzero = nonzero[:, 0]
    x_nonzero = nonzero[:, 1]

    y_up = y_nonzero.min()
    y_down = y_nonzero.max() + 1
    x_left = x_nonzero.min()
    x_right = x_nonzero.max() + 1

    return src[
        y_up:y_down,
        x_left:x_right
    ]


def resize_imgs(
        src: np.ndarray | Tensor,
        to_shape: tuple[int, int],
        dim_order: Literal['nchw', 'nhwc'] = 'nhwc',
        stretch: bool = False,
        padding_value: int = 0
) -> np.ndarray | Tensor:
    """
    Resize an image or a batch of images with optional padding.
    Input from `src` accepts:
        * a single-channel image of shape (H, W)
        * a C-channel image of shape (H, W, C) [`dim_order='nhwc'`] or (C, H, W) [`dim_order='nchw'`]
        * a batch of N C-channel equally-sized images of shape (N, H, W, C) [`dim_order='nhwc'`]
          or (N, C, H, W) [`dim_order='nchw'`]

    :param src: An image or a batch of images as a NumPy array or a PyTorch Tensor
    :param to_shape: New image shape as (H, W)
    :param dim_order: Order of dimensions in `src`; either `'nchw'` or `'nhwc'`
    :param stretch: If `True`, the image will be stretched without padding;
                    if `False`, the padding will be applied to maintain the original aspect ratio
    :param padding_value: An integer constant to pad with
    :return: Resized image or batch of images in the original format
    """

    if dim_order not in ('nchw', 'nhwc'):
        raise ValueError

    if isinstance(src, np.ndarray):
        src: Tensor = torch.from_numpy(src)
        orig_struct: str = 'numpy'
    else:
        orig_struct: str = 'torch'

    if src.ndim < 2 or src.ndim > 4:
        raise ValueError

    # Convert to NCHW
    orig_ndim = src.ndim
    src = src[*((None,) * (4 - src.ndim)), ...]
    if dim_order == 'nhwc' and orig_ndim != 2:
        src = src.permute(0, 3, 1, 2)

    src_n, src_c, src_h, src_w = src.shape
    new_h, new_w = to_shape

    # Scale ratio (new / old)
    r = min(new_h / src_h, new_w / src_w)

    if stretch:
        # Stretching without padding
        dh, dw = 0.0, 0.0
        new_unpad: tuple[int, int] = (new_h, new_w)
    else:
        # Padding
        new_unpad: tuple[int, int] = round(src_h * r), round(src_w * r)
        dh, dw = new_h - new_unpad[0], new_w - new_unpad[1]

    # Center
    dw /= 2
    dh /= 2

    # Resize
    if (src_h, src_w) != new_unpad:
        src = tvf.resize(
            src,
            size=new_unpad,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        )

    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    src = tvf.pad(
        src,
        padding=(left, top, right, bottom),
        fill=padding_value,
        padding_mode='constant'
    )

    # Convert back from NCHW to NHWC
    if dim_order == 'nhwc':
        src = src.permute(0, 2, 3, 1)

    if orig_ndim == 2:
        src = src.squeeze()
    elif orig_ndim < src.ndim:
        src = src.squeeze(tuple(range(src.ndim - orig_ndim)))

    if orig_struct == 'numpy':
        return src.numpy()

    return src


def pad_img(src: np.ndarray, to_shape: tuple[int, int], value: int = 0) -> np.ndarray:
    """
    Center-pad an image to the desired shape with a constant value

    :param src: A single- or multi-channel input image of the CHW dimension order
    :param to_shape: New shape (H, W) to pad every channel matrix to
    :param value: Padding value
    :return: Padded image
    """

    if len(to_shape) != 2:
        raise ValueError

    if src.ndim not in (2, 3):
        raise NotImplementedError

    *_, h, w = src.shape
    h_new, w_new = to_shape

    if h_new < h or w_new < w:
        raise ValueError

    h_pad = h_new - h
    w_pad = w_new - w

    pad_width = [
        (h_pad // 2, h_pad // 2 + h_pad % 2),
        (w_pad // 2, w_pad // 2 + w_pad % 2)
    ]
    if src.ndim == 3:
        pad_width = [(0, 0), *pad_width]

    return np.pad(
        src,
        pad_width=pad_width,
        mode='constant',
        constant_values=value
    )


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate mean squared error between two images.
    Both input images must have the CHW dimension order and the same number of channels.
    If the images do not match in size, they will be padded with zeros to larger size
    along each dimension

    :param img1: A single- or multi-channel image with the CHW dimension order
    :param img2: A single- or multi-channel image with the CHW dimension order
                 and the same number of channels as `img1`
    :return: Mean squared error
    """

    if img1.ndim != img2.ndim or img1.ndim not in (2, 3):
        raise ValueError

    *c1, h1, w1 = img1.shape
    *c2, h2, w2 = img2.shape

    if c1 != c2:
        raise ValueError

    if (h1, w1) != (h2, w2):
        new_shape = (max(h1, h2), max(w1, w2))
        img1 = pad_img(img1, new_shape)
        img2 = pad_img(img2, new_shape)

    return np.mean(
        (img1.astype(np.float32) - img2.astype(np.float32)) ** 2,
        dtype=np.float32
    )
