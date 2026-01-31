from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch
from torch import Tensor

from utils.misc import infer_device

if TYPE_CHECKING:
    from structures import BBox


class Snapshot:
    def __init__(
            self,
            idx: int,
            rgb: np.ndarray,
            mask: np.ndarray,
            depth: np.ndarray,
            bbox: BBox,
            features: KeypointFeatures
    ) -> None:
        self._idx = idx
        self._rgb = rgb
        self._mask = mask
        self._depth = depth
        self._bbox = bbox
        self._features = features

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def rgb(self) -> np.ndarray:
        return self._rgb

    @property
    def depth(self) -> np.ndarray:
        return self._depth

    @property
    def bbox(self) -> BBox:
        return self._bbox

    @property
    def features(self) -> KeypointFeatures:
        return self._features


class KeypointFeatures:
    def __init__(self, keypoints: Tensor, descriptors: Tensor, img_size: Tensor) -> None:
        if keypoints.ndim != 3 or descriptors.ndim != 3 or img_size.ndim != 2:
            raise ValueError

        batch, n_kpts, kpt_dim = keypoints.shape

        if kpt_dim != 2 or img_size.shape[1] != 2:
            raise ValueError

        if descriptors.shape[0] != batch or img_size.shape[0] != batch:
            raise ValueError

        if descriptors.shape[1] != n_kpts:
            raise ValueError

        self._keypoints: Tensor = keypoints           # (B, N, 2)
        self._descriptors: Tensor = descriptors       # (B, N, D)
        self._img_size: Tensor = img_size             # (B, 2): [[W, H]]

    @property
    def keypoints(self) -> Tensor:
        return self._keypoints

    @property
    def descriptors(self) -> Tensor:
        return self._descriptors

    @property
    def img_size(self) -> Tensor:
        return self._img_size

    @property
    def n_keypoints(self) -> int:
        return self._keypoints.shape[1]

    @property
    def batch_size(self) -> int:
        return self._keypoints.shape[0]

    @property
    def descriptor_dim(self) -> int:
        return self._descriptors.shape[2]

    @property
    def _dict(self) -> dict[str, Tensor]:
        return {
            'keypoints': self._keypoints,
            'descriptors': self._descriptors,
            'image_size': self._img_size
        }

    def __iter__(self) -> Iterator[Tensor]:
        yield from [self._keypoints, self._descriptors, self._img_size]

    def __getitem__(self, key: str) -> Tensor:
        return self._dict[key]

    def keys(self) -> Iterator[str]:
        yield from self._dict.keys()

    def repeat(self, n_repeats: int) -> KeypointFeatures:
        return KeypointFeatures(
            keypoints=self._keypoints.repeat(n_repeats, 1, 1),
            descriptors=self._descriptors.repeat(n_repeats, 1, 1),
            img_size=self._img_size.repeat(n_repeats, 1)
        )

    @classmethod
    def merge(cls, features: list[KeypointFeatures]) -> KeypointFeatures | None:
        if not features:
            return None

        kpts, desc, sizes = zip(*[
            (f.keypoints[0], f.descriptors[0], f.img_size[0])
            for f in features
        ])

        kpts = torch.stack(kpts, dim=0)
        desc = torch.stack(desc, dim=0)
        sizes = torch.stack(sizes, dim=0)

        return KeypointFeatures(
            keypoints=kpts,
            descriptors=desc,
            img_size=sizes
        )

    @classmethod
    def empty(cls, desc_dim: int = 256) -> KeypointFeatures:
        device: torch.device = infer_device()

        return cls(
            keypoints=torch.empty((1, 0, 2), device=device),
            descriptors=torch.empty((1, 0, desc_dim), device=device),
            img_size=torch.tensor([[0, 0]], device=device)
        )
