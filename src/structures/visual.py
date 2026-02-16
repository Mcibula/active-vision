"""
Data structures operating with visual data
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch
from torch import Tensor

from utils.misc import infer_device

if TYPE_CHECKING:
    import numpy as np

    from structures import BBox


class Snapshot:
    """
    Object grouping collected visual and geometrical data
    of a segmented rigid object
    """

    def __init__(
            self,
            idx: int,
            rgb: np.ndarray,
            mask: np.ndarray,
            depth: np.ndarray,
            bbox: BBox,
            features: KeypointFeatures
    ) -> None:
        r"""
        Create a new snapshot

        :param idx: Snapshot index with respect to the assigned rigid object
        :param rgb: Segmented RGB image (:math:`H \times W \times 3`)
        :param mask: Binary segmentation mask of the same size as `rgb`
        :param depth: Single-channel depth map of the same size as `rgb`
        :param bbox: Bounding box instance localizing the object in the global scene image
        :param features: Keypoint features computed from `rgb`
        """

        self._idx = idx
        self._rgb = rgb
        self._mask = mask
        self._depth = depth
        self._bbox = bbox
        self._features = features

    @property
    def idx(self) -> int:
        """
        Index of this snapshot with respect to the assigned rigid object
        """

        return self._idx

    @property
    def mask(self) -> np.ndarray:
        """
        Binary segmentation mask matching `Snapshot.rgb`
        """

        return self._mask

    @property
    def rgb(self) -> np.ndarray:
        """
        Segmented RGB image of the object matching `Snapshot.mask`
        """

        return self._rgb

    @property
    def depth(self) -> np.ndarray:
        """
        Single-channel depth map of the object matching `Snapshot.mask`
        """

        return self._depth

    @property
    def bbox(self) -> BBox:
        """
        Bounding box instance localizing the object in the global scene image
        """

        return self._bbox

    @property
    def features(self) -> KeypointFeatures:
        """
        Keypoint features computed from `Snapshot.rgb``
        """

        return self._features


class KeypointFeatures:
    """
    Object preserving local features of a batch of images,
    including keypoints and their descriptors
    """

    def __init__(self, keypoints: Tensor, descriptors: Tensor, img_size: Tensor) -> None:
        r"""
        Register keypoints and descriptors

        :param keypoints: Batch tensor of keypoints (:math:`B \times N \times 2`),
                          where `keypoints[b, n] = [w_coords, h_coords]`
        :param descriptors: Batch tensor of feature descriptors corresponding
                            to `keypoints` (:math:`B \times N \times D`)
        :param img_size: Batch tensor of image sizes (:math:`B \times 2`),
                         where `img_size[b] = [w, h]`
        """

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
        r"""
        Tensor of keypoints (:math:`B \times N \times 2`),
        where `KeypointFeatures.keypoints[b, n] = [w_coords, h_coords]`
        """

        return self._keypoints

    @property
    def descriptors(self) -> Tensor:
        r"""
        Feature descriptors (:math:`B \times N \times D`) corresponding
        to `KeypointFeatures.keypoints`
        """

        return self._descriptors

    @property
    def img_size(self) -> Tensor:
        r"""
        Tensor of image sizes (:math:`B \times 2`),
        where `img_size[b] = [w, h]`
        """

        return self._img_size

    @property
    def n_keypoints(self) -> int:
        """
        Number of captured keypoints
        """

        return self._keypoints.shape[1]

    @property
    def batch_size(self) -> int:
        """
        Number of images processed in this batch
        """

        return self._keypoints.shape[0]

    @property
    def descriptor_dim(self) -> int:
        """
        Dimensionality of the feature descriptors
        """

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
        """
        Return the keys of this class' dictionary representation `KeypointFeatures._dict`
        """

        yield from self._dict.keys()

    def repeat(self, n_repeats: int) -> KeypointFeatures:
        """
        Construct a larger batch by repeating this instance

        :param n_repeats: Number of copies to make
        :return: A new `KeypointFeatures` instance with repeated data
        """

        return KeypointFeatures(
            keypoints=self._keypoints.repeat(n_repeats, 1, 1),
            descriptors=self._descriptors.repeat(n_repeats, 1, 1),
            img_size=self._img_size.repeat(n_repeats, 1)
        )

    @classmethod
    def merge(cls, features: list[KeypointFeatures]) -> KeypointFeatures | None:
        """
        Combine data from multiple `KeypointFeatures` instances
        by stacking them along the axis 0

        :param features: List of `KeypointFeatures` instances to merge
        :return: A new `KeypointFeatures` instance containing merged data
        """

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
        """
        Initialize an empty `KeypointFeatures` instance

        :param desc_dim: Dimensionality of the feature descriptors
        :return: An empty `KeypointFeatures` instance
        """

        device: torch.device = infer_device()

        return cls(
            keypoints=torch.empty((1, 0, 2), device=device),
            descriptors=torch.empty((1, 0, desc_dim), device=device),
            img_size=torch.tensor([[0, 0]], device=device)
        )
