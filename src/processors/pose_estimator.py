from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator

import cv2
import kornia
import numpy as np
import torch
from kornia.feature import DeDoDe, LightGlue
from scipy.spatial.transform import Rotation
from torch import Tensor

from utils.logger import PerformanceMonitor, get_logger, timer
from utils.misc import infer_device

if TYPE_CHECKING:
    from camera.realsense import Intrinsics
    from core.scene import RigidObject, Snapshot
    from processors.segmenter import BBox


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


class ObjectPose:
    def __init__(
            self,
            x: float, y: float, z: float,
            rx: float, ry: float, rz: float,
            is_valid: bool = True
    ) -> None:
        self.x: float = x
        self.y: float = y
        self.z: float = z

        self.rx: float = rx
        self.ry: float = ry
        self.rz: float = rz

        self._is_valid: bool = is_valid

    def __repr__(self) -> str:
        if self.is_lost:
            return '<ObjectPose [LOST]>'

        return (
            f'<ObjectPose '
            f'pos:[{self.x:.3f} {self.y:.3f} {self.z:.3f}] m, '
            f'rot:[{self.rx:.3f} {self.ry:.3f} {self.rz:.3f}] rad>'
        )

    @classmethod
    def from_matrix(cls, pose_matrix: np.ndarray) -> ObjectPose:
        if pose_matrix.shape != (4, 4):
            raise ValueError

        tx, ty, tz = pose_matrix[:3, 3]

        r = Rotation.from_matrix(pose_matrix[:3, :3])
        rx, ry, rz = r.as_euler('xyz')

        return cls(
            x=tx, y=ty, z=tz,
            rx=rx, ry=ry, rz=rz,
            is_valid=True
        )

    @classmethod
    def lost(cls) -> ObjectPose:
        return cls(
            x=np.nan, y=np.nan, z=np.nan,
            rx=np.nan, ry=np.nan, rz=np.nan,
            is_valid=False
        )

    @property
    def pos(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    @property
    def rot(self) -> tuple[float, float, float]:
        return self.rx, self.ry, self.rz

    @property
    def pitch(self) -> float:
        return self.rx

    @property
    def yaw(self) -> float:
        return self.ry

    @property
    def roll(self) -> float:
        return self.rz

    @property
    def pose_6d(self) -> tuple[float, float, float, float, float, float]:
        return self.pos + self.rot

    @property
    def is_lost(self) -> bool:
        return not self._is_valid


class PoseEstimator:
    def __init__(
            self,
            camera_intrinsics: Intrinsics,
            detector_weights: str = 'L-C4-v2',
            descriptor_weights: str = 'G-upright',
            score_thresh: float = 0.5,
            match_thresh: int = 10,
            estimation_rng: tuple[float, float] = (0.1, 5.0)
    ) -> None:
        if match_thresh < 1:
            raise ValueError

        if score_thresh < 0.0 or score_thresh > 1.0:
            raise ValueError

        self.dist_min, self.dist_max = estimation_rng
        if self.dist_min < 0.0 or self.dist_max < 0.0 or self.dist_max <= self.dist_min:
            raise ValueError

        self.device: torch.device = infer_device()
        self.intrinsics: Intrinsics = camera_intrinsics

        torch.backends.cudnn.benchmark = True

        self.max_dim: int = 448
        self.detector_dtype: torch.dtype = (
            torch.float16
            if self.device.type == 'cuda'
            else torch.float32
        )
        self.detector = DeDoDe.from_pretrained(
            detector_weights=detector_weights,
            descriptor_weights=descriptor_weights,
            amp_dtype=self.detector_dtype
        ).to(self.device)

        self.matcher = LightGlue(features=f'dedode{descriptor_weights[0].lower()}').to(self.device)
        self.score_thresh: float = score_thresh
        self.match_thresh: int = match_thresh

        self.logger: logging.Logger = get_logger('PoseEstimator', level=logging.INFO)
        self.monitor: PerformanceMonitor = PerformanceMonitor()
        self.logger.info('PoseEstimator initialized')

    def compute_features(self, rgb_crop: np.ndarray, n_kpts: int = 500) -> KeypointFeatures:
        if rgb_crop.ndim != 3 or rgb_crop.shape[-1] != 3:
            raise ValueError

        with timer('PoseEstimator.compute_features.preprocess', self.logger, self.monitor):
            h, w = rgb_crop.shape[:2]

            scale = self.max_dim / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)

            rgb_crop = cv2.resize(rgb_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((self.max_dim, self.max_dim, 3), dtype=np.uint8)
            canvas[:new_h, :new_w] = rgb_crop

            img: Tensor = (
                    kornia
                    .image_to_tensor(canvas, keepdim=False)
                    .to(self.detector_dtype) / 255.0
            ).to(self.device)

        with (
            torch.inference_mode(),
            timer('PoseEstimator.compute_features.inference', self.logger, self.monitor)
        ):
            keypoints, scores, descriptors = self.detector(img, n=n_kpts)

            mask_x = keypoints[..., 0] < new_w
            mask_y = keypoints[..., 1] < new_h
            valid_mask = mask_x & mask_y
            valid_indices = valid_mask[0]

            keypoints = keypoints[:, valid_indices, :]
            descriptors = descriptors[:, valid_indices, :]

            keypoints[..., 0].clamp_(0, new_w - 1)
            keypoints[..., 1].clamp_(0, new_h - 1)

            if scale != 1.0:
                keypoints /= scale

            keypoints[..., 0].clamp_(0, w - 1)
            keypoints[..., 1].clamp_(0, h - 1)

        return KeypointFeatures(
            keypoints=keypoints,
            descriptors=descriptors,
            img_size=torch.tensor([[w, h]], device=self.device)
        )

    def check_similarity(
            self,
            new: KeypointFeatures,
            refs: list[KeypointFeatures],
            match_thresh: int = 15
    ) -> int:
        for ref in refs:
            with torch.inference_mode():
                matches_out = self.matcher({
                    'image0': dict(ref),
                    'image1': dict(new)
                })

                n_valid = (matches_out['scores'][0] > self.score_thresh).sum()
                if n_valid > match_thresh:
                    return True

        return False

    def estimate_pose(
            self,
            query: np.ndarray | KeypointFeatures,
            query_bbox: BBox,
            ref_obj: RigidObject,
            n_refs: int = -1
    ) -> ObjectPose | None:
        if n_refs == 0:
            return None

        if ref_obj.num_snapshots == 0:
            return None

        qx1, qy1, qx2, qy2 = query_bbox
        if query_bbox.w < 10 or query_bbox.h < 10:
            return None

        if isinstance(query, np.ndarray):
            if query.ndim != 3 or query.shape[-1] != 3:
                raise ValueError

            q_crop: np.ndarray = query[qy1:qy2, qx1:qx2]
            if q_crop.size == 0:
                return None

            q_feats: KeypointFeatures = self.compute_features(q_crop)
        else:
            q_feats = query

        if n_refs < 0:
            n_refs = None

        ref_feats = [
            snap.features
            for snap in ref_obj.snapshots[:n_refs]
        ]

        best_idx: int = -1
        best_score: int = 0
        best_matches: Tensor | None = None

        with torch.inference_mode():
            for idx, ref_feat in enumerate(ref_feats):
                matches_out = self.matcher({
                    'image0': dict(ref_feat),
                    'image1': dict(q_feats)
                })

                matches: Tensor = matches_out['matches'][0]              # (N, 2)
                scores: Tensor = matches_out['scores'][0]                # (N,)

                valid: Tensor = scores > self.score_thresh               # (N,)
                n_valid: int = valid.sum().item()

                if n_valid > best_score:
                    best_score = n_valid
                    best_idx = idx
                    best_matches = matches[valid]                       # (M, 2)

        if best_score < self.match_thresh:
            return None

        best_snap: Snapshot = ref_obj.snapshots[best_idx]
        matches: np.ndarray = best_matches.cpu().numpy()

        ref_kpts = best_snap.features.keypoints[0].cpu().numpy()
        ref_kpts = ref_kpts[matches[:, 0]]

        q_kpts = q_feats.keypoints[0].cpu().numpy()
        q_kpts = q_kpts[matches[:, 1]]

        valid_kpts = self._filter_kpts(ref_kpts, best_snap.mask)
        ref_kpts = ref_kpts[valid_kpts]
        q_kpts = q_kpts[valid_kpts]

        if len(ref_kpts) < self.match_thresh:
            return None

        q_kpts_global = q_kpts.copy()
        q_kpts_global[:, 0] += qx1
        q_kpts_global[:, 1] += qy1

        sx1, sy1, *_ = best_snap.bbox
        depth_map = best_snap.depth.astype(np.float32) / 1000.0
        sh, sw = depth_map.shape

        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.ppx, self.intrinsics.ppy

        obj_points = []
        img_points = []

        for idx in range(len(ref_kpts)):
            u, v = map(int, ref_kpts[idx])

            if not (0 <= u < sw and 0 <= v < sh):
                continue

            z = depth_map[v, u]

            if z <= self.dist_min or z >= self.dist_max:
                continue

            u_global = u + sx1
            v_global = v + sy1

            x = (u_global - cx) * z / fx
            y = (v_global - cy) * z / fy

            obj_points.append([x, y, z])
            img_points.append(q_kpts_global[idx])

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        if len(obj_points) < 4:
            return None

        centroid = np.mean(obj_points, axis=0)
        obj_points -= centroid

        success, rvec, tvec, _ = cv2.solvePnPRansac(
            objectPoints=obj_points,
            imagePoints=img_points,
            cameraMatrix=self.intrinsics.K,
            distCoeffs=self.intrinsics.coeffs,
            iterationsCount=100,
            reprojectionError=6.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        R, _ = cv2.Rodrigues(rvec)
        pose_6d = np.eye(4)
        pose_6d[:3, :3] = R
        pose_6d[:3, 3] = tvec.squeeze()

        return ObjectPose.from_matrix(pose_6d)

    @staticmethod
    def _filter_kpts(kpts: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        valid = []

        for idx in range(len(kpts)):
            x, y = map(int, kpts[idx])

            if 0 <= x < h and 0 <= y < w and mask[x, y] == 1:
                valid.append(idx)

        return np.array(valid, dtype=np.int32)
