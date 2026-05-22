from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import kornia
import numpy as np
import torch
from kornia.feature import DeDoDe, LightGlue
from torch import Tensor

from structures import KeypointFeatures, ObjectPose
from utils.logger import PerformanceMonitor, get_logger, timer
from utils.misc import infer_device

if TYPE_CHECKING:
    from core.scene import RigidObject
    from structures import BBox, Intrinsics, Snapshot


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

    @timer('PoseEstimator.check_similarity')
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
            query_depth: np.ndarray | None = None,
            query_mask: np.ndarray | None = None,
            n_refs: int = -1
    ) -> ObjectPose | None:
        """
        Estimate the 6-DoF pose of an object using preferably the 3D-to-3D rigid alignment
        or the 2D-to-3D PnP solver as a fallback

        :param query: RGB crop of an object to find the pose of,
                      or its extracted features
        :param query_bbox: 2D bounding box of the queried object in the global frame
        :param ref_obj: The target object instance containing the reference snapshots
        :param query_depth: Actual depth map crop corresponding to the `query` crop,
                            used for 3D-to-3D rigid alignment
        :param query_mask: Binary segmentation mask corresponding to the `query` crop,
                           used to reject background matches in the live frame
        :param n_refs: Maximum number of recent snapshots to evaluate for matching.
                       If `-1`, evaluates all available snapshots

        :return: Estimated 6-DoF ObjectPose, or None if estimation fails
        """

        if n_refs == 0:
            return None

        if ref_obj.num_snapshots == 0:
            return None

        qx1, qy1, qx2, qy2 = query_bbox.xyxy
        if query_bbox.w < 10 or query_bbox.h < 10:
            return None

        # Extract features from the query if raw
        if isinstance(query, np.ndarray):
            if query.ndim != 3 or query.shape[-1] != 3:
                raise ValueError

            if query.size == 0:
                return None

            q_feats: KeypointFeatures = self.compute_features(query)
        else:
            q_feats = query

        if n_refs < 0:
            n_refs = None

        ref_feats = [
            snap.features
            for snap in ref_obj.snapshots[:n_refs]
        ]

        # Feature matching
        best_idx: int = -1
        best_score: int = 0
        best_matches: Tensor | None = None

        with (
            torch.inference_mode(),
            timer('PoseEstimator.estimate_pose.matching', self.logger, self.monitor)
        ):
            # Find the best matching snapshot -> reference
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

        # Process the winning snapshot
        best_snap: Snapshot = ref_obj.snapshots[best_idx]
        matches: np.ndarray = best_matches.cpu().numpy()

        ref_kpts = best_snap.features.keypoints[0].cpu().numpy()
        ref_kpts = ref_kpts[matches[:, 0]]

        q_kpts = q_feats.keypoints[0].cpu().numpy()
        q_kpts = q_kpts[matches[:, 1]]

        valid_kpts = self._filter_kpts(ref_kpts, best_snap.mask)
        if query_mask is not None:
            valid_query_kpts = self._filter_kpts(q_kpts, query_mask)
            valid_kpts = np.intersect1d(valid_kpts, valid_query_kpts)

        ref_kpts = ref_kpts[valid_kpts]
        q_kpts = q_kpts[valid_kpts]

        if len(ref_kpts) < self.match_thresh:
            return None

        # Geometry recovery
        with timer('PoseEstimation.estimate_pose.geometry', self.logger, self.monitor):
            # Convert the keypoint coordinates local the `query` to global
            q_kpts_global = q_kpts.copy()
            q_kpts_global[:, 0] += qx1
            q_kpts_global[:, 1] += qy1

            # Left-top corner bbox coordinates
            sx1, sy1, *_ = best_snap.bbox

            # Convert the depth map to meters
            ref_depth_map: np.ndarray = best_snap.depth.astype(np.float32) / 1000.0
            sh, sw = ref_depth_map.shape

            # Focal length, optical center
            fx, fy = self.intrinsics.fx, self.intrinsics.fy
            cx, cy = self.intrinsics.ppx, self.intrinsics.ppy

            # Point cloud stores
            obj_points_3d = []
            img_points_2d = []
            live_points_3d = []

            for idx in range(len(ref_kpts)):
                # Reference lifting
                u_ref, v_ref = map(int, ref_kpts[idx])

                # If the keypoint is outside the reference depth crop
                if not (0 <= u_ref < sw and 0 <= v_ref < sh):
                    continue

                # Get the corresponding point's depth
                z_ref = ref_depth_map[v_ref, u_ref]

                # Check the depth is within the valid range
                if z_ref <= self.dist_min or z_ref >= self.dist_max:
                    continue

                # Back-project reference
                # Get the global coords of the ref keypoints
                u_global_ref = u_ref + sx1
                v_global_ref = v_ref + sy1

                # Perform pinhole back-projection
                x_ref = (u_global_ref - cx) * z_ref / fx
                y_ref = (v_global_ref - cy) * z_ref / fy

                obj_points_3d.append([x_ref, y_ref, z_ref])
                img_points_2d.append(q_kpts_global[idx])

                # Live lifting
                live_points_3d.append(None)

                if query_depth is None:
                    continue

                # Look up depth using local coords
                q_u_local, q_v_local = map(int, q_kpts[idx])
                dh, dw = query_depth.shape

                if not (0 <= q_u_local < dw and 0 <= q_v_local < dh):
                    continue

                z_live = query_depth[q_v_local, q_u_local] / 1000.0

                if z_live <= self.dist_min or z_live >= self.dist_max:
                    continue

                # Back-project using global coords
                q_u_global, q_v_global = map(int, q_kpts_global[idx])
                x_live = (q_u_global - cx) * z_live / fx
                y_live = (q_v_global - cy) * z_live / fy

                live_points_3d[-1] = [x_live, y_live, z_live]

            obj_points_3d = np.array(obj_points_3d, dtype=np.float32)
            img_points_2d = np.array(img_points_2d, dtype=np.float32)

            if len(obj_points_3d) < 4:
                return None

            # Center the reference points for PnP
            centroid = np.mean(obj_points_3d, axis=0)
            obj_points_centered = obj_points_3d - centroid

        # Pose solving
        pose_matrix = None

        # Try 3D-to-3D rigid alignment
        if query_depth is not None:
            with timer('PoseEstimator.estimate_pose.rigid3d', self.logger, self.monitor):
                # Filter points where both 3D clouds are valid
                valid_ref = []
                valid_live = []

                for r_pt, l_pt in zip(obj_points_3d, live_points_3d):
                    if l_pt is None:
                        continue

                    valid_ref.append(r_pt)
                    valid_live.append(l_pt)

                # Require at least 6 points for stable 3D alignment
                if len(valid_ref) >= 6:
                    valid_ref = np.array(valid_ref, dtype=np.float32)
                    valid_live = np.array(valid_live, dtype=np.float32)

                    c_ref = np.mean(valid_ref, axis=0)
                    c_live = np.mean(valid_live, axis=0)

                    # Estimate affine transform using RANSAC
                    result = cv2.estimateAffine3D(
                        src=valid_ref - c_ref,
                        dst=valid_live - c_live
                    )

                    success = result[0]
                    M = result[1] if success else None

                    if success:
                        M: np.ndarray
                        R_affine = M[:3, :3]

                        # Enforce rigid rotation (remove scale/shear via SVD)
                        U, _, Vt = np.linalg.svd(R_affine)
                        R_rigid = U @ Vt

                        # Ensure it's a valid rotation, not a reflection
                        if np.linalg.det(R_rigid) < 0:
                            Vt[-1, :] *= -1
                            R_rigid = U @ Vt

                        pose_matrix = np.eye(4)
                        pose_matrix[:3, :3] = R_rigid
                        pose_matrix[:3, 3] = c_live

        # If RA unsuccessful, fall back to 2D-to-3D PnP
        if pose_matrix is None:
            with timer('PoseEstimator.estimate_pose.pnp', self.logger, self.monitor):
                success, rvec, tvec, _ = cv2.solvePnPRansac(
                    objectPoints=obj_points_centered,
                    imagePoints=img_points_2d,
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
                pose_matrix = np.eye(4)
                pose_matrix[:3, :3] = R

                pose_matrix[:3, 3] = tvec.squeeze()

        return ObjectPose.from_matrix(pose_matrix)

    @staticmethod
    def _filter_kpts(kpts: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        valid = []

        for idx in range(len(kpts)):
            x, y = map(int, kpts[idx])

            if 0 <= y < h and 0 <= x < w and mask[y, x] > 0:
                valid.append(idx)

        return np.array(valid, dtype=np.int32)
