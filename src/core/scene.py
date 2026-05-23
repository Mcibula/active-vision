from __future__ import annotations

import logging
import queue
import random
import threading
import time
from queue import Queue
from typing import TYPE_CHECKING, Any, Iterator

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from structures import BBox, KeypointFeatures, ObjectPose, Snapshot, Trajectory
from utils.image import mse, sharpness
from utils.logger import PerformanceMonitor, get_logger, timer
from utils.misc import infer_device

if TYPE_CHECKING:
    from processors.pose_estimator import PoseEstimator
    from processors.segmenter import Segmenter
    from structures import TrackRecord


class RigidObject:
    def __init__(
            self,
            obj_id: int,
            max_snapshots: int = 100,
            dupl_history: int = 5,
            snap_subset: int = 5,
            mse_thresh: float = 600.0,
            kpt_thresh: int = 30,
            eff_thresh: int = 600,
            pose_interval: float = 0.2
    ) -> None:
        self.obj_id: int = obj_id
        self._lock = threading.Lock()

        self._snapshots: list[Snapshot] = []
        self._trajectory: Trajectory = Trajectory(capacity=None)
        self._pose_diagnostics: list[dict[str, Any]] = []

        self.last_seen: BBox = BBox.null()
        self.last_updated: float = 0.0
        self.last_pose_time: float = 0.0

        self.is_busy: bool = False
        self._staged_rgb: np.ndarray | None = None
        self._staged_mask: np.ndarray | None = None
        self._staged_depth: np.ndarray | None = None
        self._staged_bbox: BBox | None = None

        self.max_snapshots: int = max_snapshots
        self.dupl_history: int = dupl_history
        self.snap_subset: int = snap_subset
        self.mse_thresh: float = mse_thresh
        self.kpt_thresh: int = kpt_thresh
        self.eff_thresh: int = eff_thresh
        self.pose_interval: float = pose_interval

    def __repr__(self) -> str:
        return (
            f'<RigidObject #{self.obj_id} '
            f'with {self.num_snapshots} snapshot{"s" if self.num_snapshots > 1 else ''}>'
        )

    def __getitem__(self, snapshot_id: int) -> Snapshot:
        return self._snapshots[snapshot_id]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        transient = [
            '_lock', 'is_busy',
            '_staged_rgb', '_staged_mask',
            '_staged_depth', '_staged_bbox'
        ]

        for key in transient:
            state.pop(key, None)

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

        self._lock = threading.Lock()
        self.is_busy = False
        self._staged_rgb = None
        self._staged_mask = None
        self._staged_depth = None
        self._staged_bbox = None
        self._pose_diagnostics = getattr(self, '_pose_diagnostics', [])

    @property
    def num_snapshots(self) -> int:
        return len(self._snapshots)

    @property
    def snapshots(self) -> list[Snapshot]:
        return self._snapshots

    @property
    def num_poses(self) -> int:
        return len(self._trajectory)

    @property
    def pose(self) -> ObjectPose | None:
        with self._lock:
            return self._trajectory[-1] if self.num_poses > 0 else None

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def pose_diagnostics(self) -> list[dict[str, Any]]:
        return self._pose_diagnostics

    @property
    def last_pose_diagnostics(self) -> dict[str, Any] | None:
        return self._pose_diagnostics[-1] if self._pose_diagnostics else None

    def add_pose(self, obj_pose: ObjectPose) -> None:
        with self._lock:
            self._trajectory.append(obj_pose)

        self.last_updated = time.time()

    def add_pose_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        diagnostics = diagnostics.copy()
        diagnostics['obj_id'] = self.obj_id
        diagnostics['timestamp'] = time.time()

        with self._lock:
            self._pose_diagnostics.append(diagnostics)

    def register_observations(
            self,
            snapshots: list[np.ndarray],
            masks: list[np.ndarray],
            depth_maps: list[np.ndarray],
            bboxes: list[BBox],
            feat_extractor: PoseEstimator
    ) -> None:
        if (
                not snapshots
                or len({len(snapshots), len(masks), len(depth_maps), len(bboxes)}) != 1
        ):
            raise ValueError

        best_idx = self._best_snapshot(snapshots, masks)
        if best_idx == -1:
            return

        now = time.time()
        self.last_updated = now

        rgb: np.ndarray = snapshots[best_idx]
        mask: np.ndarray = masks[best_idx]
        depth_map: np.ndarray = depth_maps[best_idx]
        bbox: BBox = bboxes[best_idx]

        # RGB, mask, and depth crops must share the bbox-local half-open slice frame
        assert rgb.shape[:2] == mask.shape == depth_map.shape == bbox.shape

        with self._lock:
            self._staged_rgb = rgb
            self._staged_mask = mask
            self._staged_depth = depth_map
            self._staged_bbox = bbox

        if self.num_snapshots > 0:
            self._propagate_pose(bbox, feat_extractor)
            return

        if self.last_seen.is_null:
            self.last_seen = bbox

    def update_async(self, feat_extractor: PoseEstimator) -> None:
        with self._lock:
            rgb: np.ndarray = self._staged_rgb
            mask: np.ndarray = self._staged_mask
            depth_map: np.ndarray = self._staged_depth
            bbox: BBox = self._staged_bbox

        feats: KeypointFeatures = feat_extractor.compute_features(rgb)

        pose_found = False
        if self.num_snapshots > 0:
            pose: ObjectPose | None = feat_extractor.estimate_pose(
                query=feats,
                query_bbox=bbox,
                ref_obj=self,
                query_depth=depth_map,
                query_mask=mask
            )
            self.add_pose_diagnostics(feat_extractor.last_diagnostics)

            if pose is not None:
                self.add_pose(pose)
                self.last_seen = bbox
                pose_found = True
            else:
                self.add_pose(ObjectPose.lost())

            self.last_pose_time = time.time()

        if self.num_snapshots >= self.max_snapshots:
            return

        if self.num_snapshots == 0:
            self.last_seen = bbox

        if self.num_snapshots > 0:
            if not pose_found:
                return

            if self._pixel_duplicate(rgb):
                return

            feat_sim = feat_extractor.check_similarity(
                new=feats,
                refs=[s.features for s in self._snapshots],
                match_thresh=self.kpt_thresh
            )

            if feat_sim:
                return

        with self._lock:
            self._snapshots.append(
                Snapshot(
                    idx=self.num_snapshots,
                    rgb=rgb,
                    mask=mask,
                    depth=depth_map,
                    bbox=bbox,
                    features=feats
                )
            )

    def _pixel_duplicate(self, rgb: np.ndarray) -> bool:
        chw_snap = rgb.transpose(2, 0, 1)

        for hist_snapshot in self._snapshots[-self.dupl_history:]:
            if mse(
                    chw_snap,
                    hist_snapshot.rgb.transpose(2, 0, 1)
            ) < self.mse_thresh:
                return True

        return False

    def _best_snapshot(self, rgbs: list[np.ndarray], masks: list[np.ndarray]) -> int:
        if len(rgbs) != len(masks) or len(rgbs) == 0:
            raise ValueError

        sub_rng = range(max(0, len(rgbs) - self.snap_subset), len(rgbs))
        best_score = 0
        best_idx = -1

        for idx in sub_rng:
            rgb: np.ndarray = rgbs[idx]
            mask: np.ndarray = masks[idx]

            n_eff = np.count_nonzero(mask)
            if n_eff < self.eff_thresh:
                continue

            lvar = sharpness(rgb)
            recency = idx * 10
            score = lvar + recency

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _propagate_pose(self, new_bbox: BBox, feat_extractor: PoseEstimator) -> None:
        if self.num_poses == 0 or self.last_seen.is_null:
            self.last_seen = new_bbox
            return

        with self._lock:
            last_pose = self._trajectory[-1]

        if last_pose.is_lost:
            self.last_seen = new_bbox
            return

        old_cx, old_cy = self.last_seen.centroid
        new_cx, new_cy = new_bbox.centroid

        dx_px = new_cx - old_cx
        dy_px = new_cy - old_cy

        cur_z = last_pose.z
        if cur_z <= 0.1:
            self.last_seen = new_bbox
            return

        fx = feat_extractor.intrinsics.fx
        fy = feat_extractor.intrinsics.fy

        dx_m = (dx_px * cur_z) / fx
        dy_m = (dy_px * cur_z) / fy

        new_x = last_pose.x + dx_m
        new_y = last_pose.y + dy_m

        self.add_pose(
            ObjectPose(
                new_x, new_y, cur_z,
                *last_pose.rot,
                is_valid=True
            )
        )
        self.last_seen = new_bbox

    def show_snapshots(self, n: int = 1, ids: list[int] | None = None) -> None:
        if ids is not None:
            if not ids or any(i >= self.num_snapshots for i in ids):
                raise ValueError

            n = len(ids)
            sel_ids = ids
        else:
            if n <= 0:
                raise ValueError

            n = min(n, self.num_snapshots)
            sel_ids = random.sample(range(self.num_snapshots), n)

        n_cols = int(np.ceil(np.sqrt(n)))
        n_rows = int(np.ceil(n / n_cols))

        fig, axs = plt.subplots(
            nrows=n_rows, ncols=n_cols,
            figsize=(15, 20 * n_rows / n_cols),
            squeeze=False
        )

        for i in range(n_rows):
            for j in range(n_cols):
                ax = axs[i, j]
                ax.axis('off')

                idx = i * n_cols + j
                if idx >= n:
                    continue

                snap_id = sel_ids[idx]

                ax.imshow(self.snapshots[snap_id].rgb)
                ax.set_title(f'#{snap_id}')

        fig.tight_layout()
        plt.show()


class Scene:
    def __init__(self, segmenter: Segmenter, pose_estimator: PoseEstimator) -> None:
        self.device: torch.device = infer_device()
        self.segmenter: Segmenter | None = segmenter
        self.pose_estimator: PoseEstimator | None = pose_estimator

        self.frame_count: int = 0
        self.objects: dict[int, RigidObject] = {}

        self._pose_queue: Queue[RigidObject] = Queue(maxsize=10)

        self.temp_thresh: float = 2.0
        self.iou_thresh: float = 0.5
        self.iom_thresh: float = 0.9

        self.logger: logging.Logger = get_logger('Scene', level=logging.INFO)
        self.monitor: PerformanceMonitor = PerformanceMonitor()
        self.logger.info('Scene initialized')

    def __repr__(self) -> str:
        return f'<Scene with {self.num_objects} objects>'

    def __iter__(self) -> Iterator[RigidObject]:
        yield from self.objects.values()

    def __getitem__(self, obj_id: int | tuple[int, int]) -> RigidObject | Snapshot:
        if isinstance(obj_id, int):
            return self.objects[obj_id]

        if isinstance(obj_id, tuple):
            return self.objects[obj_id[0]][obj_id[1]]

        raise TypeError

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    def read_frames(self, frames: tuple[np.ndarray, np.ndarray] | list[tuple[np.ndarray, np.ndarray]]) -> None:
        if self.segmenter is None:
            raise RuntimeError

        if self.pose_estimator is None:
            raise RuntimeError

        if not isinstance(frames, list):
            frames = [frames]

        self.frame_count += len(frames)
        rgb_frames, d_frames = zip(*frames)

        record: dict[int, TrackRecord] = self.segmenter.track(list(rgb_frames))

        now = time.time()
        for obj_id, obj_record in record.items():
            if not obj_record.snapshots:
                continue

            cur_bbox: BBox = obj_record.xyxy[-1]

            with timer('Scene.read_frames.obj_id', self.logger, self.monitor):
                if obj_id not in self.objects:
                    best_id = -1
                    best_iou = self.iou_thresh
                    best_iom = self.iom_thresh

                    for ex_id, ex_obj in self.objects.items():
                        if (now - ex_obj.last_updated) < self.temp_thresh and ex_id not in record:
                            iou = cur_bbox.iou(ex_obj.last_seen)
                            iom = cur_bbox.iom(ex_obj.last_seen)

                            if iou > best_iou or iom > best_iom:
                                best_id = ex_id
                                best_iou = iou
                                best_iom = iom

                    if best_id != -1:
                        obj_id = best_id

            if obj_id not in self.objects:
                self.objects[obj_id] = RigidObject(obj_id)

            with timer('Scene.read_frames.register', self.logger, self.monitor):
                obj: RigidObject = self.objects[obj_id]
                obj.register_observations(
                    snapshots=obj_record.snapshots,
                    masks=obj_record.masks,
                    depth_maps=[
                        d_frames[obj_record.frame_ids[xyxy_id]][y1:y2, x1:x2]
                        for xyxy_id, (x1, y1, x2, y2) in enumerate(obj_record.xyxy)
                    ],
                    bboxes=obj_record.xyxy,
                    feat_extractor=self.pose_estimator
                )

        self._schedule_pose(now)

    def _schedule_pose(self, now: float) -> None:
        for obj in self.objects.values():
            if obj.is_busy:
                continue

            if obj.num_snapshots > 0 and (now - obj.last_pose_time) <= obj.pose_interval:
                continue

            try:
                obj.is_busy = True
                self._pose_queue.put_nowait(obj)
            except queue.Full:
                obj.is_busy = False

    def process_pose(self) -> None:
        try:
            obj = self._pose_queue.get(timeout=0.1)
        except queue.Empty:
            return

        try:
            with timer('Scene.process_pose', self.logger, self.monitor):
                obj.update_async(self.pose_estimator)
        except Exception:
            pass
        finally:
            obj.is_busy = False

    def save(self, path: str, store_models: bool = False) -> None:
        state: dict[str, Any] = self.__dict__.copy()

        if 'device' in state:
            state['device'] = str(self.device)

        if not store_models:
            state['segmenter'] = None
            state['pose_estimator'] = None

        transient = ['_pose_queue', 'logger', 'monitor']
        for key in transient:
            state.pop(key, None)

        joblib.dump(
            value=state,
            filename=path,
            compress=True
        )

    @classmethod
    def load(cls, path: str) -> Scene:
        state: dict[str, Any] = joblib.load(path)
        scene: Scene = cls.__new__(cls)

        scene.__dict__.update(state)
        scene.device = (
            torch.device(state['device'])
            if 'device' in state else infer_device()
        )
        scene._pose_queue = Queue(maxsize=10)
        scene.logger = get_logger('Scene', level=logging.INFO)
        scene.monitor = PerformanceMonitor()

        return scene
