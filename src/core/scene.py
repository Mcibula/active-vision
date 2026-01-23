from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Iterator

import joblib
import matplotlib.pyplot as plt
import numpy as np

from processors.pose_estimator import ObjectPose
from utils.image import mse
from utils.misc import infer_device

if TYPE_CHECKING:
    from processors.pose_estimator import KeypointFeatures, PoseEstimator
    from processors.segmenter import Segmenter, TrackRecord


class Snapshot:
    def __init__(
            self,
            idx: int,
            rgb: np.ndarray,
            mask: np.ndarray,
            depth: np.ndarray,
            bbox: tuple[float, float, float, float],
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
    def bbox(self) -> tuple[float, float, float, float]:
        return self._bbox

    @property
    def features(self) -> KeypointFeatures:
        return self._features


class RigidObject:
    def __init__(self, obj_id: int) -> None:
        self.obj_id = obj_id

        self._trajectory: list[ObjectPose] = []
        self._snapshots: list[Snapshot] = []

        self.last_updated: float = 0.0

        self.max_snapshots: int = 100
        self.dupl_history = 5
        self.mse_thresh: float = 600.0
        self.kpt_thresh: int = 30
        self.eff_thresh: int = 600

    def __repr__(self) -> str:
        return (
            f'<RigidObject #{self.obj_id} '
            f'with {self.num_snapshots} snapshot{"s" if self.num_snapshots > 1 else ''}>'
        )

    def __getitem__(self, snapshot_id: int) -> Snapshot:
        return self._snapshots[snapshot_id]

    @property
    def num_poses(self) -> int:
        return len(self._trajectory)

    @property
    def pose(self) -> ObjectPose | None:
        if self.num_poses == 0:
            return None

        return self._trajectory[-1]

    @property
    def num_snapshots(self) -> int:
        return len(self._snapshots)

    @property
    def snapshots(self) -> list[Snapshot]:
        return self._snapshots

    def add_pose(self, obj_pose: ObjectPose) -> None:
        self._trajectory.append(obj_pose)
        self.last_updated = time.time()

    def register_observations(
            self,
            snapshots: list[np.ndarray],
            masks: list[np.ndarray],
            depth_maps: list[np.ndarray],
            bboxes: list[tuple[int, int, int, int]],
            feat_extractor: PoseEstimator
    ) -> None:
        if (
                not snapshots
                or len({len(snapshots), len(masks), len(depth_maps), len(bboxes)}) != 1
        ):
            raise ValueError

        for idx in range(len(snapshots)):
            rgb: np.ndarray = snapshots[idx]
            mask: np.ndarray = masks[idx]
            depth_map: np.ndarray = depth_maps[idx]
            bbox: tuple[int, int, int, int] = bboxes[idx]

            n_eff = np.count_nonzero(mask)
            if n_eff < self.eff_thresh:
                continue

            feats: KeypointFeatures = feat_extractor.compute_features(rgb)

            if self.num_snapshots > 0:
                pose: ObjectPose = feat_extractor.estimate_pose(
                    query=feats,
                    query_bbox=bbox,
                    ref_obj=self
                )

                self.add_pose(
                    pose
                    if pose is not None
                    else ObjectPose.lost()
                )

            if self.num_snapshots >= self.max_snapshots:
                continue

            if self.num_snapshots > 0:
                if self._pixel_duplicate(rgb):
                    continue

                feat_sim = feat_extractor.check_similarity(
                    new=feats,
                    refs=[s.features for s in self._snapshots],
                    batch_size=4,
                    match_thresh=self.kpt_thresh
                )

                if feat_sim:
                    continue

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
        self.device: str = infer_device()
        self.segmenter: Segmenter | None = segmenter
        self.pose_estimator: PoseEstimator | None = pose_estimator

        self.frame_count: int = 0
        self.objects: dict[int, RigidObject] = {}

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

        for obj_id, obj_record in record.items():
            if not obj_record.snapshots:
                continue

            if obj_id not in self.objects:
                self.objects[obj_id] = RigidObject(obj_id)

            obj: RigidObject = self.objects[obj_id]
            obj.register_observations(
                snapshots=obj_record.snapshots,
                masks=obj_record.masks,
                depth_maps=[
                    d_frames[obj_record.frame_ids[xyxy_id]][y1:y2 + 1, x1:x2 + 1]
                    for xyxy_id, (x1, y1, x2, y2) in enumerate(obj_record.xyxy)
                ],
                bboxes=obj_record.xyxy,
                feat_extractor=self.pose_estimator
            )

    def save(self, path: str, store_models: bool = False) -> None:
        store: dict[str, ...] = {
            'device': self.device,
            'segmenter': self.segmenter if store_models else None,
            'frame_count': self.frame_count,
            'objects': self.objects
        }

        joblib.dump(
            value=store,
            filename=path,
            compress=True
        )

    @classmethod
    def load(cls, path: str) -> Scene:
        store: dict[str, ...] = joblib.load(path)
        scene: Scene = cls.__new__(cls)

        scene.device = store['device']
        scene.segmenter = store['segmenter']
        scene.frame_count = store['frame_count']
        scene.objects = store['objects']

        return scene
