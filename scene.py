import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from processors.segmenter import Segmenter, TrackRecord
from utils.image import mse


class RigidObject:
    def __init__(self, obj_id: int) -> None:
        self.obj_id = obj_id

        self.x: float | None = None
        self.y: float | None = None
        self.z: float | None = None
        self.rx: float | None = None
        self.ry: float | None = None
        self.rz: float | None = None

        self._masks: list[np.ndarray] = []
        self._snapshots: list[np.ndarray] = []

        self.duplicate_thresh: float = 600.0

    def __repr__(self) -> str:
        return (
            f'<RigidObject #{self.obj_id} '
            f'with {self.num_snapshots} snapshot{"s" if self.num_snapshots > 1 else ''}>'
        )

    def __getitem__(self, snapshot_id: int) -> tuple[np.ndarray, np.ndarray]:
        return self._snapshots[snapshot_id], self._masks[snapshot_id]

    @property
    def pose_6d(self) -> tuple[float, float, float, float, float, float]:
        return self.x, self.y, self.z, self.rx, self.ry, self.rz

    @property
    def num_snapshots(self) -> int:
        return len(self._snapshots)

    @property
    def masks(self) -> list[np.ndarray]:
        return self._masks

    @property
    def snapshots(self) -> list[np.ndarray]:
        return self._snapshots

    def set_pose_6d(
            self,
            x: float, y: float, z: float,
            rx: float, ry: float, rz: float
    ) -> None:
        (
            self.x, self.y, self.z,
            self.rx, self.ry, self.rz
        ) = (
            x, y, z,
            rx, ry, rz
        )

    def register_visuals(self, snapshots: list[np.ndarray], masks: list[np.ndarray]) -> None:
        if len(snapshots) != len(masks) or not snapshots:
            raise ValueError

        if not self._snapshots:
            self._snapshots.append(snapshots[0])
            self._masks.append(masks[0])

        for snapshot, mask in zip(snapshots[1:], masks[1:]):
            # Skip possible duplicate
            if mse(
                    snapshot.transpose(2, 0, 1),
                    self._snapshots[-1].transpose(2, 0, 1)
            ) < self.duplicate_thresh:
                continue

            self._snapshots.append(snapshot)
            self._masks.append(mask)

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

                ax.imshow(self.snapshots[snap_id])
                ax.set_title(f'#{snap_id}')

        fig.tight_layout()
        plt.show()


class Scene:
    def __init__(self, segmenter: Segmenter) -> None:
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.segmenter: Segmenter | None = segmenter

        self.frame_count: int = 0
        self.objects: dict[int, RigidObject] = {}

    def __repr__(self) -> str:
        return f'<Scene with {self.num_objects} objects>'

    def __getitem__(self, obj_id: int | tuple[int, int]) -> RigidObject | tuple[np.ndarray, np.ndarray]:
        if isinstance(obj_id, int):
            return self.objects[obj_id]

        if isinstance(obj_id, tuple):
            return self.objects[obj_id[0]][obj_id[1]]

        raise TypeError

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    def read_frames(self, frames: np.ndarray | list[np.ndarray]) -> None:
        if self.segmenter is None:
            raise RuntimeError

        if not isinstance(frames, list):
            frames = [frames]

        self.frame_count += len(frames)

        record: dict[int, TrackRecord] = self.segmenter.track(frames)

        for obj_id, obj_record in record.items():
            if not obj_record.snapshots:
                continue

            if obj_id not in self.objects:
                self.objects[obj_id] = RigidObject(obj_id)

            obj: RigidObject = self.objects[obj_id]
            obj.register_visuals(
                snapshots=obj_record.snapshots,
                masks=obj_record.masks
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
    def load(cls, path: str) -> 'Scene':
        store: dict[str, ...] = joblib.load(path)
        scene: Scene = cls.__new__(cls)

        scene.device = store['device']
        scene.segmenter = store['segmenter']
        scene.frame_count = store['frame_count']
        scene.objects = store['objects']

        return scene
