from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class TrackRecord:
    xyxy: list[BBox]
    masks: list[np.ndarray]
    snapshots: list[np.ndarray]
    frame_ids: list[int]


class BBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            raise ValueError

        if x1 > x2 or y1 > y2:
            raise ValueError

        self._x1: int = int(x1)
        self._y1: int = int(y1)
        self._x2: int = int(x2)
        self._y2: int = int(y2)

    def __repr__(self) -> str:
        return f'<BBox {self.w}x{self.h} @ {self.lu}, {self.br}>'

    def __iter__(self) -> Iterator[int]:
        yield from self.xyxy

    def __eq__(self, other: BBox) -> bool:
        return (
            self.x1 == other.x1 and self.y1 == other.y1
            and self.x2 == self.x2 and self.y2 == other.y2
        )

    def __hash__(self) -> int:
        return hash(self.xyxy)

    def __and__(self, other: BBox) -> BBox:
        return self.intersection(other)

    def __or__(self, other: BBox) -> BBox:
        return self.union(other)

    def __contains__(self, other: BBox) -> bool:
        return self.contains(other)

    @property
    def x1(self) -> int:
        return self._x1

    @property
    def y1(self) -> int:
        return self._y1

    @property
    def x2(self) -> int:
        return self._x2

    @property
    def y2(self) -> int:
        return self._y2

    @property
    def lu(self) -> tuple[int, int]:
        return self.x1, self.y1

    @property
    def br(self) -> tuple[int, int]:
        return self.x2, self.y2

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.w, self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def shape(self) -> tuple[int, int]:
        return self.h, self.w

    def union(self, other: BBox) -> BBox:
        return BBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2)
        )

    def intersection(self, other: BBox) -> BBox | None:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        if ix1 < ix2 and iy1 < iy2:
            return BBox(ix1, iy1, ix2, iy2)

        return None

    def contains(self, other: BBox) -> bool:
        return (
            other.x1 >= self.x1
            and other.y1 >= self.y1
            and other.x2 <= self.x2
            and other.y2 <= self.y2
        )

    def iou(self, other: BBox) -> float:
        inter = self & other
        if inter is None:
            return 0.0

        union_area = self.area + other.area - inter.area
        if union_area == 0:
            return 0.0

        return inter.area / union_area

    def iom(self, other: BBox) -> float:
        inter = self & other
        if inter is None:
            return 0.0

        min_area = min(self.area, other.area)
        if min_area == 0:
            return 0.0

        return inter.area / min_area

    @classmethod
    def null(cls) -> BBox:
        return BBox(0, 0, 0, 0)

    @property
    def is_null(self) -> bool:
        return self == BBox.null()

    @property
    def centroid(self) -> tuple[int, int]:
        cx = (self.x1 + self.x2) // 2
        cy = (self.y1 + self.y2) // 2

        return cx, cy


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
