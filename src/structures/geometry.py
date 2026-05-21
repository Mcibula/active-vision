"""
Data structures operating with geometrical data
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation

from utils.visualization import Color

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


@dataclass
class TrackRecord:
    """
    Object grouping data obtained by the tracker in `processors.segmenter.Segmenter`
    """

    xyxy: list[BBox]
    masks: list[np.ndarray]
    snapshots: list[np.ndarray]
    frame_ids: list[int]


class BBox:
    """
    Object representing a 2D bounding box and supporting operations on it
    """

    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """
        Construct a bounding box from the pixel coordinates
        of the left top and right bottom corners of the bounding box

        :param x1: Pixel coordinate along the width of the image
                   of the left top corner
        :param y1: Pixel coordinate along the height of the image
                   of the left top corner
        :param x2: Pixel coordinate along the width of the image
                   of the right bottom corner
        :param y2: Pixel coordinate along the height of the image
                   of the right bottom corner
        """

        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            raise ValueError

        if x1 > x2 or y1 > y2:
            raise ValueError

        # Enforce integers
        self._x1: int = int(x1)
        self._y1: int = int(y1)
        self._x2: int = int(x2)
        self._y2: int = int(y2)

    def __repr__(self) -> str:
        return f'<BBox {self.w}x{self.h} @ {self.lt}, {self.rb}>'

    def __iter__(self) -> Iterator[int]:
        yield from self.xyxy

    def __eq__(self, other: BBox) -> bool:
        if not isinstance(other, BBox):
            return False

        return (
            self.x1 == other.x1 and self.y1 == other.y1
            and self.x2 == other.x2 and self.y2 == other.y2
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
        """
        Pixel coordinate along the width of the image of the left top corner
        """

        return self._x1

    @property
    def y1(self) -> int:
        """
        Pixel coordinate along the height of the image of the left top corner
        """

        return self._y1

    @property
    def x2(self) -> int:
        """
        Pixel coordinate along the width of the image of the right bottom corner
        """

        return self._x2

    @property
    def y2(self) -> int:
        """
        Pixel coordinate along the height of the image of the right bottom corner
        """

        return self._y2

    @property
    def lt(self) -> tuple[int, int]:
        """
        XY coordinates of the left top corner of this bounding box
        """

        return self.x1, self.y1

    @property
    def rb(self) -> tuple[int, int]:
        """
        XY coordinates of the right bottom corner of this bounding box
        """

        return self.x2, self.y2

    @property
    def w(self) -> int:
        """
        Width of this bounding box
        """

        return self.x2 - self.x1

    @property
    def h(self) -> int:
        """
        Height of this bounding box
        """

        return self.y2 - self.y1

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """
        XYXY representation of this bounding box
        """

        return self.x1, self.y1, self.x2, self.y2

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        """
        XYWH representation of this bounding box
        """

        return self.x1, self.y1, self.w, self.h

    @property
    def area(self) -> int:
        """
        Area of this bounding box
        """

        return self.w * self.h

    @property
    def shape(self) -> tuple[int, int]:
        """
        Dimensions of this bounding box as `(H, W)`
        """

        return self.h, self.w

    def union(self, other: BBox) -> BBox:
        """
        Construct an union of this and the `other` bounding box
        """

        return BBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2)
        )

    def intersection(self, other: BBox) -> BBox | None:
        """
        Construct an intersection of this and the `other` bounding box

        :return: An intersecting area as a new `BBox`
                 or `None` if the intersection does not exist
        """

        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        if ix1 < ix2 and iy1 < iy2:
            return BBox(ix1, iy1, ix2, iy2)

        return None

    def contains(self, other: BBox) -> bool:
        """
        Check if this bounding box is fully contained by the `other` bounding box
        """

        return (
            other.x1 >= self.x1
            and other.y1 >= self.y1
            and other.x2 <= self.x2
            and other.y2 <= self.y2
        )

    def iou(self, other: BBox) -> float:
        r"""
        Calculate the intersection over union area between this and the `other` bounding box:
        .. math::
            \mathrm{IoU}\left(B_1, B_2\right)
              = \frac{A\left(B_1 \cap B_2\right)}{A\left(B_1\right) + A\left(B_2\right) - A\left(B_1 \cap B_2\right)}

        :return: Pixel IoU area
        """

        inter = self & other
        if inter is None:
            return 0.0

        union_area = self.area + other.area - inter.area
        if union_area == 0:
            return 0.0

        return inter.area / union_area

    def iom(self, other: BBox) -> float:
        r"""
        Calculate the intersection over minimum area between this and the `other` bounding box:
        .. math::
            \mathrm{IoM}\left(B_1, B_2\right)
              = \frac{A\left(B_1 \cap B_2\right)}{\min\left\{A\left(B_1\right), A\left(B_2\right)\right\}}

        :return: Pixel IoM area
        """

        inter = self & other
        if inter is None:
            return 0.0

        min_area = min(self.area, other.area)
        if min_area == 0:
            return 0.0

        return inter.area / min_area

    @classmethod
    def null(cls) -> BBox:
        """
        Construct a null bounding box
        """

        return BBox(0, 0, 0, 0)

    @property
    def is_null(self) -> bool:
        """
        Check if this bounding box is null
        """

        return self == BBox.null()

    @property
    def centroid(self) -> tuple[int, int]:
        """
        XY coordinates of the centroid of this bounding box
        """

        cx = (self.x1 + self.x2) // 2
        cy = (self.y1 + self.y2) // 2

        return cx, cy


class ObjectPose:
    """
    6-DoF object pose
    """

    def __init__(
            self,
            x: float, y: float, z: float,
            rx: float, ry: float, rz: float,
            is_valid: bool = True
    ) -> None:
        """
        Create a new 6-DoF object pose

        :param x: Position along the width relative to the viewpoint
        :param y: Position along the depth relative to the viewpoint
        :param z: Position along the height relative to the viewpoint
        :param rx: Euler angle along the x-axis in radians
        :param ry: Euler angle along the y-axis in radians
        :param rz: Euler angle along the z-axis in radians
        :param is_valid: If `False`, this pose will be marked as `[LOST]`
        """

        self.x: float = x
        self.y: float = y
        self.z: float = z

        self.rx: float = rx
        self.ry: float = ry
        self.rz: float = rz

        self._is_valid: bool = is_valid
        self._timestamp: float = time.time()

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
        """
        Generate the pose from the transformation matrix

        :param pose_matrix: 4x4 transformation matrix
        :return: New `ObjectPose` object
        """

        if pose_matrix.shape != (4, 4):
            raise ValueError

        # Get the translation vector
        tx, ty, tz = pose_matrix[:3, 3]

        # Convert the rotation matrix to the Euler angle vector
        r = Rotation.from_matrix(pose_matrix[:3, :3])
        rx, ry, rz = r.as_euler('xyz')

        return cls(
            x=tx, y=ty, z=tz,
            rx=rx, ry=ry, rz=rz,
            is_valid=True
        )

    @classmethod
    def lost(cls) -> ObjectPose:
        """
        Generate a lost pose

        :return: `ObjectPose` object indicating lost pose
        """

        return cls(
            x=np.nan, y=np.nan, z=np.nan,
            rx=np.nan, ry=np.nan, rz=np.nan,
            is_valid=False
        )

    @property
    def pos(self) -> tuple[float, float, float]:
        """
        Position of the object along the :math:`x`-, :math:`y`-, and :math:`z`-axis
        relative to the viewpoint
        """

        return self.x, self.y, self.z

    @property
    def rot(self) -> tuple[float, float, float]:
        """
        Euler angle orientation of the object in radians
        along the :math:`x`-, :math:`y`-, and :math:`z`-axis relative to the viewpoint
        """

        return self.rx, self.ry, self.rz

    @property
    def pitch(self) -> float:
        """
        Pitch (:math:`r_x`) angle in radians
        """

        return self.rx

    @property
    def yaw(self) -> float:
        """
        Yaw (:math:`r_y`) angle in radians
        """

        return self.ry

    @property
    def roll(self) -> float:
        """
        Roll (:math:`r_z`) angle in radians
        """

        return self.rz

    @property
    def pose_6d(self) -> tuple[float, float, float, float, float, float]:
        r"""
        6-DoF pose of the object as a vector
        .. math::
            \left[p_x, p_y, p_z, r_x, r_y, r_z\right].

        The angles are in radians.
        """

        return self.pos + self.rot

    @property
    def is_lost(self) -> bool:
        """
        If `True` the pose is invalid
        """

        return not self._is_valid

    @property
    def timestamp(self) -> float:
        """
        UNIX time when the pose was created
        """

        return self._timestamp


class Trajectory:
    """
    Object preserving series of `ObjectPose` records
    as a spatiotemporal trajectory of the movement of a `RigidObject`
    """

    def __init__(self, capacity: int | None = None) -> None:
        """
        Initialize an empty trajectory

        :param capacity: Number of poses to keep; if `None`, the capacity is unlimited.
                         If the capacity is exceeded, the oldest pose will be removed
        """

        if capacity is not None and capacity <= 0:
            raise ValueError

        self._capacity: int | None = capacity
        self._poses: list[ObjectPose] = []

    def __repr__(self) -> str:
        return f'<Trajectory of {len(self._poses)} object poses>'

    def __len__(self) -> int:
        return len(self._poses)

    def __getitem__(self, item: int) -> ObjectPose:
        return self._poses[item]

    def __iter__(self) -> Iterator[ObjectPose]:
        return iter(self._poses)

    def _pose_accessor(self, attrs: list[str]) -> list[Any] | list[list[Any]]:
        """
        Access attributes of each `ObjectPose` recorded
        and aggregate them across the trajectory

        :param attrs: Attributes to access
        :return: List of attribute values; if more than one attribute is given,
                 a 2D list is returned
        """

        if not attrs:
            raise ValueError

        return [
            [
                getattr(p, attr)
                for attr in attrs
            ]
            if len(attrs) > 1
            else getattr(p, attrs[0])
            for p in self._poses
            if not p.is_lost
        ]

    @property
    def capacity(self) -> int | None:
        """
        Maximum capacity of this trajectory
        """

        return self._capacity

    @property
    def valid(self) -> list[ObjectPose]:
        """
        Valid object poses, excluding `<ObjectPose [LOST]>`
        """

        return [
            p
            for p in self._poses
            if not p.is_lost
        ]

    @property
    def num_valid(self) -> int:
        """
        Number of valid poses, excluding `<ObjectPose [LOST]>`
        """

        return len(self.valid)

    @property
    def positions(self) -> np.ndarray | None:
        r"""
        XYZ coordinates of all poses recorded as a :math:`\mathbb{R}^{N \times 3}` matrix
        for :math:`N` valid poses. `None` if no object positions have been recorded
        """

        positions = self._pose_accessor(['pos'])
        if not positions:
            return None

        return np.array(positions, dtype=np.float32)

    @property
    def orientations(self) -> np.ndarray | None:
        r"""
        XYZ Euler angles in radians of all poses recorded
        as a :math:`\mathbb{R}^{N \times 3}` matrix for :math:`N` valid poses.
        `None` if no object orientations have been recorded
        """

        rotations = self._pose_accessor(['rot'])
        if not rotations:
            return None

        return np.array(rotations, dtype=np.float32)

    @property
    def points(self) -> np.ndarray | None:
        r"""
        Positions and orientations of all poses recorded in the form of
        a :math:`\mathbb{R}^{N \times 6}` matrix for :math:`N` valid poses.
        `None` if no object poses have been recorded
        """

        positions = self.positions

        if positions is None:
            return None

        orientations = self.orientations
        return np.hstack((positions, orientations))

    @classmethod
    def from_list(cls, poses: list[ObjectPose]) -> Trajectory:
        """
        Create a trajectory from a list of `ObjectPose`

        :param poses: List of object poses
        :return: Reconstructed trajectory
        """

        traj = cls(capacity=len(poses))
        traj._poses = poses

        return traj

    def append(self, pose: ObjectPose) -> None:
        """
        Record a new `ObjectPose`.
        If by adding this pose the capacity of the trajectory would be exceeded,
        the oldest pose is removed

        :param pose: A new object pose
        """

        self._poses.append(pose)

        if self._capacity is not None and len(self._poses) > self._capacity:
            self._poses.pop(0)

    def plot(
            self,
            ax: Axes3D | None = None,
            color: tuple[int, int, int] = Color.BLUE,
            n_orient: int = 0,
            show: bool = True
    ) -> Axes3D:
        r"""
        Plot the trajectory

        :param ax: Pre-existing 3D axes to use; if `None`, a new one will be created
        :param color: RGB color of the plotted trajectory with values in range :math:`[0, 255]`
        :param n_orient: If :math:`> 0`, the orientation of the object along the trajectory
                         will be visualized by :math:`n_\text{orient}` or :math:`n_\text{orient} + 1`
                         uniformly distributed coordinate triads
        :param show: If `True`, the final plot will be shown
        :return: 3D axes with the plot
        """

        n_orient = max(0, n_orient)

        # Convert positions to centimeters
        unit_mult: int = 100
        pos = self.positions * unit_mult

        if len(pos) < 2:
            return ax

        # If no pre-existing trajectory is given, create a new one
        created = False
        if ax is None:
            created = True
            fig, ax = plt.subplots(
                figsize=(10, 8),
                subplot_kw={'projection': '3d'}
            )

        # Permute trajectory dimensions, so they are aligned with the displayed space
        px, py, pz = pos[:, 0], pos[:, 2], -pos[:, 1]
        p_traj = np.column_stack((px, py, pz))

        # Infer limits and scale
        max_d = np.maximum(p_traj.max(axis=0), 0)
        min_d = np.minimum(p_traj.min(axis=0), 0)

        max_rng = np.max(max_d - min_d) * 0.5 * 1.2
        mid_x, mid_y, mid_z = (max_d + min_d) * 0.5

        min_x = mid_x - max_rng
        max_x = mid_x + max_rng
        ax.set_xlim(min_x, max_x)

        min_y = mid_y - max_rng
        max_y = mid_y + max_rng
        ax.set_ylim(min_y, max_y)

        min_z = mid_z - max_rng
        max_z = mid_z + max_rng
        ax.set_zlim(min_z, max_z)

        # Split trajectory to line segments
        traj_reshaped = p_traj.reshape(-1, 1, 3)
        segments = np.concatenate(
            [traj_reshaped[:-1], traj_reshaped[1:]],
            axis=1
        )

        # Color each segment with increasingly saturated colors
        color = np.array(color) / 255.0
        base_h, _, base_v = mplc.rgb_to_hsv(color)
        colors = np.array([
            mplc.hsv_to_rgb([base_h, s, base_v])
            for s in np.linspace(0.2, 1.0, len(segments))
        ])
        ax.add_collection(Line3DCollection(segments, colors=colors))

        # Add end-point markers
        ax.scatter(*p_traj[0], c=colors[0], s=30, label='Start')
        ax.scatter(*p_traj[-1], c=colors[-1], marker='x', s=60, label='End')

        # Plot camera axes at the origin
        length = (max_x - min_x) * 0.1
        ax.quiver(
            0, 0, 0, length, 0, 0,
            color='r', arrow_length_ratio=0.3, linewidth=2
        )
        ax.quiver(
            0, 0, 0, 0, length, 0,
            color='g', arrow_length_ratio=0.3, linewidth=2
        )
        ax.quiver(
            0, 0, 0, 0, 0, length,
            color='b', arrow_length_ratio=0.3, linewidth=2
        )

        # If the orientation drawing is allowed
        if n_orient > 0:
            # Get a subset of poses to use
            points: list[ObjectPose] = self.valid
            step = max(1, len(points) // n_orient)
            subset = points[::step]

            # Forcibly include the last pose
            if points[-1] not in subset:
                subset.append(points[-1])

            for pose in subset:
                # Object origin
                rpx = pose.pos[0] * unit_mult
                rpy = pose.pos[2] * unit_mult
                rpz = -pose.pos[1] * unit_mult

                # Direction vectors
                R = Rotation.from_euler('xyz', pose.rot).as_matrix()
                vx_x, vx_y, vx_z = R[0, 0], R[2, 0], -R[1, 0]
                vy_x, vy_y, vy_z = R[0, 1], R[2, 1], -R[1, 1]
                vz_x, vz_y, vz_z = R[0, 2], R[2, 2], -R[1, 2]

                # Plot the triads
                ax.quiver(
                    rpx, rpy, rpz,
                    vx_x, vx_y, vx_z,
                    color='r', length=length, normalize=True, alpha=0.8
                )
                ax.quiver(
                    rpx, rpy, rpz,
                    vy_x, vy_y, vy_z,
                    color='g', length=length, normalize=True, alpha=0.8
                )
                ax.quiver(
                    rpx, rpy, rpz,
                    vz_x, vz_y, vz_z,
                    color='b', length=length, normalize=True, alpha=0.8
                )

        # Get the trajectory range per dimension
        d_min_x, d_max_x = px.min(), px.max()
        d_min_y, d_max_y = py.min(), py.max()
        d_min_z, d_max_z = pz.min(), pz.max()

        # Plot the range indicators
        style = {
            'linewidth': 2,
            'linestyle': '-',
            'markersize': 10,
            'markeredgewidth': 3
        }

        ax.plot(
            (d_min_x, d_max_x), (min_y, min_y), (min_z, min_z),
            color='red', marker='|', **style
        )
        ax.plot(
            (max_x, max_x), (d_min_y, d_max_y), (min_z, min_z),
            color='green', marker='|', **style
        )
        ax.plot(
            (max_x, max_x), (max_y, max_y), (d_min_z, d_max_z),
            color='blue', marker='_', **style
        )

        # Display the axis labels with an offset
        ax.set_xlabel('X (width) [cm]', labelpad=10)
        ax.set_ylabel('Z (depth) [cm]', labelpad=10)
        ax.set_zlabel('Y (height) [cm]', labelpad=10)

        if created:
            ax.view_init(elev=30, azim=-30)
            fig.legend()
            fig.tight_layout()

        if show:
            plt.show()

        return ax
