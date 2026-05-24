from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from core.scene import RigidObject, Scene
    from structures import BBox, ObjectPose


class SceneDescriptor:
    """
    Extract lightweight real-time descriptors from a tracked scene
    """

    def __init__(self, visible_timeout: float | None = None) -> None:
        """
        Initialize the scene descriptor

        :param visible_timeout: Maximum object age in seconds for `visible == True`;
                                if `None`, `Scene.temp_thresh` is used
        """

        if visible_timeout is not None and visible_timeout <= 0.0:
            raise ValueError

        self.visible_timeout: float | None = visible_timeout

    def __call__(self, scene: Scene, timestamp: float | None = None) -> dict[str, Any]:
        """
        Describe the current object-centric state of a scene

        :param scene: Scene to describe
        :param timestamp: Optional descriptor timestamp; if `None`, current time is used
        :return: JSON-friendly scene descriptor
        """

        timestamp = (
            time.time()
            if timestamp is None
            else float(timestamp)
        )
        visible_timeout = (
            scene.temp_thresh
            if self.visible_timeout is None
            else self.visible_timeout
        )

        objects = [
            self._object_descriptor(
                obj,
                timestamp=timestamp,
                visible_timeout=visible_timeout
            )
            for obj in scene
        ]

        return {
            'timestamp': timestamp,
            'frame_count': int(scene.frame_count),
            'num_objects': int(scene.num_objects),
            'num_visible_objects': sum(
                1
                for obj in objects
                if obj['visible']
            ),
            'objects': objects,
        }

    @staticmethod
    def _object_descriptor(
            obj: RigidObject,
            timestamp: float,
            visible_timeout: float
    ) -> dict[str, Any]:
        """
        Describe the current state of a tracked object

        :param obj: `RigidObject` to describe
        :param timestamp: Descriptor timestamp
        :param visible_timeout: Maximum object age in seconds for `visible == True`
        :return: JSON-friendly object descriptor
        """

        age_s = (
            timestamp - obj.last_updated
            if obj.last_updated > 0.0
            else None
        )

        pose = obj.pose
        valid_pose = pose is not None and not pose.is_lost

        velocity = (
            SceneDescriptor._velocity(obj)
            if valid_pose
            else None
        )

        return {
            'obj_id': int(obj.obj_id),
            'visible': bool(age_s is not None and age_s <= visible_timeout),
            'age_s': None if age_s is None else float(age_s),
            'last_updated': float(obj.last_updated),
            'is_busy': bool(obj.is_busy),
            'bbox': SceneDescriptor._bbox_descriptor(obj.last_seen),
            'num_snapshots': int(obj.num_snapshots),
            'num_poses': int(obj.num_poses),
            'num_valid_poses': int(obj.trajectory.num_valid),
            'has_pose': bool(valid_pose),
            'pose': (
                SceneDescriptor._pose_descriptor(pose)
                if valid_pose
                else None
            ),
            'velocity': velocity,
            'last_pose_diagnostics': SceneDescriptor._json_value(
                obj.last_pose_diagnostics
            ),
        }

    @staticmethod
    def _bbox_descriptor(bbox: BBox) -> dict[str, Any] | None:
        """
        Convert a bounding box to a JSON-friendly descriptor

        :param bbox: Bounding box to describe
        :return: Bounding box descriptor or `None` for a null box
        """

        if bbox.is_null:
            return None

        return {
            'xyxy': list(map(int, bbox.xyxy)),
            'xywh': list(map(int, bbox.xywh)),
            'centroid': list(map(int, bbox.centroid)),
            'area': int(bbox.area),
        }

    @staticmethod
    def _pose_descriptor(pose: ObjectPose) -> dict[str, Any]:
        """
        Convert an object pose to a JSON-friendly descriptor

        :param pose: Object pose to describe
        :return: Pose descriptor
        """

        return {
            'position': list(map(float, pose.pos)),
            'orientation': list(map(float, pose.rot)),
            'timestamp': float(pose.timestamp),
        }

    @staticmethod
    def _velocity(obj: RigidObject) -> dict[str, Any] | None:
        """
        Estimate current translational velocity from the last two valid poses

        :param obj: `RigidObject` to describe
        :return: Velocity descriptor or `None` if velocity cannot be estimated
        """

        poses = obj.trajectory.valid
        if len(poses) < 2:
            return None

        prev, cur = poses[-2], poses[-1]
        dt = cur.timestamp - prev.timestamp
        if dt <= 0.0:
            return None

        vx = (cur.x - prev.x) / dt
        vy = (cur.y - prev.y) / dt
        vz = (cur.z - prev.z) / dt

        return {
            'linear': [float(vx), float(vy), float(vz)],
            'dt_s': float(dt),
        }

    @staticmethod
    def _json_value(value: Any) -> Any:
        """
        Convert common runtime values to JSON-friendly Python values

        :param value: Value to convert
        :return: JSON-friendly value
        """

        if value is None or isinstance(value, str | int | float | bool):
            return value

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, dict):
            return {
                str(k): SceneDescriptor._json_value(v)
                for k, v in value.items()
            }

        if isinstance(value, list | tuple):
            return [
                SceneDescriptor._json_value(v)
                for v in value
            ]

        return str(value)
