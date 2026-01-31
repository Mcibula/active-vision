from __future__ import annotations

import contextlib
import logging
import queue
import signal
import threading
import time
from queue import Queue
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

from utils.logger import PerformanceMonitor, get_logger, timer
from utils.visualization import Color

if TYPE_CHECKING:
    from camera import RealsenseCamera
    from core import RigidObject, Scene
    from structures import ObjectPose


class PipelineController:
    def __init__(
            self,
            camera: RealsenseCamera,
            scene: Scene,
            batch_size: int = 1,
            batch_timeout: float = 0.05,
            capture_limit: int = 500,
            process_every: int = 1
    ) -> None:
        self.shutdown = threading.Event()

        self.camera = camera
        if (
            self.camera.num_streams != 2
            or 'color' not in self.camera
            or 'depth' not in self.camera
            or self.camera['color'].stype != rs.stream.color
            or self.camera['depth'].stype != rs.stream.depth
        ):
            raise ValueError

        self.scene = scene

        self.capture_buffer: Queue[tuple[np.ndarray, np.ndarray]] = Queue()
        self.display_buffer: Queue[np.ndarray] = Queue(maxsize=4)
        self.capture_limit: int = capture_limit

        self.latest_frame_lock = threading.Lock()
        self.latest_camera_frame: np.ndarray | None = None
        self.display_mode: Literal['annotated', 'live'] = 'annotated'

        self.fps_capture: float = 0.0
        self.fps_inference: float = 0.0
        self.dropped_sec: int = 0

        self.batch_size: int = batch_size
        self.batch_timeout: float = batch_timeout
        self.process_every: int = process_every
        self.visibility_thresh: float = 0.5

        self.logger: logging.Logger = get_logger('PipelineController', level=logging.INFO)
        self.monitor: PerformanceMonitor = PerformanceMonitor()

        self.scene.logger = self.logger
        self.scene.monitor = self.monitor
        self.scene.segmenter.logger = self.logger
        self.scene.segmenter.monitor = self.monitor
        self.scene.pose_estimator.logger = self.logger
        self.scene.pose_estimator.monitor = self.monitor

        self.logger.info('PipelineController initialized')

    def warmup(self) -> None:
        self.logger.info('Warming up models...')

        rgb_stream = self.camera['color']
        dummy = np.zeros((rgb_stream.h_res, rgb_stream.w_res, 3), dtype=np.uint8)

        if self.scene.segmenter is not None:
            self.logger.info('Warming up segmenter...')
            self.scene.segmenter.track([dummy])

        if self.scene.pose_estimator is not None:
            self.logger.info('Warming up pose estimator...')
            self.scene.pose_estimator.compute_features(dummy[:640, :640])

        self.logger.info('Warmup completed')

    def run(self) -> None:
        signal.signal(signal.SIGINT, lambda *_: self.shutdown.set())

        self.warmup()

        self.logger.info('Starting threads...')
        t_cam = threading.Thread(target=self._camera_loop, daemon=True)
        t_segm = threading.Thread(target=self._segmenter_loop, daemon=True)
        t_pose = threading.Thread(target=self._pose_loop, daemon=True)

        t_cam.start()
        t_segm.start()
        t_pose.start()

        try:
            self._display_loop()
        except KeyboardInterrupt:
            self.shutdown.set()
        finally:
            self.shutdown.set()
            t_cam.join(timeout=1.0)
            t_segm.join(timeout=1.0)
            t_pose.join(timeout=1.0)
            cv2.destroyAllWindows()

    def _camera_loop(self) -> None:
        self.logger.info('Starting camera stream...')

        self.camera.start_streaming()
        self.camera.warmup(t=4.0)

        num_total = 0
        num_frames = 0
        num_dropped = 0
        t0 = time.time()

        try:
            while not self.shutdown.is_set():
                frames = self.camera.get_frame(['color', 'depth'])

                if frames is None:
                    time.sleep(0.001)
                    continue

                rgb_frame = frames[0].copy()
                d_frame = (frames[1] if frames[1].ndim == 2 else frames[1][:, :, 0]).copy()

                with self.latest_frame_lock:
                    self.latest_camera_frame = rgb_frame

                if num_total % self.process_every == 0:
                    if self.capture_buffer.qsize() > self.capture_limit:
                        with contextlib.suppress(queue.Empty):
                            self.capture_buffer.get_nowait()
                            num_dropped += 1

                    self.capture_buffer.put((rgb_frame, d_frame))

                num_total += 1
                num_frames += 1

                if time.time() - t0 > 1.0:
                    self.fps_capture = num_frames / (time.time() - t0)
                    self.dropped_sec = num_dropped

                    num_frames = 0
                    num_dropped = 0
                    t0 = time.time()

                self.monitor.log_periodically(self.logger)
                time.sleep(0)

        finally:
            self.logger.info('Stopping camera stream...')
            self.camera.stop_streaming()

    def _segmenter_loop(self) -> None:
        num_frames = 0
        t0 = time.time()

        while not self.shutdown.is_set():
            batch = []
            t0_batch = time.time()

            while (
                    len(batch) < self.batch_size
                    and (time.time() - t0_batch <= self.batch_timeout or len(batch) == 0)
            ):
                try:
                    composite_frame = self.capture_buffer.get(timeout=0.005)
                    batch.append(composite_frame)
                except queue.Empty:
                    if len(batch) > 0:
                        break

                    continue

            if not batch:
                continue

            self.scene.read_frames(batch)
            annotated = self.scene.segmenter.last_annotated_frame()

            if annotated is not None:
                with contextlib.suppress(queue.Full):
                    self.display_buffer.put(annotated, timeout=0.01)

            num_frames += len(batch)
            if time.time() - t0 > 1.0:
                self.fps_inference = num_frames / (time.time() - t0)
                num_frames = 0
                t0 = time.time()

            time.sleep(0.001)

    def _pose_loop(self) -> None:
        while not self.shutdown.is_set():
            self.scene.process_pose()

    def _display_loop(self) -> None:
        last_annotated: np.ndarray | None = None

        while not self.shutdown.is_set():
            with contextlib.suppress(queue.Empty):
                annotated = self.display_buffer.get_nowait()
                last_annotated = annotated

            with self.latest_frame_lock:
                live = (
                    None
                    if self.latest_camera_frame is None
                    else self.latest_camera_frame.copy()
                )

            frame = (
                last_annotated
                if self.display_mode == 'annotated' and last_annotated is not None
                else live
            )

            if frame is not None:
                display: np.ndarray = frame.copy()

                now = time.time()
                for obj in list(self.scene):
                    if (now - obj.last_updated) > self.visibility_thresh:
                        continue

                    self._draw_pose(display, obj)

                self._draw_hud(display)
                cv2.imshow(
                    'Inference',
                    cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.logger.info('Shutting down...')
                self.shutdown.set()
                break
            if key == ord('m'):
                self.display_mode = (
                    'annotated'
                    if self.display_mode == 'live'
                    else 'live'
                )

    @timer('PipelineController._draw_hud')
    def _draw_hud(self, frame: np.ndarray) -> None:
        backlog = self.capture_buffer.qsize()

        status_color = (
            Color.RED if backlog > 50
            else Color.YELLOW if backlog > 10
            else Color.GREEN
        )

        info = (
            f'Mode:     {self.display_mode.upper()}',
            f'Cam FPS: {self.fps_capture:.2f} (PROC: 1/{self.process_every})',
            f'Proc FPS: {self.fps_inference:.2f}',
            f'Backlog:  {backlog} frames' + (
                f' (DROP: {self.dropped_sec}/s)'
                if self.dropped_sec > 0 else ''
            )
        )

        colors = {
            'mode': Color.CYAN,
            'backlog': status_color
        }

        y = 25
        for line in info:
            cv2.putText(
                img=frame, text=line, org=(10, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=2,
                color=colors.get(line.split(':')[0].lower(), Color.BLACK)
            )
            y += 25

    @timer('PipelineController._draw_pose')
    def _draw_pose(self, frame: np.ndarray, obj: RigidObject) -> None:
        pose: ObjectPose = obj.pose

        if pose is None or pose.is_lost:
            return

        try:
            R: np.ndarray = Rotation.from_euler('xyz', pose.rot).as_matrix()
            rvec, _ = cv2.Rodrigues(R)
            tvec = np.array(pose.pos, dtype=np.float32)

            axis_len = 0.1
            obj_points = np.vstack(
                (
                    np.zeros(3,),
                    np.eye(3) * axis_len
                ),
                dtype=np.float32
            )

            intrinsics = self.camera['color'].intrinsics
            img_points, _ = cv2.projectPoints(
                objectPoints=obj_points,
                rvec=rvec, tvec=tvec,
                cameraMatrix=intrinsics.K,
                distCoeffs=intrinsics.coeffs
            )
            img_points = img_points.astype(int).reshape(-1, 2)

            origin = tuple(img_points[0])
            for idx, clr in enumerate((Color.RED, Color.GREEN, Color.BLUE)):
                cv2.line(
                    img=frame,
                    pt1=origin, pt2=tuple(img_points[idx + 1]),
                    color=clr, thickness=3
                )

            label = f'#{obj.obj_id} {pose.z:.3f}m'
            cv2.putText(
                img=frame, text=label, org=(origin[0], origin[1] - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2,
                color=Color.BLACK
            )
        except Exception:
            return
