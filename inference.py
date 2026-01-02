import contextlib
import queue
import signal
import threading
import time
from queue import Queue
from typing import Literal

import cv2
import numpy as np

from camera.realsense import RealsenseCamera
from camera.realsense_profiles import FHD_RGB
from processors.segmenter import Segmenter
from scene import Scene
from utils.visualization import BLACK, CYAN, GREEN, RED, YELLOW


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
        self.scene = scene

        self.capture_buffer: Queue[np.ndarray] = Queue()
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

    def run(self) -> None:
        signal.signal(signal.SIGINT, lambda *_: self.shutdown.set())

        t_cam = threading.Thread(target=self._camera_loop, daemon=True)
        t_proc = threading.Thread(target=self._processing_loop, daemon=True)

        t_cam.start()
        t_proc.start()

        try:
            self._display_loop()
        except KeyboardInterrupt:
            self.shutdown.set()
        finally:
            self.shutdown.set()
            t_cam.join(timeout=1.0)
            t_proc.join(timeout=1.0)
            cv2.destroyAllWindows()

    def _camera_loop(self) -> None:
        self.camera.start_streaming()
        self.camera.warmup()

        num_total = 0
        num_frames = 0
        num_dropped = 0
        t0 = time.time()

        try:
            while not self.shutdown.is_set():
                frames = self.camera.get_frame(['color'])

                if frames is None:
                    time.sleep(0.001)
                    continue

                frame = frames[0].copy()

                with self.latest_frame_lock:
                    self.latest_camera_frame = frame

                if num_total % self.process_every == 0:
                    if self.capture_buffer.qsize() > self.capture_limit:
                        with contextlib.suppress(queue.Empty):
                            self.capture_buffer.get_nowait()
                            num_dropped += 1

                    self.capture_buffer.put(frame)

                num_total += 1
                num_frames += 1

                if time.time() - t0 > 1.0:
                    self.fps_capture = num_frames / (time.time() - t0)
                    self.dropped_sec = num_dropped

                    num_frames = 0
                    num_dropped = 0
                    t0 = time.time()

                time.sleep(0)

        finally:
            self.camera.stop_streaming()

    def _processing_loop(self) -> None:
        num_frames = 0
        t0 = time.time()

        while not self.shutdown.is_set():
            batch = []
            t0_batch = time.time()

            while (
                    len(batch) < self.batch_size
                    and ((time.time() - t0_batch) <= self.batch_timeout or len(batch) == 0)
            ):
                try:
                    frame = self.capture_buffer.get(timeout=0.005)
                    batch.append(frame)
                except queue.Empty:
                    if len(batch) > 0:
                        break

                    continue

            if not batch:
                continue

            self.scene.read_frames(batch)
            annotated = self.scene.segmenter.last_annotated_frames()

            for af in annotated:
                with contextlib.suppress(queue.Full):
                    self.display_buffer.put(af, timeout=0.01)

            num_frames += len(batch)
            if time.time() - t0 > 1.0:
                self.fps_inference = num_frames / (time.time() - t0)
                num_frames = 0
                t0 = time.time()

            time.sleep(0.001)

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
                self._draw_hud(display)
                cv2.imshow(
                    'Inference',
                    cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.shutdown.set()
                break
            if key == ord('m'):
                self.display_mode = (
                    'annotated'
                    if self.display_mode == 'live'
                    else 'live'
                )

    def _draw_hud(self, frame: np.ndarray) -> None:
        backlog = self.capture_buffer.qsize()

        status_color = (
            RED if backlog > 50
            else YELLOW if backlog > 10
            else GREEN
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
            'mode': CYAN,
            'backlog': status_color
        }

        y = 25
        for line in info:
            cv2.putText(
                img=frame, text=line, org=(10, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=2,
                color=colors.get(line.split(':')[0].lower(), BLACK)
            )
            y += 25


def main() -> Scene:
    camera = RealsenseCamera(FHD_RGB)
    segmenter = Segmenter(
        engine='yoloe',
        weights='./models/yolo/yoloe-11l-seg-pf.pt'
    )
    scene = Scene(segmenter=segmenter)

    controller = PipelineController(
        camera=camera,
        scene=scene,
        batch_size=4,
        batch_timeout=0.2,
        process_every=2
    )
    controller.run()

    return scene


if __name__ == '__main__':
    out_scene = main()
