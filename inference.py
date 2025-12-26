import asyncio
import signal
import threading
from asyncio import AbstractEventLoop

import cv2
import numpy as np

from camera.realsense import RealsenseCamera
from camera.realsense_profiles import FHD_RGB
from processors.segmenter import Segmenter
from scene import Scene
from utils.pipeline import BoundedFIFO, FrameBufferMetrics, InstrumentedBuffer, LosslessFIFO


class PipelineController:
    def __init__(
            self,
            camera: RealsenseCamera,
            scene: Scene,
            batch_size: int = 1,
            batch_timeout: float | None = None
    ) -> None:
        self.shutdown = threading.Event()

        self.camera = camera
        self.scene = scene

        self.capture_metrics = FrameBufferMetrics('capture')
        self.display_metrics = FrameBufferMetrics('display')

        self.capture_buffer: InstrumentedBuffer[np.ndarray] = InstrumentedBuffer(
            inner=LosslessFIFO(),
            metrics=self.capture_metrics,
            timestamp=True
        )
        self.display_buffer: InstrumentedBuffer[np.ndarray] = InstrumentedBuffer(
            inner=BoundedFIFO(32),
            metrics=self.display_metrics,
            timestamp=False
        )

        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

    async def run(self):
        signal.signal(signal.SIGINT, lambda *_: self.shutdown.set())

        loop = asyncio.get_running_loop()
        threading.Thread(
            target=self._camera_loop,
            args=(loop,),
            daemon=True
        ).start()
        threading.Thread(
            target=self._display_loop,
            daemon=True
        ).start()

        await self._processing_loop()

    def _camera_loop(self, loop: AbstractEventLoop) -> None:
        try:
            self.camera.start_streaming()
            self.camera.warmup()

            while not self.shutdown.is_set():
                try:
                    frame = self.camera.get_frame(['color'])[0]
                except RuntimeError:
                    continue

                loop.call_soon_threadsafe(
                    self.capture_buffer.write,
                    frame
                )
        finally:
            self.camera.stop_streaming()

    async def _processing_loop(self) -> None:
        queue: LosslessFIFO[tuple[np.ndarray, float]] = self.capture_buffer.inner

        while not self.shutdown.is_set():
            self.scene.read_frames([
                frame
                for frame, _ in await queue.read_batch(
                    batch_size=self.batch_size,
                    timeout=self.batch_timeout
                )
            ])

            self.display_buffer.write_batch(
                self.scene
                    .segmenter
                    .last_annotated_frames()
            )

    def _display_loop(self) -> None:
        last_frame: np.ndarray | None = None

        while not self.shutdown.is_set():
            frame = self.display_buffer.read()

            if frame is not None:
                last_frame = frame

            if last_frame is not None:
                self.capture_metrics.on_size(len(self.capture_buffer))
                self.display_metrics.on_size(len(self.display_buffer))

                display: np.ndarray = last_frame.copy()

                cap = self.capture_metrics.report()
                disp = self.display_metrics.report()

                y = 25
                for name, m in [('CAP', cap), ('DSP', disp)]:
                    report = (
                        f'{name} '
                        f'W:{m["writes"]} '
                        f'R:{m["reads"]} '
                        f'S:{m["size"]} '
                        f'L:{m["lat_ms"]:.1f}ms'
                    )
                    cv2.putText(
                        img=display,
                        text=report,
                        org=(10, y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6,
                        color=(255, 255, 0),
                        thickness=2,
                    )
                    y += 25

                cv2.imshow(
                    'Inference',
                    cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
                )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown.set()
                break

        cv2.destroyAllWindows()


async def main() -> Scene:
    camera = RealsenseCamera(FHD_RGB)
    segmenter = Segmenter(
        engine='yoloe',
        weights='./models/yolo/yoloe-11l-seg-pf.pt'
    )
    scene = Scene(segmenter=segmenter)

    controller = PipelineController(
        camera=camera,
        scene=scene,
        batch_size=30,
        batch_timeout=0.02
    )
    await controller.run()

    return scene


if __name__ == '__main__':
    out_scene = asyncio.run(main())
