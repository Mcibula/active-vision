from collections.abc import MutableMapping
from typing import Callable, Iterator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs


class Stream:
    def __init__(
            self,
            name: str,
            stype: rs.stream,
            w_res: int,
            h_res: int,
            dtype: rs.format,
            fps: int,
            extractor: Callable[[rs.composite_frame], rs.frame],
            colorizer: rs.colorizer | None = None,
            sid: int = -1
    ) -> None:
        self.name = name
        self.stype = stype
        self.w_res = w_res
        self.h_res = h_res
        self.dtype = dtype
        self.fps = fps
        self.extractor = extractor
        self.colorizer = colorizer
        self.sid = sid

    @property
    def config(self) -> tuple:
        if self.sid > 0:
            return self.stype, self.sid, self.w_res, self.h_res, self.dtype, self.fps

        return self.stype, self.w_res, self.h_res, self.dtype, self.fps

    @property
    def resolution(self) -> tuple[int, int]:
        return self.w_res, self.h_res

    def extract(self, composite_frame: rs.composite_frame) -> rs.frame:
        return self.extractor(composite_frame)

    def to_numpy(self, frame: rs.composite_frame | rs.frame) -> np.ndarray:
        if isinstance(frame, rs.composite_frame):
            frame = self.extract(frame)

        if self.colorizer is not None:
            frame = self.colorizer.colorize(frame)

        image = np.asanyarray(frame.get_data())

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.ndim == 3:
            return image

        raise ValueError


class Streams(MutableMapping):
    def __init__(self, streams: list[Stream]) -> None:
        self._streams: dict[str, Stream] = {}
        self.update({
            stream.name: stream
            for stream in streams
        })

    def __getitem__(self, key: str) -> Stream:
        return self._streams[key]

    def __setitem__(self, key: str, value: Stream) -> None:
        self._streams[key] = value

    def __delitem__(self, key: str) -> None:
        del self._streams[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._streams)

    def __len__(self) -> int:
        return len(self._streams)

    @property
    def lowest_resolution(self) -> tuple[int, int]:
        return min(
            [
                stream.resolution
                for stream in self._streams.values()
            ],
            key=lambda res: res[0] * res[1]
        )

    def any(self, composite_frame: rs.composite_frame) -> bool:
        return any(
            stream.extract(composite_frame)
            for stream in self._streams.values()
        )

    def values(self) -> list[Stream]:
        return list(self._streams.values())

    def keys(self) -> list[str]:
        return list(self._streams.keys())


class RealsenseCamera:
    def __init__(self, streams: Streams) -> None:
        self.streams: Streams = streams
        self._devices: rs.device_list = rs.context().query_devices()

        self._reset_devices()

        self.pipeline: rs.pipeline = rs.pipeline()
        self.config: rs.config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        self.device: rs.device = pipeline_profile.get_device()
        self.name: str = self.device.get_info(rs.camera_info.name)

        print(
            f'Initializing {self.name} '
            f'with {self.num_sensors} sensor{"s" if self.num_sensors > 1 else ""}: '
            f'{", ".join(self.sensors)}...'
        )

        for stream in self.streams.values():
            self.config.enable_stream(*stream.config)

        self.streaming: bool = False

    def __del__(self) -> None:
        if self.streaming:
            self.stop_streaming()

    @property
    def sensors(self) -> dict[str, rs.sensor]:
        return {
            s.get_info(rs.camera_info.name): s
            for s in self.device.sensors
        }

    @property
    def num_sensors(self) -> int:
        return len(self.sensors)

    def _reset_devices(self) -> None:
        for dev in self._devices:
            dev.hardware_reset()

    def start_streaming(self) -> None:
        self._reset_devices()
        self.pipeline.start(self.config)
        self.streaming = True

    def stop_streaming(self) -> None:
        self.pipeline.stop()
        self.streaming = False

    def get_composite_frame(self) -> rs.composite_frame | None:
        if not self.streaming:
            self.start_streaming()

        success, composite = self.pipeline.try_wait_for_frames(timeout_ms=0)

        if success:
            return composite

        return None

    def get_frame(self, stream_names: list[str]) -> list[np.ndarray] | None:
        if not stream_names or not set(self.streams).issuperset(set(stream_names)):
            raise ValueError

        composite = self.get_composite_frame()
        if composite is None:
            return None

        frames = []

        for stream_name in stream_names:
            stream = self.streams[stream_name]
            frames.append(stream.to_numpy(composite))

        return frames

    def warmup(self) -> None:
        if not self.streaming:
            self.start_streaming()

        stream = self.streams.keys()[0]

        for _ in range(2 * self.streams[stream].fps):
            self.get_frame([stream])


if __name__ == '__main__':
    camera = RealsenseCamera(
        Streams([
            Stream(
                name='color',
                stype=rs.stream.color,
                w_res=1920,
                h_res=1080,
                dtype=rs.format.bgr8,
                fps=30,
                extractor=lambda frames: frames.get_color_frame()
            )
        ])
    )

    try:
        camera.start_streaming()
        frame = camera.get_frame(['color'])[0]
    finally:
        camera.stop_streaming()

    plt.imshow(frame)
    plt.show()
