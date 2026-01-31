import time

import numpy as np
import pyrealsense2 as rs

from structures import Stream, Streams


class RealsenseCamera:
    def __init__(self, streams: Streams | list[Stream] | Stream) -> None:
        if isinstance(streams, list):
            if len(streams) == 0:
                raise ValueError

            streams = Streams(streams)

        elif isinstance(streams, Stream):
            streams = Streams([streams])

        self.streams: Streams = streams
        self._devices: rs.device_list = rs.context().query_devices()

        self._reset_devices()

        self.pipeline: rs.pipeline = rs.pipeline()
        self.config: rs.config = rs.config()
        self.align: rs.align = rs.align(rs.stream.color)

        pipeline_profile = self.config.resolve(self.pipeline)
        self.device: rs.device = pipeline_profile.get_device()
        self.name: str = self.device.get_info(rs.camera_info.name)

        print(
            f'Initializing {self.name} '
            f'with {self.num_sensors} sensor{"s" if self.num_sensors > 1 else ""}: '
            f'{", ".join(self.sensors)}...'
        )

        for stream in self.streams.values():
            self.config.enable_stream(*stream.config)

        for sprofile in self.config.resolve(self.pipeline).get_streams():
            sprofile: rs.video_stream_profile = sprofile.as_video_stream_profile()
            stream = self.streams[sprofile.stream_type(), sprofile.stream_index()]
            stream.profile = sprofile

        self.streaming: bool = False

    def __repr__(self) -> str:
        return f'<Camera {self.name} operating {len(self.streams)} stream{"s" if len(self.streams) > 1 else ""}>'

    def __del__(self) -> None:
        if self.streaming:
            self.stop_streaming()

    def __contains__(self, item: str | Stream) -> bool:
        if isinstance(item, str):
            return item in self.streams

        return item in self.streams.values()

    def __getitem__(self, key: str) -> Stream:
        return self.streams[key]

    @property
    def sensors(self) -> dict[str, rs.sensor]:
        return {
            s.get_info(rs.camera_info.name): s
            for s in self.device.sensors
        }

    @property
    def num_sensors(self) -> int:
        return len(self.sensors)

    @property
    def num_streams(self) -> int:
        return len(self.streams)

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
            return self.align.process(composite)

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

    def warmup(self, t: float = 2.0) -> None:
        if not self.streaming:
            self.start_streaming()

        stream = self.streams.keys()[0]
        t0 = time.time()

        while time.time() - t0 < t:
            self.get_frame([stream])
