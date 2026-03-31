"""
RealSense 3D camera interface module
"""

import time

import numpy as np
import pyrealsense2 as rs

from structures import Stream, Streams


class RealsenseCamera:
    """
    RealSense camera interface
    """

    def __init__(self, streams: Streams | list[Stream] | Stream) -> None:
        """
        Initialize the connected RealSense camera

        :param streams: Descriptors of the video streams to capture with this instance
        """

        # Convert all the possible input stream formats to a `Streams` instance
        if isinstance(streams, list):
            if len(streams) == 0:
                raise ValueError

            streams = Streams(streams)

        elif isinstance(streams, Stream):
            streams = Streams([streams])

        # Scan the connected devices
        self.streams: Streams = streams
        self._devices: rs.device_list = rs.context().query_devices()

        # Perform a hardware reset of the camera
        # to fix potential connectivity issues
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

        # Activate the specified streams
        for stream in self.streams.values():
            self.config.enable_stream(*stream.config)

        # Map the camera's internal stream profiles to the `Stream` instances
        for sprofile in self.config.resolve(self.pipeline).get_streams():
            sprofile: rs.video_stream_profile = sprofile.as_video_stream_profile()
            stream = self.streams[sprofile.stream_type(), sprofile.stream_index()]
            stream.profile = sprofile

        self.streaming: bool = False

    def __repr__(self) -> str:
        return f'<Camera {self.name} operating {len(self.streams)} stream{"s" if len(self.streams) > 1 else ""}>'

    def __del__(self) -> None:
        """
        Safely destroy the instance
        """

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
        """
        Camera's sensor attributes
        """

        return {
            s.get_info(rs.camera_info.name): s
            for s in self.device.sensors
        }

    @property
    def num_sensors(self) -> int:
        """
        Number of sensors / specific cameras
        """

        return len(self.sensors)

    @property
    def num_streams(self) -> int:
        """
        Number of broadcast video streams
        """

        return len(self.streams)

    def _reset_devices(self) -> None:
        """
        Perform a hardware reset of all the connected devices
        """

        for dev in self._devices:
            dev.hardware_reset()

    def start_streaming(self) -> None:
        """
        Reset the devices and start streaming
        """

        self._reset_devices()
        self.pipeline.start(self.config)
        self.streaming = True

    def stop_streaming(self) -> None:
        """
        Safely stop the streaming
        """

        self.pipeline.stop()
        self.streaming = False

    def get_composite_frame(self) -> rs.composite_frame | None:
        """
        Asynchronously get a raw composite frame of all the allowed streams,
        and geometrically align all the frames. If the camera is not streaming yet,
        the streaming will be started

        :return: Aligned composite frame of all the allowed streams
                 or `None` if the async capture failed in this call
        """

        if not self.streaming:
            self.start_streaming()

        success, composite = self.pipeline.try_wait_for_frames(timeout_ms=0)

        if success:
            return self.align.process(composite)

        return None

    def get_frame(self, stream_names: list[str]) -> list[np.ndarray] | None:
        """
        Get aligned frames from the specified streams

        :param stream_names: List of streams to get frames from
        :return: List of aligned frames following the order of `stream_names`
        """

        # Check if all the specified streams are valid and have been allowed
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
        """
        Perform a hardware warm-up to get the camera ready for streaming.
        The camera will be capturing and discarding frames for `t` seconds

        :param t: Number of seconds to capture frames for
        """

        if not self.streaming:
            self.start_streaming()

        stream = self.streams.keys()[0]
        t0 = time.time()

        while time.time() - t0 < t:
            self.get_frame([stream])
