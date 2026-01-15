import time
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pyrealsense2 as rs


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    coeffs: list[float]
    dist_model: rs.distortion

    def __repr__(self) -> str:
        return (
            f'<Intrinsics '
            f'{self.width}x{self.height} '
            f'p[{self.ppx:.3f} {self.ppy:.3f}] '
            f'f[{self.fx:.3f} {self.fy:.3f}] '
            f'{self._model_name} [{" ".join([f"{x:.2f}" for x in self.coeffs])}]'
            f'>'
        )

    @property
    def _model_name(self) -> str:
        return self.dist_model.name.replace('_', ' ').title()

    @classmethod
    def from_rs(cls, intrinsics: rs.intrinsics) -> 'Intrinsics':
        return cls(
            width=intrinsics.width, height=intrinsics.height,
            fx=intrinsics.fx, fy=intrinsics.fy,
            ppx=intrinsics.ppx, ppy=intrinsics.ppy,
            coeffs=intrinsics.coeffs, dist_model=intrinsics.model
        )


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
            sid: int = 0,
            roi: tuple[int, int, int, int] | None = None
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

        self._profile: rs.video_stream_profile | None = None

        self.roi_l: int | None = None
        self.roi_t: int | None = None
        self.roi_w: int | None = None
        self.roi_h: int | None = None

        if roi is not None:
            if len(roi) != 4:
                raise ValueError

            (
                self.roi_l, self.roi_t,
                self.roi_w, self.roi_h
            ) = roi

            if not (
                    0 <= self.roi_l < self.w_res and 0 <= self.roi_t < self.h_res
                    and self.roi_w > 0 and self.roi_h > 0
            ):
                raise ValueError

    def __repr__(self) -> str:
        return f'<Stream "{self.name}" {self.w_res}x{self.h_res} @ {self.fps}fps {self.dtype.name.upper()}>'

    @property
    def config(self) -> tuple:
        if self.sid > 0:
            return self.stype, self.sid, self.w_res, self.h_res, self.dtype, self.fps

        return self.stype, self.w_res, self.h_res, self.dtype, self.fps

    @property
    def resolution(self) -> tuple[int, int]:
        return self.w_res, self.h_res

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        roi = (self.roi_l, self.roi_t, self.roi_w, self.roi_h)

        if None not in roi:
            return roi

        return None

    def extract(self, composite_frame: rs.composite_frame) -> rs.frame:
        return self.extractor(composite_frame)

    def to_numpy(self, frame: rs.composite_frame | rs.frame) -> np.ndarray:
        if isinstance(frame, rs.composite_frame):
            frame = self.extract(frame)

        if self.colorizer is not None:
            frame = self.colorizer.colorize(frame)

        image = np.asanyarray(frame.get_data())

        if self.roi is not None:
            image = image[
                self.roi_t:self.roi_t + self.roi_h,
                self.roi_l:self.roi_l + self.roi_w
            ]

        if image.ndim == 2:
            return np.repeat(
                image[:, :, np.newaxis],
                repeats=3,
                axis=2
            )

        if image.ndim == 3:
            return image

        raise ValueError

    @property
    def profile(self) -> rs.video_stream_profile | None:
        return self._profile

    @profile.setter
    def profile(self, profile: rs.video_stream_profile) -> None:
        if (
            profile.stream_type() != self.stype
            or profile.format() != self.dtype
            or (profile.width(), profile.height()) != self.resolution
            or profile.fps() != self.fps
        ):
            raise ValueError

        self._profile = profile

    @property
    def intrinsics(self) -> Intrinsics | None:
        if self.profile is None:
            return None

        return Intrinsics.from_rs(self.profile.get_intrinsics())


class Streams(MutableMapping):
    def __init__(self, streams: list[Stream]) -> None:
        self._streams: dict[str, Stream] = {}
        self._stype_sids: list[tuple[rs.stream, int]] = []

        for stream in streams:
            if stream.name in self:
                raise ValueError

            if stream.stype in self._stype_sids:
                if stream.sid <= 0:
                    raise ValueError

                if (stream.stype, stream.sid) in self._stype_sids:
                    raise ValueError

            self[stream.name] = stream
            self._stype_sids.append((stream.stype, stream.sid))

    def __repr__(self) -> str:
        return f'<Streams {", ".join(self.keys())}>'

    def __getitem__(self, key: str | rs.stream | tuple[rs.stream, int]) -> Stream:
        if isinstance(key, str):
            return self._streams[key]

        if isinstance(key, rs.stream):
            for stream, (stype, sid) in zip(self.values(), self._stype_sids):
                if stype == key and sid <= 0:
                    return stream

            raise ValueError

        stream_id = self._stype_sids.index(key)
        return self.values()[stream_id]

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

    def values(self) -> list[Stream]:
        return list(self._streams.values())

    def keys(self) -> list[str]:
        return list(self._streams.keys())


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

    def warmup(self, t: float = 2.0) -> None:
        if not self.streaming:
            self.start_streaming()

        stream = self.streams.keys()[0]
        t0 = time.time()

        while time.time() - t0 < t:
            self.get_frame([stream])
