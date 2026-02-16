"""
Data structures pertinent to the RealSense camera operation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, MutableMapping

import numpy as np
import pyrealsense2 as rs


@dataclass
class Intrinsics:
    """
    Intrinsic parameters of a RealSense camera
    """

    # Camera resolution
    width: int
    height: int

    # Focal lengths
    fx: float
    fy: float

    # Coordinates of the principal point
    ppx: float
    ppy: float

    # Distortion parameters
    coeffs: np.ndarray
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

    @property
    def K(self) -> np.ndarray:
        r"""
        Intrinsic parameters in the matrix representation:
        .. math::
            K = \begin{bmatrix}
                f_x &   0 & c_x \\
                  0 & f_y & c_y \\
                  0 &   0 &   1
            \end{bmatrix}
        """

        return np.array([
            [self.fx,     0.0, self.ppx],
            [    0.0, self.fy, self.ppy],
            [    0.0,     0.0,      1.0]
        ], dtype=np.float32)

    @classmethod
    def from_rs(cls, intrinsics: rs.intrinsics) -> Intrinsics:
        """
        Construct an `Intrinsics` object from a RealSense camera intrinsics
        object `pyrealsense2.intrinsics`

        :param intrinsics: RealSense camera intrinsics object
        """

        return cls(
            width=intrinsics.width, height=intrinsics.height,

            fx=intrinsics.fx, fy=intrinsics.fy,
            ppx=intrinsics.ppx, ppy=intrinsics.ppy,

            coeffs=np.array(intrinsics.coeffs, dtype=np.float32),
            dist_model=intrinsics.model
        )


class Stream:
    """
    RealSense camera stream descriptor
    """

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
        """
        Construct a stream descriptor

        :param name: Arbitrary name of the stream
        :param stype: Stream type as a `pyrealsense2.stream` object
        :param w_res: Horizontal resolution
        :param h_res: Vertical resolution
        :param dtype: Data type to be used as a `pyrealsense2.format` object
        :param fps: Frame rate
        :param extractor: Function extracting the `pyrealsense2.frame` representation
                          from the raw `pyrealsense2.composite_frame` object
        :param colorizer: Optional colorizer
        :param sid: Optional stream index in the case of multiple streams of the same stream type `stype`
                    available (e.g., RealSense stereoscopic cameras offer two streams for the
                    `pyrealsense2.stream.infrared` type: `sid = 0` denotes the left IR camera,
                    `sid = 1` the right one)
        :param roi: Optional region of interest to be cropped as a tuple `(lt_x, lt_y, w, h)`
        """

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
        self.intrinsics: Intrinsics | None = None

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

            # Check validity of the ROI parameters
            if not (
                    0 <= self.roi_l < self.w_res and 0 <= self.roi_t < self.h_res
                    and self.roi_w > 0 and self.roi_h > 0
            ):
                raise ValueError

    def __repr__(self) -> str:
        return f'<Stream "{self.name}" {self.w_res}x{self.h_res} @ {self.fps}fps {self.dtype.name.upper()}>'

    @property
    def config(self) -> tuple:
        """
        Stream configuration tuple
        """

        if self.sid > 0:
            return self.stype, self.sid, self.w_res, self.h_res, self.dtype, self.fps

        return self.stype, self.w_res, self.h_res, self.dtype, self.fps

    @property
    def resolution(self) -> tuple[int, int]:
        """
        WH resolution of the stream
        """

        return self.w_res, self.h_res

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        """
        Region of the interest parameter tuple `(lt_x, lt_y, w, h)` or `None` if not set
        """

        roi = (self.roi_l, self.roi_t, self.roi_w, self.roi_h)

        if None not in roi:
            return roi

        return None

    def extract(self, composite_frame: rs.composite_frame) -> rs.frame:
        """
        Extract the `pyrealsense2.frame` object
        from the raw `pyrealsense2.composite_frame` representation
        """

        return self.extractor(composite_frame)

    def to_numpy(self, frame: rs.composite_frame | rs.frame) -> np.ndarray:
        """
        Convert a `pyrealsense2` frame to NumPy array

        :param frame: Source frame
        :return: The frame as a NumPy array
        """

        if isinstance(frame, rs.composite_frame):
            frame = self.extract(frame)

        if self.colorizer is not None:
            frame = self.colorizer.colorize(frame)

        image = np.asanyarray(frame.get_data())

        # Crop if ROI is set
        if self.roi is not None:
            image = image[
                self.roi_t:self.roi_t + self.roi_h,
                self.roi_l:self.roi_l + self.roi_w
            ]

        # If the frame is single-channel, convert it to grayscale RGB
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
        """
        RealSense camera video stream profile associated with this stream
        """

        return self._profile

    @profile.setter
    def profile(self, profile: rs.video_stream_profile) -> None:
        """
        Associate this stream with a video stream of an initialized RealSense camera
        and get the corresponding intrinsic parameters
        """

        # Check if the stream profile matches this stream's parameters
        if (
            profile.stream_type() != self.stype
            or profile.format() != self.dtype
            or (profile.width(), profile.height()) != self.resolution
            or profile.fps() != self.fps
        ):
            raise ValueError

        self._profile = profile
        self.intrinsics = Intrinsics.from_rs(self.profile.get_intrinsics())


class Streams(MutableMapping):
    """
    Container grouping multiple streams
    """

    def __init__(self, streams: list[Stream]) -> None:
        """
        Construct the container

        :param streams: List of individual stream descriptors. The stream descriptors must have
                        unique names and cannot duplicitously refer to the same camera stream
        """

        self._streams: dict[str, Stream] = {}
        self._stype_sids: list[tuple[rs.stream, int]] = []

        for stream in streams:
            # Enforce unique stream names
            if stream.name in self:
                raise ValueError

            # Enforce unique camera streams
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
        """
        The lowest WH resolution among all streams
        """

        return min(
            [
                stream.resolution
                for stream in self._streams.values()
            ],
            key=lambda res: res[0] * res[1]
        )

    def values(self) -> list[Stream]:
        """
        Get the stream descriptors
        """

        return list(self._streams.values())

    def keys(self) -> list[str]:
        """
        Get the stream names
        """

        return list(self._streams.keys())
