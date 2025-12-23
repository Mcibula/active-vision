"""
Pre-defined RealSense camera profiles
"""

import pyrealsense2 as rs

from realsense import RealsenseCamera, Stream, Streams

FHD_RGB = RealsenseCamera(
    Streams([
        Stream(
            name='color',
            stype=rs.stream.color,
            w_res=1920,
            h_res=1080,
            dtype=rs.format.rgb8,
            fps=30,
            extractor=lambda frames: frames.get_color_frame()
        )
    ])
)

HD_DEPTH = RealsenseCamera(
    Streams([
        Stream(
            name='depth',
            stype=rs.stream.depth,
            w_res=1280,
            h_res=720,
            dtype=rs.format.z16,
            fps=30,
            extractor=lambda frames: frames.get_depth_frame(),
            colorizer=rs.colorizer(color_scheme=0)
        )
    ])
)
