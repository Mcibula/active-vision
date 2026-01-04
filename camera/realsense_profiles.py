"""
Pre-defined RealSense camera profiles
"""

import pyrealsense2 as rs

from .realsense import Stream

FHD_RGB = Stream(
    name='color',
    stype=rs.stream.color,
    w_res=1920,
    h_res=1080,
    dtype=rs.format.rgb8,
    fps=30,
    extractor=lambda frames: frames.get_color_frame()
)

HD_DEPTH = Stream(
    name='depth',
    stype=rs.stream.depth,
    w_res=1280,
    h_res=720,
    dtype=rs.format.z16,
    fps=30,
    extractor=lambda frames: frames.get_depth_frame(),
    colorizer=rs.colorizer(color_scheme=0)
)

HD_IR1 = Stream(
    name='ir1',
    stype=rs.stream.infrared,
    sid=1,
    w_res=1280,
    h_res=720,
    dtype=rs.format.y8,
    fps=30,
    extractor=lambda frames: frames.get_infrared_frame(1),
    colorizer=rs.colorizer(color_scheme=2)
)

HD_IR2 = Stream(
    name='ir2',
    stype=rs.stream.infrared,
    sid=2,
    w_res=1280,
    h_res=720,
    dtype=rs.format.y8,
    fps=30,
    extractor=lambda frames: frames.get_infrared_frame(2),
    colorizer=rs.colorizer(color_scheme=2)
)
