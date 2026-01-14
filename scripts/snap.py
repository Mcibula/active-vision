import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

from camera import RealsenseCamera, Stream, Streams

camera = RealsenseCamera(
    Streams([
        Stream(
            name='color',
            stype=rs.stream.color,
            w_res=1920,
            h_res=1080,
            dtype=rs.format.rgb8,
            fps=30,
            extractor=lambda frames: frames.get_color_frame(),
            roi=(504, 0, 1038, 1080)
        ),
        Stream(
            name='depth',
            stype=rs.stream.depth,
            w_res=1280,
            h_res=720,
            dtype=rs.format.z16,
            fps=30,
            extractor=lambda frames: frames.get_depth_frame(),
            # colorizer=rs.colorizer(color_scheme=0),
            roi=(411, 106, 478, 491)
        )
    ])
)

try:
    camera.start_streaming()
    camera.warmup(t=4.0)

    frame: np.ndarray | None = None
    while frame is None:
        frame = camera.get_frame(['color', 'depth'])

    rgb_frame, depth_frame = frame

finally:
    camera.stop_streaming()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
ax1.imshow(rgb_frame)
ax2.imshow(cv2.normalize(depth_frame.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX))

fig.tight_layout()
plt.show()

# np.save('../experiments/test_imgs/rgb_frame.npy', rgb_frame)
# np.save('../experiments/test_imgs/depth_frame.npy', depth_frame)
