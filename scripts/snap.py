import cv2
import matplotlib.pyplot as plt
import numpy as np

from camera import RealsenseCamera
from camera.profiles import FHD_RGB, HD_DEPTH_ALIGN

camera = RealsenseCamera([FHD_RGB, HD_DEPTH_ALIGN])

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
