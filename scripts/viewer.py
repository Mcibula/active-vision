import time

import cv2
import numpy as np

from camera import RealsenseCamera
from camera.profiles import FHD_RGB, HD_DEPTH, HD_IR1, HD_IR2


class RealsenseViewer:
    def __init__(self, output_prefix: str | None = None) -> None:
        self.camera = RealsenseCamera([FHD_RGB, HD_DEPTH, HD_IR1, HD_IR2])

        self.window_name = 'RealSense Viewer'
        self.win_w = 1280
        self.win_h = 720
        self.is_running = True

        self.stream_idx = 0
        self.fps = 0
        self.prev_time = time.time()
        self.frame_count = 0

        self.output_prefix: str = output_prefix
        self.writers: dict[str, cv2.VideoWriter] = {}

    @staticmethod
    def _process_depth(frame: np.ndarray) -> np.ndarray:
        depth_clipped = np.where((frame > 3000) | (frame <= 0), 0, frame)
        depth_norm = cv2.normalize(
            src=depth_clipped, dst=None,
            alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    @staticmethod
    def _letterbox(frame: np.ndarray, to_shape: tuple[int, int]) -> np.ndarray:
        h, w = frame.shape[:2]
        new_w, new_h = to_shape
        scale = min(new_h / h, new_w / w)

        nw = int(w * scale)
        nh = int(h * scale)
        w_off = (new_w - nw) // 2
        h_off = (new_h - nh) // 2

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return np.pad(resized, pad_width=((h_off,), (w_off,), (0,)))

    def _catch_blank(self, frame: np.ndarray | None, is_rgb: bool = False) -> np.ndarray:
        if frame is None:
            if is_rgb:
                return np.zeros_like(self.camera.streams['color'].resolution, dtype=np.uint8)

            return np.zeros(self.camera.streams['depth'].resolution, dtype=np.uint16)

        return frame

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        on_trackbar = lambda _: _
        cv2.createTrackbar('Stream', self.window_name, 0, 3, on_trackbar)
        cv2.createTrackbar('Crop Left', self.window_name, 0, 900, on_trackbar)
        cv2.createTrackbar('Crop Right', self.window_name, 0, 900, on_trackbar)
        cv2.createTrackbar('Crop Top', self.window_name, 0, 500, on_trackbar)
        cv2.createTrackbar('Crop Bottom', self.window_name, 0, 500, on_trackbar)

        try:
            self.camera.start_streaming()
            self.camera.warmup()
            print('Stream started')

            while self.is_running:
                frames = self.camera.get_frame(['color', 'depth', 'ir1', 'ir2'])

                if not frames:
                    time.sleep(0.01)
                    continue

                color = frames[0]
                depth = frames[1] if len(frames) > 1 else None
                ir1 = frames[2] if len(frames) > 2 else None
                ir2 = frames[3] if len(frames) > 3 else None

                if self.output_prefix is not None:
                    rec_color = cv2.cvtColor(self._catch_blank(color, is_rgb=True), cv2.COLOR_RGB2BGR)
                    rec_depth = self._process_depth(self._catch_blank(depth))

                    ir1_safe = self._catch_blank(ir1).astype(np.uint8)
                    ir2_safe = self._catch_blank(ir2).astype(np.uint8)

                    rec_dict = {
                        'color': rec_color,
                        'depth': rec_depth,
                        'ir1': ir1_safe,
                        'ir2': ir2_safe
                    }

                    if not self.writers:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                        for name, f in rec_dict.items():
                            h, w = f.shape[:2]
                            fname = f'{self.output_prefix}_{name}.mp4'

                            self.writers[name] = cv2.VideoWriter(
                                filename=fname,
                                fourcc=fourcc,
                                fps=30.0,
                                frameSize=(w, h)
                            )
                            print(f'Recording started: {fname} ({w}x{h} @ 30 FPS)')

                    for name, f in rec_dict.items():
                        self.writers[name].write(f)

                streams: dict[int, tuple[np.ndarray, str]] = {
                    0: (color, 'Color (RGB)'),
                    1: (
                        self._process_depth(self._catch_blank(depth)),
                        'Depth (Jet)'
                    ),
                    2: (
                        self._catch_blank(ir1).astype(np.uint8),
                        'IR 1 (Left)'
                    ),
                    3: (
                        self._catch_blank(ir2).astype(np.uint8),
                        'IR 2 (Right)'
                    )
                }

                self.stream_idx = cv2.getTrackbarPos('Stream', self.window_name)
                display_img, stream_name = streams[self.stream_idx]

                if display_img is None:
                    display_img = np.zeros((720, 1280), dtype=np.uint8)

                h, w = display_img.shape[:2]

                cl = cv2.getTrackbarPos('Crop Left', self.window_name)
                cr = cv2.getTrackbarPos('Crop Right', self.window_name)
                ct = cv2.getTrackbarPos('Crop Top', self.window_name)
                cb = cv2.getTrackbarPos('Crop Bottom', self.window_name)

                cl = min(w - 10, cl)
                cr = min(w - cl - 10, cr)
                ct = min(h - 10, ct)
                cb = min(h - ct - 10, cb)
                cropped_img = display_img[ct: h - cb, cl: w - cr].copy()

                view_img = (
                    cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                    if self.stream_idx == 0
                    else cropped_img
                )

                if view_img.ndim == 2:
                    view_img = cv2.cvtColor(view_img, cv2.COLOR_GRAY2BGR)

                view_img = self._letterbox(view_img, (self.win_w, self.win_h))

                sh, sw = view_img.shape[:2]
                cv2.putText(
                    img=view_img, text=f'{stream_name} | Raw: {w}x{h}', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2,
                    color=(0, 255, 0)
                )
                cv2.putText(
                    img=view_img, text=f'Crop: {sw}x{sh}', org=(10, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2,
                    color=(0, 255, 255)
                )
                cv2.putText(
                    img=view_img, text=f'FPS: {self.fps:.1f}', org=(10, 90),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2,
                    color=(0, 255, 0)
                )

                cv2.imshow(self.window_name, view_img)

                self.frame_count += 1
                if time.time() - self.prev_time > 1.0:
                    self.fps = self.frame_count / (time.time() - self.prev_time)
                    self.frame_count = 0
                    self.prev_time = time.time()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                elif key == ord('p'):
                    print(f'[{stream_name}] Crop: L={cl}, R={cr}, T={ct}, B={cb}')

        finally:
            print('Cleaning up...')

            for name, writer in self.writers.items():
                writer.release()
                print(f'Saved {name} stream to disk')

            self.camera.stop_streaming()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    viewer = RealsenseViewer(output_prefix='session_1')
    viewer.run()
