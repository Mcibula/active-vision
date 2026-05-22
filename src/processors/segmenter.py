from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator

import cv2
import numpy as np
import torch
from torch import Tensor
from ultralytics import SAM, YOLO, YOLOE
from ultralytics.utils.checks import check_imgsz

from structures import BBox, TrackRecord
from utils.image import resize_imgs
from utils.logger import PerformanceMonitor, get_logger, timer
from utils.misc import infer_device

if TYPE_CHECKING:
    from PIL.Image import Image
    from ultralytics.engine.model import Model
    from ultralytics.engine.results import Results

logging.getLogger('ultralytics').setLevel(logging.ERROR)


class Segmenter:
    def __init__(self, engine: str, weights: str) -> None:
        self.engines: dict[str, type[YOLO | YOLOE | SAM]] = {
            'yolo': YOLO,
            'yoloe': YOLOE,
            'sam': SAM
        }

        engine = engine.lower().strip()

        if engine not in self.engines:
            raise NotImplementedError

        self.tracking = engine != 'sam'
        self.model: Model = self.engines[engine](weights)
        self.device: torch.device = infer_device()

        self._results: list[Results] = []

        self.logger: logging.Logger = get_logger('Segmenter', level=logging.INFO)
        self.monitor: PerformanceMonitor = PerformanceMonitor()
        self.logger.info('Segmenter initialized')

    @property
    def engine_type(self) -> str:
        return self.model.__class__.__name__.lower()

    @property
    def gpu(self) -> bool:
        return self.device.type == 'cuda'

    def segment_all(
            self,
            src: np.ndarray | list[Image],
            stream: bool = False
    ) -> list[Results] | Iterator[Results]:
        if self.engine_type in ('yolo', 'yoloe'):
            return self.model.predict(src, stream=stream, verbose=False)

        return self.model(src, stream=stream, verbose=False)

    def track(self, src: np.ndarray | list[np.ndarray]) -> dict[int, TrackRecord]:
        if self.engine_type == 'sam':
            raise NotImplementedError

        if not isinstance(src, list):
            src: list[np.ndarray] = [src]

        frame_h, frame_w, frame_c = src[0].shape
        correct_h, correct_w = check_imgsz(
            imgsz=[frame_h, frame_w],
            stride=32
        )

        scale = min(correct_h / frame_h, correct_w / frame_w)
        unpad_h = round(frame_h * scale)
        unpad_w = round(frame_w * scale)
        pad_h = correct_h - unpad_h
        pad_w = correct_w - unpad_w
        pad_top = round(pad_h / 2 - 0.1)
        pad_left = round(pad_w / 2 - 0.1)

        with torch.no_grad():
            with timer('Segmenter.track.preprocess', self.logger, self.monitor):
                try:
                    # Convert uint8 RGB NHWC to float32 RGB NCHW
                    frames: Tensor = (
                            torch.from_numpy(np.asarray(src))
                                 .permute(0, 3, 1, 2)
                                 .float()
                                 .to(self.device) / 255
                    )
                except ValueError as e:
                    raise ValueError('All the source frames must have the same shape.') from e

                model_in = (
                    frames
                    if correct_h == frame_h and correct_w == frame_w
                    else resize_imgs(
                        src=frames,
                        to_shape=(correct_h, correct_w),
                        dim_order='nchw',
                        stretch=False,
                        padding_value=0
                    )
                )

            with timer('Segmenter.track.inference', self.logger, self.monitor):
                self._results: list[Results] = self.model.track(
                    model_in,
                    stream=False,
                    persist=True,
                    verbose=False
                )

        with timer('Segmenter.track.postprocess', self.logger, self.monitor):
            objects: dict[int, TrackRecord] = {}
            for frame_idx, r in enumerate(self._results):
                if r.boxes.id is None:
                    continue

                if r.masks is None:
                    continue

                obj_ids: list[float] = r.boxes.id.cpu().tolist()
                n_objs: int = len(obj_ids)

                xyxy: np.ndarray = r.boxes.xyxy.cpu().numpy()

                if n_objs == 0:
                    continue

                # CHW frame, O1HW masks
                frame: Tensor = frames[frame_idx] * 255
                masks: Tensor = r.masks.data

                if masks.shape[-2:] == (correct_h, correct_w):
                    masks = masks[
                        :,
                        pad_top:pad_top + unpad_h,
                        pad_left:pad_left + unpad_w
                    ]

                upscaled_masks_ch: Tensor = resize_imgs(
                    src=masks.unsqueeze(1),
                    to_shape=(frame_h, frame_w),
                    dim_order='nchw',
                    stretch=True
                )
                upscaled_masks: Tensor = upscaled_masks_ch.squeeze(1)

                # OCHW masked frames, OHW masks
                masked_frames: Tensor = upscaled_masks_ch * frame.expand(n_objs, -1, -1, -1)

                for idx, obj_id in enumerate(obj_ids):
                    obj_id: int = int(obj_id)

                    x1, y1, x2, y2 = xyxy[idx]
                    if (correct_h, correct_w) != (frame_h, frame_w):
                        x1 = (x1 - pad_left) / scale
                        x2 = (x2 - pad_left) / scale
                        y1 = (y1 - pad_top) / scale
                        y2 = (y2 - pad_top) / scale

                    x1 = max(0, min(frame_w, int(np.floor(x1))))
                    y1 = max(0, min(frame_h, int(np.floor(y1))))
                    x2 = max(0, min(frame_w, int(np.ceil(x2))))
                    y2 = max(0, min(frame_h, int(np.ceil(y2))))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # BBox-aligned crops. The local origin of the RGB crop, mask, and depth crop
                    # is always the bbox left-top corner.
                    cropped_mask = upscaled_masks[idx, y1:y2, x1:x2]
                    cropped_snapshot = masked_frames[idx].permute(1, 2, 0)[y1:y2, x1:x2]

                    # Blank frame or blank mask
                    if cropped_mask.numel() == 0 or torch.count_nonzero(cropped_mask) == 0:
                        continue

                    if obj_id not in objects:
                        objects[obj_id] = TrackRecord(
                            xyxy=[],
                            masks=[],
                            snapshots=[],
                            frame_ids=[]
                        )

                    record: TrackRecord = objects[obj_id]
                    record.xyxy.append(BBox(x1, y1, x2, y2))
                    record.frame_ids.append(frame_idx)
                    record.masks.append((cropped_mask > 0.5).cpu().numpy().astype(np.uint8))
                    record.snapshots.append(cropped_snapshot.cpu().numpy().astype(np.uint8))

        return objects

    @timer('Segmenter.last_annotated_frame')
    def last_annotated_frame(
            self,
            conf: bool = False,
            labels: bool = False,
            boxes: bool = False,
            masks: bool = True,
            probs: bool = False
    ) -> np.ndarray | None:
        if not self._results:
            return None

        return cv2.cvtColor(
            self._results[-1].plot(
                conf=conf,
                labels=labels,
                boxes=boxes,
                masks=masks,
                probs=probs,
                show=False,
                pil=False,
                color_mode='instance'
            ),
            cv2.COLOR_BGR2RGB
        )
