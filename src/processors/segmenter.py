from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import cv2
import numpy as np
import torch
from torch import Tensor
from ultralytics import SAM, YOLO, YOLOE
from ultralytics.utils.checks import check_imgsz

from utils.image import crop_zeros, resize_imgs
from utils.misc import infer_device

if TYPE_CHECKING:
    from PIL.Image import Image
    from ultralytics.engine.model import Model
    from ultralytics.engine.results import Results

logging.getLogger('ultralytics').setLevel(logging.ERROR)


@dataclass
class TrackRecord:
    xyxy: list[tuple[int, int, int, int]]
    masks: list[np.ndarray]
    snapshots: list[np.ndarray]
    frame_ids: list[int]


class BBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            raise ValueError

        if x1 > x2 or y1 > y2:
            raise ValueError

        self._x1: int = int(x1)
        self._y1: int = int(y1)
        self._x2: int = int(x2)
        self._y2: int = int(y2)

    def __repr__(self) -> str:
        return f'<BBox {self.w}x{self.h} @ {self.lu}, {self.br}>'

    def __iter__(self) -> Iterator[int]:
        yield from self.xyxy

    def __eq__(self, other: BBox) -> bool:
        return (
            self.x1 == other.x1 and self.y1 == other.y1
            and self.x2 == self.x2 and self.y2 == other.y2
        )

    def __hash__(self) -> int:
        return hash(self.xyxy)

    def __and__(self, other: BBox) -> BBox:
        return self.intersection(other)

    def __or__(self, other: BBox) -> BBox:
        return self.union(other)

    @property
    def x1(self) -> int:
        return self._x1

    @property
    def y1(self) -> int:
        return self._y1

    @property
    def x2(self) -> int:
        return self._x2

    @property
    def y2(self) -> int:
        return self._y2

    @property
    def lu(self) -> tuple[int, int]:
        return self.x1, self.y1

    @property
    def br(self) -> tuple[int, int]:
        return self.x2, self.y2

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.w, self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def shape(self) -> tuple[int, int]:
        return self.h, self.w

    def union(self, other: BBox) -> BBox:
        return BBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2)
        )

    def intersection(self, other: BBox) -> BBox | None:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        if ix1 < ix2 and iy1 < iy2:
            return BBox(ix1, iy1, ix2, iy2)

        return None

    def iou(self, other: BBox) -> float:
        inter = self & other

        if inter is None:
            return 0.0

        union_area = self.area + other.area - inter.area
        if union_area == 0:
            return 0.0

        return inter.area / union_area


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

        with torch.no_grad():
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

            correct_h, correct_w = check_imgsz(
                imgsz=[frame_h, frame_w],
                stride=32
            )
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

            self._results: list[Results] = self.model.track(
                model_in,
                stream=False,
                persist=True,
                verbose=False
            )

        objects: dict[int, TrackRecord] = {}
        for frame_idx, r in enumerate(self._results):
            if r.boxes.id is None:
                continue

            obj_ids: list[float] = r.boxes.id.cpu().tolist()
            n_objs: int = len(obj_ids)

            xyxy: np.ndarray = r.boxes.xyxy.cpu().numpy()

            if n_objs == 0:
                continue

            # CHW frame, O1HW masks
            frame: Tensor = frames[frame_idx] * 255
            upscaled_masks: Tensor = resize_imgs(
                src=r.masks.data.unsqueeze(1),
                to_shape=(frame_h, frame_w),
                dim_order='nchw',
                stretch=True
            )

            # OCHW masked frames, OHW masks
            masked_frames: Tensor = upscaled_masks * frame.expand(n_objs, -1, -1, -1)
            upscaled_masks: Tensor = upscaled_masks.squeeze(1)

            for idx, obj_id in enumerate(obj_ids):
                obj_id: int = int(obj_id)

                if obj_id not in objects:
                    objects[obj_id] = TrackRecord(
                        xyxy=[],
                        masks=[],
                        snapshots=[],
                        frame_ids=[]
                    )

                record: TrackRecord = objects[obj_id]

                x1, y1, x2, y2 = xyxy[idx]
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(correct_w, int(x2))
                y2 = min(correct_h, int(y2))
                record.xyxy.append((x1, y1, x2, y2))
                record.frame_ids.append(frame_idx)

                # HW mask, HWC snapshot
                cropped_mask = crop_zeros(upscaled_masks[idx])
                cropped_snapshot = crop_zeros(masked_frames[idx].permute(1, 2, 0))

                # Blank frame or blank mask
                if cropped_snapshot is None or cropped_mask is None:
                    continue

                record.masks.append(cropped_mask.cpu().numpy().astype(np.uint8))
                record.snapshots.append(cropped_snapshot.cpu().numpy().astype(np.uint8))

        return objects

    def last_annotated_frames(
            self,
            conf: bool = False,
            labels: bool = False,
            boxes: bool = False,
            masks: bool = True,
            probs: bool = False
    ) -> list[np.ndarray]:
        return [
            cv2.cvtColor(
                r.plot(
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
            for r in self._results
        ]
