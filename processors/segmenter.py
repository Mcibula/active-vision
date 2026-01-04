import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import cv2
import numpy as np
import torch
from PIL.Image import Image
from torch import Tensor
from ultralytics import SAM, YOLO, YOLOE
from ultralytics.engine.results import Results
from ultralytics.utils.checks import check_imgsz

from utils.image import crop_zeros, resize_imgs

if TYPE_CHECKING:
    from ultralytics.engine.model import Model

logging.getLogger('ultralytics').setLevel(logging.ERROR)


@dataclass
class TrackRecord:
    xyxy: list[tuple[float, float, float, float]]
    masks: list[np.ndarray]
    snapshots: list[np.ndarray]


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
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._results: list[Results] = []

    @property
    def engine_type(self) -> str:
        return self.model.__class__.__name__.lower()

    @property
    def gpu(self) -> bool:
        return self.device == 'cuda'

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
                        snapshots=[]
                    )

                record: TrackRecord = objects[obj_id]
                record.xyxy.append(xyxy[idx])

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
