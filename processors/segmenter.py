from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import cv2
import numpy as np
from PIL.Image import Image
from ultralytics import SAM, YOLO, YOLOE
from ultralytics.engine.results import Results

from utils.image import crop_zeros, resize_img

if TYPE_CHECKING:
    from ultralytics.engine.model import Model


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

        self._results: list[Results] = []

    @property
    def engine_type(self) -> str:
        return self.model.__class__.__name__.lower()

    def segment_all(
            self,
            src: np.ndarray | list[Image],
            stream: bool = False
    ) -> list[Results] | Iterator[Results]:
        if self.engine_type in ('yolo', 'yoloe'):
            return self.model.predict(src, stream=stream, verbose=False)

        return self.model(src, stream=stream, verbose=False)

    @staticmethod
    def _segment_object(src: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if (
                src.ndim < 2
                or mask.ndim != 2
                or mask.dtype != np.uint8
                or mask.min() != 0 or mask.max() != 1
        ):
            raise ValueError

        upscaled_mask = resize_img(mask, src.shape)
        masked_img = src * upscaled_mask

        return crop_zeros(masked_img)

    def track(self, src: np.ndarray | list[np.ndarray]) -> dict[int, TrackRecord]:
        if self.engine_type == 'sam':
            raise NotImplementedError

        if not isinstance(src, list):
            src = [src]

        self._results: list[Results] = self.model.track(
            src,
            stream=False,
            persist=True,
            verbose=False
        )

        objects: dict[int, TrackRecord] = {}
        for r, src_img in zip(self._results, src):
            obj_ids: list[float] = r.boxes.id.cpu().tolist()
            xyxy: np.ndarray = r.boxes.xyxy.cpu().numpy()
            masks: np.ndarray = r.masks.data.cpu().numpy()

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

                cropped_mask = crop_zeros(resize_img(masks[idx], src_img.shape))
                record.masks.append(cropped_mask)

                record.snapshots.append(self._segment_object(src_img, masks[idx]))

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

