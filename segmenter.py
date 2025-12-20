from typing import Iterator

import numpy as np
from PIL.Image import Image
from ultralytics import YOLO, YOLOE, SAM
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results


class Segmenter:
    engines: dict[str, Model] = {
        'yolo': YOLO,
        'yoloe': YOLOE,
        'sam': SAM
    }

    def __init__(self, engine: str, weights: str) -> None:
        engine = engine.lower().strip()

        if engine not in self.engines:
            raise NotImplementedError

        self.model: Model = self.engines[engine](weights)

    @property
    def engine_type(self) -> str:
        return self.model.__class__.__name__.lower()

    def segment_all(
            self,
            src: np.ndarray | list[Image],
            stream: bool = False
    ) -> list[Results] | Iterator[Results]:
        if self.engine_type in ('yolo', 'yoloe'):
            return self.model.predict(src, stream=stream)

        return self.model(src, stream=stream)
