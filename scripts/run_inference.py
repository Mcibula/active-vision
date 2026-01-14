from camera import FHD_RGB, RealsenseCamera
from core import PipelineController, Scene
from processors import Segmenter


def main() -> Scene:
    camera = RealsenseCamera(FHD_RGB)
    segmenter = Segmenter(
        engine='yoloe',
        weights='../models/yolo/yoloe-11l-seg-pf.pt'
    )
    scene = Scene(segmenter=segmenter)

    controller = PipelineController(
        camera=camera,
        scene=scene,
        batch_size=1,
        batch_timeout=0.1,
        process_every=5,
        capture_limit=500
    )
    controller.run()

    return scene


if __name__ == '__main__':
    out_scene = main()
