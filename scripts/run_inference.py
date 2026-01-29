from camera import RealsenseCamera
from camera.profiles import FHD_RGB, HD_DEPTH_ALIGN
from core import PipelineController, Scene
from processors import PoseEstimator, Segmenter


def main() -> Scene:
    camera = RealsenseCamera([FHD_RGB, HD_DEPTH_ALIGN])

    segmenter = Segmenter(
        engine='yoloe',
        weights='../models/yolo/yoloe-11l-seg-pf.pt'
    )
    pose_estimator = PoseEstimator(
        camera_intrinsics=camera['color'].intrinsics,
        detector_weights='L-C4-v2',
        descriptor_weights='G-upright',
        score_thresh=0.5,
        match_thresh=6,
        estimation_rng=(0.1, 5.0)
    )

    scene = Scene(
        segmenter=segmenter,
        pose_estimator=pose_estimator
    )

    batch_size: int = 2
    controller = PipelineController(
        camera=camera,
        scene=scene,
        batch_size=batch_size,
        batch_timeout=0.1 * batch_size,
        process_every=1,
        capture_limit=100
    )
    controller.run()

    return scene


if __name__ == '__main__':
    out_scene = main()
