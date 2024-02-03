from dataclasses import dataclass
import os
from typing import Any, Dict, List, Tuple

from absl import app
from absl import flags
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from utils import system_usage_utils
from utils import torch_utils


@dataclass
class ProcessingTime:
    preprocess: float
    inference: float
    postprocess: float
    total: float


@dataclass
class FrameDetectionResult:
    """ Detection result of one frame. """
    boxes: np.ndarray  # np.array (k, 4) # (x, y, w, h). if nothing, (0, 4)
    masks: List[np.ndarray]  # List[np.array(t, 2)] # (x, y). if nothing, []
    period: ProcessingTime
    fps: ProcessingTime


# HYPERPARAMETERS
flags.DEFINE_integer("vid_stride", 1, "Stride number for input data.")
flags.DEFINE_float("confidence", 0.5, "Confidence threshold.")

# CHECK!!!!
flags.DEFINE_string("data_parent_dir", "data/", "Data parent directory.")
flags.DEFINE_string("data_name", None,
                    "if not set, all data in directory will be used.")

flags.DEFINE_boolean("save_result", True, "Save result.")
flags.DEFINE_boolean(
    "do_segmentation", False,
    "Do segmentation with detection. Detection is always performed.")
flags.DEFINE_boolean(
    "stream_buffer", False,
    "Buffer all streaming frames (True) or return the most recent frame (False)"
)
flags.DEFINE_boolean("show_labels", False, "Show labels.")
flags.DEFINE_boolean("show_conf", False, "Show confidences.")
flags.DEFINE_boolean("check_resource_usage", False, "Check resource usage.")
flags.DEFINE_float("monitoring_interval", 0.1, "Monitoring interval.")
flags.DEFINE_boolean("use_gpu", False, "Use GPU.")

flags.DEFINE_enum("resize_mode", "custom", ["original", "custom", "default"],
                  " Input image resize mode.")
flags.DEFINE_list(
    "input_resize_shapes", [(384, 640), (192, 320)],
    "Input image resize shapes. If not set, original size is used.")
flags.DEFINE_enum("model_size", "nano",
                  ["nano", "small", "medium", "large", "xlarge"], "Model size.")
flags.DEFINE_enum("input_type", "image", ["image", "video"],
                  "Choose a input type.")

FLAGS = flags.FLAGS


def _get_frame_detection_result(result: Any) -> FrameDetectionResult:
    # BOXES
    a_boxes = result.boxes.xywh.cpu().numpy()  # (k, 4) # (x, y, w, h)
    # MASKS
    a_masks = []
    if FLAGS.do_segmentation:
        if result.masks:
            a_masks = result.masks.xy  # List[np.array(t, 2)] # (x, y)
    # SPEED
    period_dict = result.speed
    total_period = sum(period_dict.values())
    period_dict["total"] = total_period
    fps_dict = {}
    for key, a_speed in period_dict.items():
        # milliseconds to seconds and round to 3 decimal places.
        period_dict[key] = round(a_speed / 1000, 6)
        fps_dict[key] = round(1 / period_dict[key], 3)
    preiod_info = ProcessingTime(**period_dict)
    fps_info = ProcessingTime(**fps_dict)
    detection_result = FrameDetectionResult(boxes=a_boxes,
                                            masks=a_masks,
                                            period=preiod_info,
                                            fps=fps_info)
    return detection_result


def run(model: YOLO, data_paths: list[str],
        input_resize_shape: tuple[int,
                                  int]) -> List[List[FrameDetectionResult]]:
    paths_detection_results: List[List[FrameDetectionResult]] = []
    for data_path in data_paths:

        # data_path: data/test_video/test_160x90.mp4
        # save_folder_name: test_160x90
        save_folder_name = os.path.basename(data_path).split(".")[0]
        # save_folder_name: test_160x90->(160, 90)
        save_folder_name = f"({save_folder_name})->{input_resize_shape}"
        # Remove space from save_folder_name.
        save_folder_name = save_folder_name.replace(" ", "")
        print("data_path:", data_path)
        print("save_folder_name:", save_folder_name)
        # results: List[Results]
        # len(results): Length of frames.
        results = model.predict(
            source=data_path,
            save=FLAGS.save_result,
            name=save_folder_name,
            conf=FLAGS.confidence,
            classes=[0],  # 0: person
            imgsz=input_resize_shape,
            vid_stride=FLAGS.vid_stride,
            stream_buffer=FLAGS.stream_buffer,
            show_labels=FLAGS.show_labels,
            show_conf=FLAGS.show_conf)  # device=device
        # pylint: disable=pointless-string-statement
        """
        device=device Option device works.
        BUT, speed is much faster at CPU than MPS.

        MPS
        Speed: 0.9ms preprocess, 13.7ms inference, 12.8ms postprocess per image
        at shape (1, 3, 192, 352)
        CPU
        Speed: 0.4ms preprocess, 16.8ms inference, 0.5ms postprocess per image
        at shape (1, 3, 192, 352)

        """

        a_path_detection_results: List[FrameDetectionResult] = []
        # results: results of a data_path.
        for result in results:  # result: Result of a frame in a data_path.
            if FLAGS.input_type == "video":
                a_path_detection_results.append(
                    _get_frame_detection_result(result))
        paths_detection_results.append(a_path_detection_results)
    return paths_detection_results


def _set_model() -> YOLO:
    algo_name = "yolov8"
    # Adapt model size into the algo_name.
    # "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"
    algo_name += FLAGS.model_size[0]
    if FLAGS.do_segmentation:
        algo_name += "-seg"
    algo_name += ".pt"
    model = YOLO(algo_name)
    print("algo_name:", algo_name)
    return model


def _set_data_paths() -> List[str]:
    data_dir = os.path.join(FLAGS.data_parent_dir, FLAGS.input_type)
    if FLAGS.data_name:
        data_paths = [os.path.join(data_dir, FLAGS.data_name)]
    else:
        if FLAGS.input_type == "video":
            end_str = (".mp4", ".MOV")
        else:
            end_str = (".jpg", ".png")
        data_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(end_str)
        ]
        data_paths.sort()
    return data_paths


def _get_total_frames_number(data_paths: List[str]) -> int:

    total_frames_number = 0

    for data_path in data_paths:
        if FLAGS.input_type == "video":
            # 비디오 파일의 경우 OpenCV를 사용하여 프레임 수를 구함
            cap = cv2.VideoCapture(data_path)
            total_frames_number += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            # 이미지 파일의 경우 1 프레임을 추가
            total_frames_number += 1

    return total_frames_number


def _get_image_resolution(data_paths: List[str]) -> Tuple[int, int]:
    # Get the size of the first frame of the first video.
    if FLAGS.input_type == "video":
        cap = cv2.VideoCapture(data_paths[0])
        if not cap.isOpened():
            raise ValueError("Could not open the video file")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (width, height)
    with Image.open(data_paths[0]) as img:
        return img.size  # (width, height) 형태의 튜플을 반환


def main(argv):
    del argv
    model = _set_model()
    data_paths = _set_data_paths()

    total_frames_number = _get_total_frames_number(data_paths)
    total_process_frames_number = total_frames_number // FLAGS.vid_stride

    device = torch_utils.get_device(use_gpu=FLAGS.use_gpu)
    # TODO: check it works on GPU and get better performance.
    if FLAGS.use_gpu:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    system_info = system_usage_utils.get_basic_system_info(device)

    if FLAGS.resize_mode == "custom":
        input_resize_shapes = FLAGS.input_resize_shapes
    elif FLAGS.resize_mode == "default":
        input_resize_shapes = None
        assert FLAGS.input_resize_shapes is None
    elif FLAGS.resize_mode == "original":
        input_resize_shapes = [_get_image_resolution(data_paths)]
        assert FLAGS.input_resize_shapes is None
    else:
        raise ValueError(f"Invalid resize_mode: {FLAGS.resize_mode}")

    resource_usage_data: Dict[str, Dict[str, Any]] = {}

    for input_resize_shape in input_resize_shapes:
        kwargs = {
            "model": model,
            "data_paths": data_paths,
            "input_resize_shape": input_resize_shape,
        }
        if FLAGS.check_resource_usage:
            a_usage_data: Dict[
                str, Any] = system_usage_utils.check_cpu_and_memory_usage(
                    function=run,
                    kwargs=kwargs,
                    monitoring_interval=FLAGS.monitoring_interval)
            resource_usage_data[
                f"[Resize]{str(input_resize_shape)}"] = a_usage_data
        else:
            paths_detection_results = run(**kwargs)  # pylint: disable=unused-variable
    if FLAGS.check_resource_usage:
        system_usage_utils.plot_multiple_usage_data(
            resource_usage_data,
            system_info,
            FLAGS.monitoring_interval,
            total_frames=total_process_frames_number)


if __name__ == '__main__':
    app.run(main)  # absl은 내부적으로 sys.argv를 파싱하여 정의된 플래그들을 처리
