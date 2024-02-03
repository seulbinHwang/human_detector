"""
프로파일러 설정:
    profile 함수를 사용하여, 프로파일링할 활동(CPU, CUDA 등)과 다른 옵션들을 설정
    record_shapes=True
        - 텐서 형태를 기록하여 메모리 사용량 분석에 도움을 줌.
프로파일링 대상 지정:
    with record_function("model_predict"):
        블록 내에서 model.predict 호출을 통해 추론을 실행
        이렇게 하면 model.predict 함수 실행 시간과 리소스 사용량이 기록
프로파일링 결과 출력:
    프로파일링 세션이 끝난 후, prof.key_averages().table(...)을 호출하여
    중요한 통계와 함께 프로파일링 결과를 테이블 형태로 출력
    sort_by 옵션을 통해 결과를 정렬할 기준을 지정할 수 있음

이 코드는 model.predict 함수의 성능을 분석하기 위해 PyTorch Profiler를 사용하는 방법
torch.cuda.is_available()을 통해 CUDA 사용 가능 여부를 확인하고,
GPU를 사용할 수 있는 경우 ProfilerActivity.CUDA를 활동에 추가하여 GPU 사용량도 함께 프로파일링
"""
from collections import defaultdict
import os
import sys
from typing import Any, Dict, List, Tuple

from absl import app
from absl import flags
import cv2
import numpy as np
from PIL import Image
import test2
import torch
from ultralytics import YOLO


def parse_tuple(flag_value):
    try:
        # 입력값을 쉼표로 분리하여 튜플로 변환
        return tuple(map(str.strip, flag_value.split(',')))
    except ValueError as e:
        raise flags.ValidationError(str(e))


flags.DEFINE(
    parser=parse_tuple,
    name="resize_shape",
    default=(384, 640),  # (192, 320)
    help="Example of a custom tuple flag")

flags.DEFINE_string("name", None, "Your name")
flags.DEFINE_integer("vid_stride", 3, "Stride number for input data.")
flags.DEFINE_boolean("debug", False, "Produces debugging output")
flags.DEFINE_enum('mode', 'blue', ['detection', 'segmentation', 'blue'],
                  'Choose a color')
flags.DEFINE_boolean("do_segmentation", False, "Do sementation with detection.")
flags.DEFINE_boolean("do_tracking", False, "Do tracking.")
flags.DEFINE_boolean("check_resource_usage", True, "Check resource usage.")
flags.DEFINE_boolean("use_gpu", False, "Use GPU.")
flags.DEFINE_enum("resize_mode", "custom", ["original", "custom", "default"],
                  " Input image resize mode.")

flags.DEFINE_boolean("draw_trajectory", False, "Draw trajectory.(tracking)")
flags.DEFINE_enum("model_size", "nano",
                  ["nano", "small", "medium", "large", "xlarge"], "Model size.")
flags.DEFINE_enum('input_type', 'image', ['image', 'video'],
                  'Choose a input type.')
flags.DEFINE_string("data_parent_dir", "data/", "Data paraent directory.")
flags.DEFINE_string("data_name", None,
                    "if not set, all data in directory will be used.")

# 프로그램 실행에 사용되는 모든 플래그(커맨드 라인 옵션)를 저장하고 관리
# 프로그램의 시작 부분에서 정의하여, 전역적으로 커맨드 라인 옵션에 접근할 수 있게 함
FLAGS = flags.FLAGS(sys.argv)

FLAGS.draw_trajectory = FLAGS.do_tracking and FLAGS.draw_trajectory


def run(model: YOLO,
        data_paths: list[str],
        input_resize_shape: tuple[int, int],
        vid_stride: int = 3) -> None:
    for data_path in data_paths:
        # data_path: data/test_video/test_160x90.mp4
        # save_folder_name: test_160x90
        save_folder_name = os.path.basename(data_path).split(".")[0]
        # save_folder_name: test_160x90->(160, 90)
        save_folder_name = f"({save_folder_name})->{input_resize_shape}"
        # Remove space from save_folder_name.
        save_folder_name = save_folder_name.replace(" ", "")
        print("save_folder_name:", save_folder_name)
        if FLAGS.do_tracking:
            ### TODO (h.sb): 여기서부터 아래 표시한 부분까지 수정해서 쓰기.
            """
            tracker="bytetrack.yaml"  # Tracking with ByteTrack tracker
            """
            save_folder_name = f"[Tracker]{save_folder_name}"
            # results: List[Results]
            # len(results): Length of frames.
            results = model.track(source=data_path,
                                  tracker="bytetrack.yaml",
                                  save=not FLAGS.draw_trajectory,
                                  name=save_folder_name,
                                  conf=0.5,
                                  classes=[0],
                                  imgsz=input_resize_shape,
                                  stream_buffer=True,
                                  show_labels=True,
                                  show_conf=False,
                                  device="mps")  # , persist=True
            """
            구현 하고 싶은 것
                - inference time 결과를 파일로 저장
                - Tracking 결과의 시각화
                    - “id: 3 person” -> "3"으로 변경 (실패)
                    - 글씨 작게 하는 것도 실패
                    - Tracking 궤적을 그리기 (실패)
            구현 되어야 하는 것
                - 원래대로, 비디오로 결과물을 저장.

            Results 객체
                - Attributes
                    - boxes
                        - xywh : torch.Tensor
                        - xywhn
                        - xyxy
                        - xyxyn
                    - masks
                        - xy
                        - xyn
                    - keypoints
                    - obb
                    - speed: (dict) ms
                        - preprocess
                        - inference
                        - postprocess
                    - path: (str) The path to the image file.
                - Method
                    - __len__()
                    - plot(): (np.ndarray)
                        - Plots the detection results.
                        - Returns a numpy array of the annotated image.
                    -

            """
            parant_dir = os.path.dirname(data_path)
            video_filename = os.path.join(parant_dir,
                                          f"{save_folder_name}_track.mp4")
            print("video_filename:", video_filename)
            fps = 10  # 초당 프레임 수
            fourcc = cv2.VideoWriter_fourcc(*"XVID")  # 비디오 코덱 설정
            video_writer = None
            if draw_trajectory:

                # TODO: 아래 코드 작성해야함.
                # Get the boxes and track IDs
                track_history = defaultdict(lambda: [])
                for result in results:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    # Visualize the results on the frame
                    """
                    font_size
                    img: (np.ndarray) The image to draw on.
                    im_gpu: (Tensor) Normalized image in gpu
                        with shape (1, 3, 640, 640), for faster mask plotting.
                    labels: (bool) default: True
                    probs (bool) default: True
                    """
                    # plot a BGR numpy array of predictions
                    annotated_frame = result.plot(labels=True, probs=False)
                    if not video_writer:
                        frame_height, frame_width = annotated_frame.shape[:2]
                        # cv2.VideoWriter 객체 생성
                        video_writer = cv2.VideoWriter(
                            video_filename, fourcc, fps,
                            (frame_width, frame_height))
                    # Plot the tracks
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        # Dict[str, List[Tuple[float, float]]]
                        # track: List[Tuple[float, float]]
                        track = track_history[track_id]
                        new_point = (float(x), float(y))
                        last_point = track[-1] if track else new_point
                        if np.linalg.norm(
                                np.array(last_point) -
                                np.array(new_point)) < 30:
                            track.append(new_point)  # x, y center point

                        if len(track) > 30:  # retain 90 tracks for 90 frames
                            track.pop(0)

                        # Draw the tracking lines
                        # points: (k, 1 , 2)
                        points = np.hstack(track).astype(np.int32).reshape(
                            (-1, 1, 2))
                        cv2.polylines(annotated_frame, [points],
                                      isClosed=False,
                                      color=(230, 230, 230),
                                      thickness=5)
                        ######################################
                    video_writer.write(annotated_frame)
                # 비디오 쓰기 작업 완료
                video_writer.release()
            ### TODO (h.sb): 위에 표시 한 부분부터 여기까지 수정해서 쓰기.
        else:
            save_folder_name = f"[Predict]{save_folder_name}"
            model.predict(source=data_path,
                          save=True,
                          name=save_folder_name,
                          conf=0.5,
                          classes=[0],
                          imgsz=input_resize_shape,
                          vid_stride=vid_stride,
                          stream_buffer=True,
                          show_labels=False,
                          show_conf=False,
                          device="mps")


def _set_model() -> YOLO:
    algo_name = "yolov8"
    # Adapt model size into the algo_name.
    # "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"
    algo_name += FLAGS.model_size[0]
    if FLAGS.do_segmentation:
        algo_name += "-seg"
    algo_name += ".pt"
    model = YOLO(algo_name)
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
    else:
        with Image.open(data_paths[0]) as img:
            return img.size  # (width, height) 형태의 튜플을 반환


def main(argv):
    del argv
    model = _set_model()
    data_paths = _set_data_paths()

    total_frames_number = _get_total_frames_number(data_paths)
    total_process_frames_number = total_frames_number // FLAGS.vid_stride

    usage_data: Dict[str, Dict[str, Any]] = {}
    monitoring_interval = 0.1
    device = test2.get_device(use_gpu=FLAGS.use_gpu)
    system_info = test2.get_basic_system_info(device)

    flags.DEFINE_enum("resize_mode", "custom",
                      ["original", "custom", "default"],
                      " Input image resize mode.")
    if FLAGS.resize_mode == "custom":
        input_resize_shapes = [FLAGS.resize_shape]
    elif FLAGS.resize_mode == "default":
        input_resize_shapes = None
    elif FLAGS.resize_mode == "original":
        input_resize_shapes = [_get_image_resolution(data_paths)]
    else:
        raise ValueError(f"Invalid resize_mode: {FLAGS.resize_mode}")

    for input_resize_shape in input_resize_shapes:
        kwargs = {
            "model": model,
            "sources": data_paths,
            "input_resize_shape": input_resize_shape,
            "vid_stride": FLAGS.vid_stride
        }
        if FLAGS.check_resource_usage:
            a_usage_data: Dict[str, Any] = test2.check_cpu_and_memory_usage(
                function=run,
                kwargs=kwargs,
                monitoring_interval=monitoring_interval)
            usage_data[f"[Resize]{str(input_resize_shape)}"] = a_usage_data
        else:
            run(**kwargs)
    if FLAGS.check_resource_usage:
        test2.plot_multiple_usage_data(usage_data,
                                       system_info,
                                       monitoring_interval,
                                       total_frames=total_process_frames_number)
