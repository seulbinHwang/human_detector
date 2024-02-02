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
from typing import Any, Dict, List, Tuple
import numpy as np
import os
from ultralytics import YOLO
import test2
import cv2
import torch
from collections import defaultdict


def run(model: YOLO, sources: list[str], input_size: tuple[int, int], vid_stride:int=3) -> None:
    for source in sources:
        # source: data/test_video/test_160x90.mp4
        # name: test_160x90
        name = os.path.basename(source).split(".")[0]
        name = f"{(name)}->{input_size}"
        # Remove space from name.
        name = name.replace(" ", "")
        print("name:", name)
        if use_tracking:
            """
            tracker="bytetrack.yaml"  # Tracking with ByteTrack tracker
            """
            name = f"{name}_tracker"
            # results: List[Results]
            # len(results): Length of frames.
            results = model.track(source=source, tracker = "bytetrack.yaml",
                                  save=not draw_tracking,

                                  name=name,
                                  conf=0.5,
                                  classes=[0, 32],
                                  imgsz=input_size,
                                  stream_buffer=True,
                                  show_labels=True,
                                  show_conf=False, device="mps") # , persist=True
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
            parant_dir = os.path.dirname(source)
            video_filename = os.path.join(parant_dir, f"{name}_track.mp4")
            print("video_filename:", video_filename)
            fps = 10  # 초당 프레임 수
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 비디오 코덱 설정
            video_writer = None
            if draw_tracking:
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
                        video_writer = cv2.VideoWriter(video_filename, fourcc,
                                                       fps,
                                                       (frame_width,
                                                        frame_height))
                    # Plot the tracks
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        # Dict[str, List[Tuple[float, float]]]
                        # track: List[Tuple[float, float]]
                        track = track_history[track_id]
                        new_point = (float(x), float(y))
                        last_point = track[-1] if track else new_point
                        if np.linalg.norm(np.array(last_point) - np.array(new_point)) < 30:
                            track.append(new_point)  # x, y center point

                        if len(track) > 30:  # retain 90 tracks for 90 frames
                            track.pop(0)

                        # Draw the tracking lines
                        # points: (k, 1 , 2)
                        points = np.hstack(track).astype(np.int32).reshape(
                            (-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False,
                                      color=(230, 230, 230), thickness=5)
                        ######################################
                    video_writer.write(annotated_frame)
                # 비디오 쓰기 작업 완료
                video_writer.release()

        else:
            name = f"{name}_predict"
            model.predict(source=source,
                          save=True,
                          name= name,
                          conf=0.5,
                          classes=[0, 32],
                          imgsz=input_size,
                          vid_stride=vid_stride,
                          stream_buffer=True,
                          show_labels=False,
                          show_conf=False, device="mps")




if __name__ == "__main__":
    use_predict = True
    use_segmentation = False
    use_video = True
    use_tracking = False
    draw_tracking = False
    draw_tracking = use_tracking and draw_tracking
    use_resize = False
    use_large_model = True

    if use_segmentation:
        algo_name = "yolov8n-seg.pt"
    else:
        algo_name = "yolov8n.pt"
    if use_large_model:
        # change "n" to "x" in the algo_name
        algo_name = algo_name.replace("n", "x")
    model = YOLO(algo_name)


    video_dir = "data/test_fsa/"
    if use_video:
        sources = [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.endswith((".mp4", ".MOV")) and f.startswith("test")
        ]
        # sources: ['data/test_video/test_160x90.mp4',
        # 'data/test_video/test_320x180.mp4',
        # 'data/test_video/test_640x360.mp4']
        sources.sort()
    else:
        sources = ["data/test_image/test.jpg"]
    vid_stride=3

    total_frames = 0
    num_cores_to_use = 10
    torch.set_num_threads(num_cores_to_use)
    os.environ["OMP_NUM_THREADS"] = str(num_cores_to_use)
    for source in sources:
        if use_video:
            # 비디오 파일의 경우 OpenCV를 사용하여 프레임 수를 구함
            cap = cv2.VideoCapture(source)
            total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            # 이미지 파일의 경우 1 프레임을 추가
            total_frames += 1
    # shrink total_frames by vid_stride
    total_frames = total_frames // vid_stride
    usage_data: Dict[str, Dict[str, Any]] = {}
    monitoring_interval = 0.1
    device = test2.get_device(use_gpu=False)
    system_info = test2.get_basic_system_info(device)
    if use_resize:
        input_sizes = [(384, 640)] #, (192, 320)] # , (96, 160)
    else: # use original size
        print("sources:", sources)
        # sources[0]: 'data/test_fsa/test_1920x1080.mp4'
        # Get the size of the first frame of the first video.
        input_sizes = [(1080, 1920)] # code original is 384 * 640

    for input_size in input_sizes:
        kwargs = {"model": model, "sources": sources, "input_size": input_size,
                  "vid_stride": vid_stride}
        a_usage_data: Dict[str, Any] = test2.check_cpu_and_memory_usage(
            function=run,
            kwargs=kwargs,
            monitoring_interval=monitoring_interval)
        usage_data[f"[Resize]{str(input_size)}"] = a_usage_data
    test2.plot_multiple_usage_data(usage_data, system_info, monitoring_interval, total_frames)

    # check_cpu_and_memory_usage(interval=0.1, use_gpu=True)
