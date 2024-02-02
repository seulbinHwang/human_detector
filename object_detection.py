from ultralytics import YOLO
import numpy as np
import os
use_predict = True
use_segmentation = False
use_video = True
# Load a pretrained YOLOv8n-seg Segment model
if use_segmentation:
    model = YOLO('yolov8n-seg.pt')
else:
    # Load a model
    model = YOLO('yolov8n.pt')
# Run batched inference on a list of images.
"""
source
<테스트>
- image: "image.jpg"
- video: "video.mp4" (MP4, AVI, ...)
- YouTube: "https://www.youtube.com/watch?v=9j6DsbUuFwM"
<실제>
- numpy ndarray: (320, 320, 3)
  - HWC format with BGR channels uint8 (0-255).
- torch: (16,3,320,640)
  - BCHW format with RGB channels float32 (0.0-1.0)
"""
video_dir = "data/test_video/"
if use_video:
    # Find all video files (mp4) which name is start with "test" in the directory.
    sources = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith((".mp4")) and f.startswith("test")
    ]
    # Sort by name.
    sources.sort()
else:
    sources = ["data/test_image/test.jpg"]
if use_predict:
    # Run inference on 'bus.jpg' with arguments
    for source in sources:
        """
        160*90 -> 384*640: 1.0ms preprocess, 39.1ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)
        160*90 -> 192x320:  0.3ms preprocess, 16.9ms inference, 0.4ms postprocess per image at shape (1, 3, 192, 320)
        160*90 -> 96x160:  0.2ms preprocess, 7.8ms inference, 0.2ms postprocess per image at shape (1, 3, 96, 160)
        """
        # extract input_size from source name. (source: source: data/test_video/test_160x90.mp4)
        input_size = os.path.basename(source).split(".")[0].split("_")[1]
        # "160x90" -> (160, 90)
        print("input_size:", input_size)
        # input_size = tuple(map(int, input_size.split("x")))
        input_size = (32*6, 32*10)
        # WARNING ⚠️ imgsz=[160, 90] must be multiple of max stride 32, updating to [160, 96]
        # imgsz=input_size,
        model.predict(source=source, save=True, conf=0.5, classes=0, imgsz=input_size, vid_stride=3, stream_buffer=True, show_labels=False, show_conf=False)
    """
    show: (bool)
    save: (bool)
    save_txt: (bool) save results as .txt file
        - bounding box의 좌표를 저장하는 듯 함.
        - 예: "test_160x90_1304.txt"
            0 0.556801 0.461859 0.0568791 0.269895
            0 0.462516 0.436262 0.0623912 0.314578
            0 0.613204 0.416858 0.0474324 0.202324
    imgsz: (int or tuple) image size as scalar or (h, w) list, i.e. (640, 480)
        - 설정 하던 안하던, output size는 Input size와 같게 출력됨
    conf: (float) object confidence threshold for detection.
    classes: (list[int]): filter results by class, i.e. classes=0, or classes=[0,2,3]
      - 0: person
    retina_masks: (bool): use high-resolution segmentation masks
    half: (bool) use FP16 half-precision inference
        - FP16 반정밀도(half-precision) 연산을 사용할지 여부를 결정
        - 특히 GPU에서의 계산 속도를 높이고 메모리 사용량을 줄이는 효과
        
    stream_buffer: (bool) buffer all streaming frames (True) 
        or return the most recent frame (False)
        - 만약 True로 설정된다면, 모델은 비디오 스트림에서 오는 모든 프레임을 버퍼에 저장하게 됩니다. 
            - 이는 처리 속도보다 더 빠른 속도로 비디오 스트림이 전송될 때 유용합니다. 
            - 모든 프레임이 처리되어 중요한 정보가 누락되지 않도록 합니다. 
        - 반면, False로 설정되면, 모델은 가장 최근에 수신된 프레임만을 반환하고 처리합니다. 
            - 이는 리소스가 제한적이거나 가장 최신의 데이터만 중요한 경우에 유용할 수 있습니다.
            
    vid_stride: (bool): video frame-rate stride
        - boolean이 아니라 정수(int) 타입으로 설정되어야 합니다. 
        - vid_stride는 비디오 프레임을 처리할 때의 간격을 지정합니다. 
        - 예를 들어, vid_stride가 1로 설정되면 모든 프레임이 처리됩니다. 
        - vid_stride가 2로 설정되면, 하나 걸러 하나씩 프레임이 처리됩니다. 
        - 이는 처리해야 할 프레임 수를 줄여 성능을 향상시킬 수 있지만, 
        - 중요한 정보를 놓칠 수 있는 위험이 있습니다.
        
    hide_labels: (bool)
    hide_conf: (bool)
    """
else:
    #  # return a generator of Results objects
    results = model(source=sources, stream=True)
    # Process results generator
    #     save_txt(): Save predictions into a txt file.
    for result in results:
        # (height, width) of original image -> (3000, 4000)
        orig_shape = result.orig_shape
        print("orig_shape:", orig_shape)
        # A dictionary of preprocess, inference, and postprocess speeds
        # in milliseconds per image.
        # speed: {'preprocess': 7.455, 'inference': 90.99, 'postprocess': 9.265}
        speed = result.speed
        print("speed:", speed)
        # boxes = result.boxes  # Boxes object for bbox outputs
        # masks = result.masks  # Masks object for segmentation masks outputs
        # # print("masks.xyn:", len(masks.xyn)) # 1
        # # print("masks.xyn:", len(masks.xyn[0])) # 390
        # # print("masks.xyn:", len(masks.xyn[0][0])) # 2 (x,y)
        # # print("masks.xy:", len(masks.xy))
        # # print("masks.xy:", masks.xy[0].shape())
        # # print("masks.xy:", masks.xy[0].shape())
        # keypoints = result.keypoints  # Keypoints object for pose outputs
        # probs = result.probs  # Probs object for classification outputs
        # # A Probs object containing probabilities of each class
        # # for classification task.


"""
result.boxes

cls: tensor([0.])
conf: tensor([0.2632])
  - Return the confidence values of the boxes.
data: tensor([[1.4745e+03, 8.0217e+02, 3.3660e+03, 2.2029e+03, 2.6322e-01, 0.0000e+00]])
id: None
is_track: False
orig_shape: (3000, 4000)
shape: torch.Size([1, 6])
xywh: tensor([[2420.2317, 1502.5157, 1891.4443, 1400.6960]])
xywhn: tensor([[0.6051, 0.5008, 0.4729, 0.4669]])
  - Return the boxes in xywh format normalized by original image size.
xyxy: tensor([[1474.5095,  802.1677, 3365.9539, 2202.8638]])
  - Return the boxes in xyxy format.
xyxyn: tensor([[0.3686, 0.2674, 0.8415, 0.7343]])
  - Return the boxes in xyxy format normalized by original image size.

"""

"""
Masks
- data:
- orig_shape: (3000, 4000)
- shape: torch.Size([1, 480, 640])
- xy: (torch.Tensor) # (1, 390, 2) 
  - low 값을 -> np.array로 변환하거나, torch.Tensor로 변환하거나 해야함.
  - A list of normalized segments represented as tensors.
- xyn: (torch.Tensor) # (1, 390, 2)
  - low 값을 -> np.array로 변환하거나, torch.Tensor로 변환하거나 해야함.
  - A list of segments in pixel coordinates represented as tensors.



"""


"""
probs
- detection을 하던, segmentation을 하던 -> 모두 None이 나왔는데 확인 필요
"""
