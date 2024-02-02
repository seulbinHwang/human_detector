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
from ultralytics import YOLO
import os
from torch.profiler import profile, ProfilerActivity, record_function
import torch  # torch를 사용하는 경우 추가
from test2 import check_cpu_and_memory_usage

use_predict = True
use_segmentation = False
use_video = True

if use_segmentation:
    model = YOLO('yolov8n-seg.pt')
else:
    model = YOLO('yolov8n.pt')

video_dir = "data/test_video/"
if use_video:
    sources = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith((".mp4")) and f.startswith("test")
    ]
    sources.sort()
else:
    sources = ["data/test_image/test.jpg"]

# PyTorch Profiler 설정에 profile_memory=True 추가
activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

for source in sources:
    input_size = (32 * 6, 32 * 10)

    with profile(
            activities=activities,
            record_shapes=True,  # whether to record shapes of ops inputs.
            profile_memory=True
    ) as prof:  # profile_memory: model의 Tensor가 소비한 메모리의 양을 기록
        with record_function("model_predict"):
            model.predict(source=source,
                          save=True,
                          conf=0.5,
                          classes=0,
                          imgsz=input_size,
                          vid_stride=3,
                          stream_buffer=True,
                          show_labels=False,
                          show_conf=False)
    # prof.export_chrome_trace("trace.json")

    # 프로파일링 결과 출력
    print("==============[start]==============")
    # NOTE: the first use of CUDA profiling may bring extra overhead.
    print(prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else
        "cpu_time_total",  # "cuda_time_total" or "cpu_time_total".
        row_limit=10))
    # 메모리 사용량에 대한 정보도 포함하여 출력
    print("-----------------------------")
    print(prof.key_averages().table(
        sort_by="self_cpu_memory_usage",  #
        row_limit=10))
    print("==============[end]==============")
"""

### 해석 방법

- **model_predict**는 전체 프로파일링 시간의 100%를 차지하며, 이는 프로파일링된 전체 과정을 의미합니다. 이는 단일 호출로 측정되어 총 4.361초가 소요되었습니다.
- **aten::_slow_conv2d_forward** 연산이 전체 시간의 상당 부분(약 35.41%의 Self CPU %)을 차지하고 있으며, 이는 CNN(Convolutional Neural Network) 모델에서 주로 발생하는 연산 중 하나입니다. 이 연산은 평균적으로 118.081us(마이크로초)의 CPU 시간을 소요하며, 총 14208번 호출되었습니다.
- **aten::silu_** (Swish activation function의 in-place 버전)도 상당한 CPU 시간(약 26.79%)을 차지합니다. 평균적으로 92.338us의 CPU 시간을 소요하며, 12654번 호출되었습니다.
- **aten::copy_**와 **aten::cat** 같은 연산들은 데이터를 복사하거나 연결하는 작업으로, 메모리 관련 연산에 대한 정보를 제공합니다.

### 성능 최적화 포인트

- **aten::_slow_conv2d_forward**와 같이 상당한 시간을 차지하는 연산은 성능 최적화의 주요 대상입니다. 이는 모델 아키텍처의 변경, 더 효율적인 연산 사용, 또는 하드웨어 가속을 통해 최적화할 수 있습니다.
- 호출 횟수가 많은 연산들은 최적화를 통해 전체 실행 시간을 줄일 수 있는 잠재적인 기회를 제공합니다. 예를 들어, 연산을 더 적게 호출하거나, 더 효율적인 방법으로 같은 작업을 수행할 수 있는지 검토할 수 있습니다.
- 전체적인 성능 분석과 최적화 전략 수립을 위해, 가장 많은 시간을 소비하는 연산과 가장 자주 호출되는 연산에 주목해야 합니다.

- **Name**: 연산의 이름입니다. 
    예를 들어, `model_predict`는 사용자 정의 함수명으로, 모델의 예측을 실행하는 전체 과정을 의미
    `aten::conv2d`, `aten::silu_` 등은 PyTorch 내부에서 실행된 연산을 나타냄.
- **Self CPU %**: 전체 프로파일링 시간 중, 해당 연산이 직접 소비한 CPU 시간의 비율
- **Self CPU**: 해당 연산이 직접 소비한 CPU 시간 (children operator call이 소요한 시간은 포함하지 않음)
- **CPU total %**: 전체 프로파일링 시간 중, 해당 연산 및 하위 호출이 소비한 CPU 시간의 비율
    예를 들어, `aten::convolution`은 
        `aten::conv2d`를 포함하여 해당 연산과 관련된 모든 하위 호출의 CPU 시간을 합한 것입니다.
- **CPU total**: 해당 연산 및 하위 호출이 소비한 총 CPU 시간 (children operator call이 소요한 시간도 포함)
- **CPU time avg**: 해당 연산을 실행하는 데 평균적으로 소요된 CPU 시간
- **# of Calls**: 해당 연산이 호출된 횟수입니다.

---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                    model_predict        16.53%     705.110ms       100.00%        4.265s        4.265s           0 b      -4.04 Gb             1  
                     aten::conv2d         0.94%      40.055ms        39.48%        1.684s     118.494us       1.88 Gb      -6.53 Mb         14208  
                aten::convolution         2.79%     118.979ms        39.32%        1.677s     118.029us       1.91 Gb      42.89 Mb         14208  
               aten::_convolution        -1.76%  -74847.000us        39.18%        1.671s     117.615us       1.98 Gb    -405.47 Mb         14208  
                aten::thnn_conv2d         5.38%     229.505ms        38.63%        1.647s     115.955us       2.27 Gb      83.52 Mb         14208  
       aten::_slow_conv2d_forward        35.44%        1.511s        38.50%        1.642s     115.555us       2.47 Gb      -5.01 Gb         14208  
                      aten::silu_        27.37%        1.167s        27.37%        1.167s      92.249us      -5.00 Mb      -5.00 Mb         12654  
                      aten::copy_         4.10%     174.949ms         4.10%     174.949ms       7.319us     168.37 Mb      90.17 Mb         23903  
                        aten::cat         2.18%      92.868ms         2.37%     101.064ms      22.783us       1.19 Gb     915.73 Mb          4436  
                 aten::max_pool2d         0.06%       2.447ms         1.84%      78.406ms     117.727us      26.25 Mb     -28.27 Mb           666  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  


---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                      aten::empty         0.05%       2.251ms         0.05%       2.251ms       0.081us       4.85 Gb       4.85 Gb         27665  
                    aten::resize_         0.13%       5.678ms         0.13%       5.678ms       0.394us       1.04 Gb       1.04 Gb         14426  
                        aten::cat         2.18%      92.868ms         2.37%     101.064ms      22.783us       1.19 Gb     915.73 Mb          4436  
                    aten::reshape         0.10%       4.304ms         0.23%       9.911ms       0.667us       1.15 Gb     889.53 Mb         14862  
                       aten::view         0.27%      11.441ms         0.27%      11.441ms       0.318us     884.53 Mb     883.76 Mb         35948  
                     aten::narrow         0.19%       8.037ms         0.26%      11.022ms       0.843us     306.44 Mb     301.94 Mb         13078  
                 aten::empty_like         0.03%       1.181ms         0.04%       1.564ms       0.652us     197.50 Mb     159.78 Mb          2397  
                        aten::add         0.27%      11.461ms         0.27%      11.461ms       4.701us     123.72 Mb     123.72 Mb          2438  
              aten::empty_strided         0.03%       1.197ms         0.03%       1.197ms       0.319us     108.78 Mb     108.78 Mb          3754  
                      aten::copy_         4.10%     174.949ms         4.10%     174.949ms       7.319us     168.37 Mb      90.17 Mb         23903  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  

Self CPU time total: 4.361s

"""
