from collections import defaultdict
import subprocess
import psutil
import time
import matplotlib.pyplot as plt
import threading
from typing import List, Callable, Optional, Dict, Any
import os
import pynvml  # NVIDIA Management Library Python Bindings
import platform

import torch
# torch.mps is available only on macOS with PyTorch built for Metal support
if torch.backends.mps.is_built():
    from torch.mps import profiler, event


def get_device(use_gpu: bool = False) -> torch.device:
    """
    Returns the torch device to use for model inference.

    Args:
        use_gpu (bool): Whether to use GPU for model inference.

    Returns:
        torch.device: Torch device to use for model inference.
    """
    if use_gpu:
        device = get_torch_gpu_device()
    else:
        device = torch.device("cpu")
    return device


def get_torch_gpu_device(gpu_idx: int = 0) -> torch.device:
    if platform.system() == "Darwin" and platform.uname().processor == "arm":
        assert torch.backends.mps.is_available(
        ), "MPS is not available on this device."
        device = torch.device(f"mps:{gpu_idx}")
    else:
        assert torch.cuda.is_available(
        ), "CUDA is not available on this device."
        device = torch.device(f"cuda:{gpu_idx}")
    return device


def monitor_mps_gpu_usage(interval: float, usage_data: Dict[str, List[float]]):
    """
    Monitors GPU memory usage using PyTorch MPS backend.

    Args:
        interval (float): Time interval in seconds for recording GPU usage.
        usage_data (Dict[str, List[float]]): Dictionary to store GPU usage data.
    """
    while getattr(threading.currentThread(), "do_run", True):
        if torch.backends.mps.is_built():
            # GPU memory usage
            current_mem = torch.mps.current_allocated_memory() / (
                1024**3)  # Convert to GB
            driver_mem = torch.mps.driver_allocated_memory() / (
                1024**3)  # Convert to GB

            usage_data["[MPS]current_allocated_memory(GB)"].append(current_mem)
            usage_data["[MPS]driver_allocated_memory(GB)"].append(driver_mem)

        time.sleep(interval)


def monitor_gpu_usage(interval: float,
                      usage_data: Dict[str, List[float]],
                      device_id: int = 0):
    """
    Monitors the GPU utilization and memory usage.

    Args:
        interval (float): Time interval in seconds for recording GPU usage.
        usage_data (Dict[str, List[float]]): Dictionary to store GPU usage and memory usage.
        device_id (int): GPU device ID to monitor.
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    while getattr(threading.currentThread(), "do_run", True):
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage_data[f"gpu_{device_id}_utilization(%)"].append(util.gpu)
        usage_data[f"gpu_{device_id}_memory_used(GB)"].append(mem_info.used /
                                                              (1024**3))
        time.sleep(interval)


def monitor_system_usage(interval: float,
                         usage_data: Dict[str, List[float]]) -> None:
    """
    Monitors system CPU and RAM usage, including total and used memory, at regular intervals.

    Args:
        interval (float): Time interval in seconds for recording system usage.
        usage_data (Dict[str, List[float]]): Dictionary to store CPU usage, RAM usage,
            total memory, and used memory. Keys are "cpu_usage(%)", "ram_usage(%)", "total_memory(GB)", and "used_memory(GB)",
            each mapping to a list of values.
    """
    gpu_monitor_threads = []
    if torch.cuda.is_available():
        # torch.cuda.device_count(): 기계에 설치된 CUDA 호환 GPU의 총 개수를 정수로 제공
        for device_idx in range(torch.cuda.device_count()):
            gpu_thread = threading.Thread(target=monitor_gpu_usage,
                                          args=(interval, usage_data,
                                                device_idx))
            gpu_thread.start()
            gpu_monitor_threads.append(gpu_thread)
    # elif torch.backends.mps.is_built():
    #     gpu_thread = threading.Thread(target=monitor_mps_gpu_usage,
    #                                   args=(interval, usage_data))
    #     gpu_thread.start()
    #     gpu_monitor_threads.append(gpu_thread)

    while getattr(threading.currentThread(), "do_run", True):
        """
            psutil.cpu_percent()
                CPU 사용률 백분율은 모든 코어에 걸쳐 계산됩니다.
                따라서 100% 값은 모든 코어가 완전히 사용되고 있음을 의미합니다.
            psutil.virtual_memory().percent
                (총 메모리 - 사용 가능 메모리) / 총 메모리 * 100
            psutil.virtual_memory().total
                총 메모리 (바이트) -> GB로 변환하려면 1024 ** 3으로 나눕니다.
            psutil.virtual_memory().available
                사용 가능한 메모리 (바이트) -> GB로 변환하려면 1024 ** 3으로 나눕니다.
        """
        vmem = psutil.virtual_memory()
        usage_data["cpu_usage(%)"].append(psutil.cpu_percent())
        usage_data["ram_usage(%)"].append(vmem.percent)
        if usage_data["total_memory(GB)"] == []:
            usage_data["total_memory(GB)"].append(vmem.total / (1024**3))
        usage_data["used_memory(GB)"].append(
            (vmem.total - vmem.available) / (1024**3))
        time.sleep(interval)

    for thread in gpu_monitor_threads:
        thread.do_run = False
        thread.join()


def plot_multiple_usage_data(usage_data: Dict[str, Dict[str, List[float]]],
                             system_info: str,
                             interval: float,total_frames: int, show: bool = True,
                             save: bool = True, save_path: str = None) -> None:
    """
    Plots multiple sets of CPU, RAM, and GPU (if available) usage data during model inference.
    Args:
        usage_data: {"name": usage_data}
        system_info (str): System information to display.
        interval (float): Time interval for the data points.
        show (bool, optional): Whether to show the plot. Default is True.
        save (bool, optional): Whether to save the plot as an image. Default is False.
        save_path (str, optional): File path to save the plot (if save is True). Default is None.
    """


    usage_data_list = list(usage_data.values())

    # [1, 0] 위치에 각 Usage_data당 소요 시간 추가
    system_info += f"\n[Total Frames]: {total_frames}\n"
    for i, (name, a_usage_data) in enumerate(usage_data.items()):
        execution_time = len(a_usage_data['cpu_usage(%)']) * interval
        fps = total_frames / execution_time
        execution_period_per_frame = execution_time / total_frames
        system_info += f"\n[Data {name} Execution Time]: {execution_time:.2f} seconds."
        system_info += f"\n[Data {name} FPS]: {fps:.2f} frames per second."
        system_info += f"\n[Data {name} Execution Period]: {execution_period_per_frame:.3f} seconds per frame.\n"

    # 가장 긴 데이터를 기반으로 time_axis 생성
    max_data_length = max(len(data['cpu_usage(%)']) for data in usage_data_list)
    time_axis = [i * interval for i in range(max_data_length)]

    # 모든 데이터를 동일한 길이로 맞춤
    for i, a_usage_data in enumerate(usage_data_list):
        data_length = len(a_usage_data['cpu_usage(%)'])
        if data_length < max_data_length:
            # 데이터 길이가 짧으면 나머지를 0으로 채움
            usage_data_list[i]['cpu_usage(%)'].extend(
                [0] * (max_data_length - data_length))
            usage_data_list[i]['ram_usage(%)'].extend(
                [0] * (max_data_length - data_length))
            a_usage_data['used_memory(GB)'].extend(
                [0] * (max_data_length - data_length))

    # 2x2 그리드에 그래프 표시 (마지막 자리는 비워둠)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 각 데이터를 겹쳐서 그래프에 그림
    for i, (name, a_usage_data) in enumerate(usage_data.items()):
        # CPU 사용량 그래프
        axs[0, 0].plot(time_axis,
                       a_usage_data['cpu_usage(%)'],
                       linestyle='-',
                       label=f'Data {name}')

        # RAM 사용량 그래프
        axs[0, 1].plot(time_axis,
                       a_usage_data['ram_usage(%)'],
                       linestyle='-',
                       label=f'Data {name}')
        # 총 메모리와 사용된 메모리 그래프
        axs[1, 1].plot(time_axis,
                       a_usage_data['used_memory(GB)'],
                       marker='x',
                       linestyle='-',
                       label=f'Used Memory Data {name}')

    # 총 메모리 그래프 (한 번만 그림)
    total_memory_gb = usage_data_list[0]['total_memory(GB)'][
        0]  # 첫 번째 데이터의 total_memory 값 사용
    axs[1, 1].plot([time_axis[0], time_axis[-1]],
                   [total_memory_gb, total_memory_gb], linestyle='--',
                   label='Total Memory')

    # 그래프 타이틀과 레이블 설정
    axs[0, 0].set_title('CPU Usage Over Time (%)')
    axs[0, 0].set_xlabel('Time (seconds)')
    axs[0, 0].set_ylabel('CPU Usage (%)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].set_title('RAM Usage Over Time (%)')
    axs[0, 1].set_xlabel('Time (seconds)')
    axs[0, 1].set_ylabel('RAM Usage (%)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()


    axs[1, 0].text(0.5, 0.5, system_info, ha='center', va='center', fontsize=10)
    axs[1, 0].axis('off')



    axs[1, 1].set_title('RAM Usage Over Time (GB)')
    axs[1, 1].set_xlabel('Time (seconds)')
    axs[1, 1].set_ylabel('Memory (GB)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()



    plt.tight_layout()

    if save:
        if save_path:
            plt.savefig(save_path)
        else:
            current_time = time.strftime("%H%M%S")
            save_path = f"usage_data_plot_{current_time}.png"
            plt.savefig(save_path)

    if show:
        plt.show()


def plot_usage_data(usage_data: Dict[str, List[float]], system_info: str,
                    interval: float) -> None:
    """
    Plots CPU, RAM, and GPU (if available) usage data during model inference.
    """
    # 2x2 그리드에 그래프 표시 (마지막 자리는 비워둠)
    time_axis = [i * interval for i in range(len(usage_data['cpu_usage(%)']))]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # CPU 사용량 그래프
    axs[0, 0].plot(time_axis,
                   usage_data['cpu_usage(%)'],
                   marker='o',
                   linestyle='-',
                   color='blue')
    axs[0, 0].set_title('CPU Usage Over Time')
    axs[0, 0].set_xlabel('Time (seconds)')
    axs[0, 0].set_ylabel('CPU Usage (%)')
    axs[0, 0].grid(True)

    # RAM 사용량 그래프
    axs[0, 1].plot(time_axis,
                   usage_data['ram_usage(%)'],
                   marker='o',
                   linestyle='-',
                   color='red')
    axs[0, 1].set_title('RAM Usage Over Time')
    axs[0, 1].set_xlabel('Time (seconds)')
    axs[0, 1].set_ylabel('RAM Usage (%)')
    axs[0, 1].grid(True)

    # 총 메모리와 사용된 메모리 그래프
    axs[1, 1].plot(time_axis,
                   usage_data['total_memory(GB)'],
                   marker='o',
                   linestyle='-',
                   color='green',
                   label='Total Memory')
    axs[1, 1].plot(time_axis,
                   usage_data['used_memory(GB)'],
                   marker='x',
                   linestyle='-',
                   color='purple',
                   label='Used Memory')
    axs[1, 1].set_title('Memory Usage Over Time')
    axs[1, 1].set_xlabel('Time (seconds)')
    axs[1, 1].set_ylabel('Memory (GB)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 마지막 그리드에 시스템 정보 표시
    axs[1, 0].text(0.5, 0.5, system_info, ha='center', va='center', fontsize=10)
    axs[1, 0].axis('off')

    plt.tight_layout()
    plt.show()


def run_model() -> None:
    """

    basic calculation that add 1 to 1000000000.
    iteration : 1000000000
    """
    iteration = 100000000
    for i in range(iteration):
        i += 1
    print(f"Total iteration: {iteration}")


def get_basic_system_info(device: torch.device) -> str:
    total_cores = os.cpu_count()
    system_info = ""
    system_info += "======[Basic System Information]======"
    system_info += f"\n[Total CPU cores]: {total_cores}"
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    system_info += f"\n[Total CPU memory]: {total_memory_gb} GB"
    if not device.type == "cpu":
        if device.type == "cuda":
            total_gpu_cores = pynvml.nvmlDeviceGetCount()
            system_info += f"\n[Total GPU cores]: {total_gpu_cores}"
            total_gpu_memory_gb = sum(
                pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(i)).total
                for i in range(total_gpu_cores)) / (1024**3)
            system_info += f"\n[Total GPU memory]: {total_gpu_memory_gb} GB"
        elif torch.backends.mps.is_built():
            system_info += f"\n[Total GPU cores]: {torch.mps._get_num_gpus()}"
            # using  "ioreg -l | grep gpu-core-coun" to get the number of GPU cores.
            total_gpu_cores = subprocess.check_output(
                "ioreg -l | grep 'gpu-core-count'", shell=True, text=True)
            # 출력에서 숫자 값 추출
            total_gpu_cores = int(
                total_gpu_cores.strip().split('=')[-1].strip())
            system_info += f"\n[Total GPU cores]: {total_gpu_cores}"
            system_info += f"\n[MAC][Total GPU memory](= mac shares memory with CPU): {total_memory_gb} GB "
    system_info += "\n======================================"
    return system_info


def check_cpu_and_memory_usage(
    function: Callable = run_model,
    kwargs: Optional[Dict[str, Any]] = None,
    monitoring_interval: float = 0.1,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    if device.type == "cuda":
        pynvml.nvmlInit()  # Initialize NVML.
    usage_data = defaultdict(list)
    monitor_thread = threading.Thread(target=monitor_system_usage,
                                      args=(monitoring_interval, usage_data))
    monitor_thread.start()

    start_time = time.time()
    function(**kwargs)
    end_time = time.time()

    monitor_thread.do_run = False
    monitor_thread.join()
    if device.type == "cuda":
        pynvml.nvmlShutdown()  # Clean up NVML resources
    return usage_data
