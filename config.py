# -*- coding: utf-8 -*-
import os
import torch
import sys

# ==============================================================================
# 1. 路径配置 (Path Configuration)
# ==============================================================================
# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据集路径
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "dataset_ieee33_extreme_full.csv")

# PySR 产出路径 (gamma_calculator.py 所在位置)
PYSR_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "PySR_gamma", "output_vector")

# 实验结果输出根目录
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs")

# ==============================================================================
# 2. 硬件加速配置 (Hardware Acceleration)
# ==============================================================================
def get_device():
    """
    自动检测并返回最佳计算设备:
    1. NVIDIA GPU (CUDA) - Windows/Linux
    2. Apple Silicon (MPS) - Mac M1/M2/M3/M4
    3. CPU - 兜底方案
    """
    if torch.cuda.is_available():
        # print("配置: 检测到 NVIDIA GPU，启用 CUDA 加速")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # print("配置: 检测到 Apple Silicon，启用 MPS 加速")
        return torch.device("mps")
    else:
        # print("配置: 未检测到加速设备，使用 CPU")
        return torch.device("cpu")

# 全局设备变量，其他文件直接 import DEVICE 即可
DEVICE = get_device()

# ==============================================================================
# 3. 其他全局设置 (Global Settings)
# ==============================================================================
# 确保输出目录存在
os.makedirs(OUTPUTS_ROOT, exist_ok=True)
os.makedirs(PYSR_OUTPUT_DIR, exist_ok=True)

# 字体设置 (用于 Matplotlib 绘图时避免中文乱码)
FONT_SANS_SERIF = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "sans-serif"]