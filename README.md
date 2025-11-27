# XAI_for_Microgrid_Optimization(PySR + SAC)

本项目结合了 **可解释性人工智能 (PySR)** 与 **深度强化学习 (SAC)**，旨在解决 IEEE 33 节点微电网在极端场景下的电压控制与能源调度问题。

项目通过符号回归（PySR）挖掘环境特征生成 `gamma` 因子，辅助 Soft Actor-Critic (SAC) 智能体进行更高效、可解释的决策。

## 项目结构

```text
XAI_test/
├── main_run.py                     # [入口] 主程序：负责训练 SAC 智能体并调用绘图
├── main.py                         # 不知道什么用的一个主程序
├── requirements.txt                # 项目依赖列表（已锁定版本以防冲突）
├── config.py                       # 全局路径与设备配置
│
├── dataset/                        # 数据集目录
│   └── dataset_ieee33_extreme.csv  # IEEE 33 节点微电网负荷与光伏数据
│
├── PySR_gamma/                     # [模块 1] 符号回归模块
│   ├── analyze_ieee33...py         # PySR 运行脚本，用于挖掘公式
│   └── output_vector/              # PySR 输出结果
│       ├── gamma_calculator.py     # 自动生成的公式计算代码（供 SAC 调用）
│       └── ...
│
├── SAC_gama/                       # [模块 2] 强化学习核心算法
│   ├── sac.py                      # SAC 算法实现
│   ├── microgrid_env...v11.py      # IEEE 33 节点微电网仿真环境 (基于 Pandapower)
│   ├── model_gamma.py              # 神经网络模型定义
│   └── replay_memory.py            # 经验回放池
│
└── outputs/                        # 实验结果归档（自动生成）
    └── YYYY-MM-DD_HH-MM-SS.../     # 每次运行的独立文件夹
        ├── logs/                   # TensorBoard 训练日志
        ├── node_voltages.csv       # 节点电压详细数据
        ├── plot_summary.png        # 综合分析图表（功率、SOC、电压）
        └── ...
```

## 安装与环境配置

建议使用 Conda 创建 Python 3.9 环境：

```bash
conda create -n xai_grid python=3.9
conda activate xai_grid
```

安装依赖（请务必使用本项目提供的 `requirements.txt` 以避免 NumPy 版本冲突）：

```bash
pip install -r requirements.txt
```

> **注意**：本项目依赖 `pandapower` 和 `gym` 的旧版本，因此 `numpy` 被锁定在 `<2.0`。请勿随意升级 numpy。

## 运行指南

### 第一步：运行 PySR 符号回归

如果你需要重新生成特征公式：

```bash
python PySR_gamma/analyze_ieee33_whole_gamma_programmatic.py
```

- 这将在 `PySR_gamma/output_vector/` 下生成新的 `gamma_calculator.py`。

### 第二步：运行 SAC 训练 (主程序)

加载数据、环境与生成的公式，开始训练并自动保存结果：

```bash
python main_run.py
```

- 程序会自动检测你的硬件（CUDA/MPS/CPU）。
- 训练完成后，会自动进行绘图并保存。

## 结果输出

运行结束后，请查看 `outputs/` 文件夹下的最新目录。包含以下内容：

1.  **可视化图表**：
    - `plot_summary.png`: 包含多源功率堆叠图、系统能量平衡、SOC 曲线、关键节点电压。
    - `plot_voltage_heatmap.png`: 全节点电压热力图。
    - `plot_all_nodes_voltage.png`: 所有 33 个节点的电压曲线。
2.  **数据文件**：
    - `node_voltages.csv`: 详细的电压时序数据。
    - `voltage_analysis_report.csv`: 越限分析报告。
3.  **训练日志**：
    - `logs/`: 可使用 TensorBoard 查看 (`tensorboard --logdir outputs/`)。
