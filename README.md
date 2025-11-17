# PySR_gamma 使用说明

> 本项目中的路径均已使用相对路径，请放心使用
>
> 但是源文件是在 MacOS 26 上创建的，在 Windows 上打开时可能会出现奇奇怪怪的问题，请注意
>
> 比如 bash 指令的部分名称不同，路径分隔符不同等
>
> 在 Ubuntu 22.04 上测试没有问题

## 项目结构

```
/XAI_for_Microgrid_Optimization/
│
├── PySR_gamma/
│   ├── dataset/
│   │   └── dataset_ieee33_extreme.csv            # 原始输入数据
│   │
│   ├── output_vector/
│   │   ├── gamma_calculator.py                   # 自动生成的、可直接调用的 Python 模块
│   │   └── gamma_formulas.csv                    # 包含公式的 CSV 文件
│   │   └── result.md                             # 分析结果的 Markdown 文件，内含有特征索引映射表和所有以 LaTeX 格式表示的公式，便于查看
│   │
│   └── analyze_ieee_whole_gamma_programmatic.py  # 用于分析数据并生成 gamma_calculator.py 的主脚本
│   │
│   ├── outputs                                   # PySR 运行时生成的中间文件和结果文件夹
│
├── model_gamma.py                                # 强化学习模型定义文件
│
├── main.py                                       # 调用 gamma_calculator.py 的主程序
│
└── README.md                                     # 本文档
```

## PySR 环境配置

### 创建并激活环境

```bash
conda create -n pysr-env python=3.9 pysr pandas -y
conda activate pysr-env
```

### 启动 python

```bash
python
```

### 在 Python 中安装 Julia

在 Python 解释器中执行一下命令，会自动下载并安装 Julia

```python
import pysr
pysr.install()
```

在 Windows 上，如果报错，可以尝试在 Python 解释器中直接运行

```python
import pysr
```

### （可选）验证是否配置成功

可使用以下程序进行验证

```python
import numpy as np

X = np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0]**2 - 2.0
from pysr import PySRRegressor

model = PySRRegressor(
    niterations=5,  # 这里的迭代次数设置得比较小，用于快速测试
    binary_operators=["+", "*", "/"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",  # 也可以自定义操作符
    ],
    model_selection="best",
)

model.fit(X, y)
```

## 工作流程

### 运行拟合脚本进行分析以及模块生成

在 `XAI_for_Microgrid_Optimization` 目录下运行

```bash
python PySR_gamma/analyze_ieee33_whole_gamma_programmatic.py
```

这个过程会为数据集中的每一个 `yhat_` 变量运行一次完整的 PySR 分析，时间预估在 $3\sim 5$ 分钟左右（$26$ 个因变量，$20$ 个自变量的情况下）

### 在主脚本中使用生成的模块

**注意：pysr-env 中可能不含有部分深度学习相关的库**

这里用 ai 生成了一个 `main.py` 作为示例，展示如何调用 `gamma_calculator.py` 模块

自己写时，可以参考 `main.py` 的写法

1.  **导入 `GeneratedGammaCalculator` 类**：在主脚本中，从生成的模块里导入这个类

    ```python
    from PySR_gamma.output_vector.gamma_calculator import GeneratedGammaCalculator
    ```

2.  **创建实例并计算 `gamma`**：在您的训练循环中，根据当前的环境状态计算 `gamma` 向量。

    ```python
    # 在初始化阶段
    gamma_calculator = GeneratedGammaCalculator()

    # ... 在训练循环的每一步 ...

    # state_dict 是从环境中获取的当前状态 (一个 Python 字典)
    state_dict = {'temperature_C': 30.0, 'wind_speed_ms': 5.0, ...}

    # 计算 gamma 张量
    gamma_tensor = gamma_calculator.compute(state_dict)

    # 将 state_tensor 和 gamma_tensor 传入您的策略网络
    action, log_prob, _ = policy.sample(state_tensor, gamma_tensor)
    ```

    `gamma_tensor` 现在包含了所有 `yhat_` 变量根据当前状态计算出的预测值，可以直接用于调制 `GaussianPolicy` 网络
