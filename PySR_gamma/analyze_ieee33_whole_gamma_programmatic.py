import pandas as pd
import numpy as np
from pysr import PySRRegressor
import os
import warnings

# 忽略一些 PySR 可能产生的警告信息
warnings.filterwarnings("ignore", category=FutureWarning)


# --- 1. 定义输出目录并创建 ---
output_dir = 'PySR_gamma/output_vector'
os.makedirs(output_dir, exist_ok=True)
print(f"所有输出结果将被保存在目录: '{output_dir}'\n")


# --- 2. 数据加载与预处理 ---

file_path = 'PySR_gamma/dataset/dataset_ieee33_extreme.csv' 
try:
    df = pd.read_csv(file_path)
    print(f"成功从 '{file_path}' 加载数据。\n")
except FileNotFoundError:
    print(f"错误: 找不到文件 at '{file_path}'。请检查路径和文件名是否正确。")
    exit()

df.replace({False: 0, True: 1}, inplace=True)


# --- 3. 准备特征和目标列表 ---

gamma_target_columns = [col for col in df.columns if col.startswith('yhat_')]
if not gamma_target_columns:
    print("错误: 在CSV文件中没有找到任何以 'yhat_' 开头的列作为 gamma 信号。")
    exit()

print(f"将要为以下 {len(gamma_target_columns)} 个 gamma 信号寻找公式: ")
print(gamma_target_columns)

columns_to_drop_for_X = gamma_target_columns + ['time', 'regime']
existing_columns_to_drop = [col for col in columns_to_drop_for_X if col in df.columns]
X_df = df.drop(columns=existing_columns_to_drop, errors='ignore')
X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)
X = X_df.to_numpy()

feature_names = X_df.columns.to_list()

print("\n" + "*"*60)
print("特征索引映射表 (用于将公式中的 x_i 映射到实际特征):")
for i, name in enumerate(feature_names):
    print(f"  x{i} = {name}")
print("*"*60)


# --- 4. 循环建模 ---

models = {}
for target_column in gamma_target_columns:
    print(f"\n{'='*80}\n正在分析 gamma 信号: {target_column}\n{'='*80}")
    y = df[target_column].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
    model = PySRRegressor(niterations=40, binary_operators=["+", "*", "/", "-"], unary_operators=["cos", "exp", "sin", "log"], model_selection="best")
    model.fit(X, y)
    models[target_column] = model


# --- 5. 汇总所有公式并生成所有输出文件 ---

print("\n\n" + "#"*80)
print("所有 gamma 信号的分析已完成！正在生成所有输出文件...")
print("#"*80)

# --- 准备用于写入文件的数据结构 ---
gamma_formulas_data = []
python_code_lines = [
    "# This file is auto-generated. Do not edit it manually.",
    "import numpy as np", "import torch\n\n", "class GeneratedGammaCalculator:",
    "    def __init__(self):", f"        self.feature_names = {feature_names!r}",
    f"        self.gamma_signals = {gamma_target_columns!r}\n",
    "    def compute(self, state):",
]
markdown_lines = [
    "# PySR Gamma 信号公式汇总\n",
    "## 特征索引映射表\n",
    "公式中的 $x_i$ 符号与以下实际特征相对应：\n",
    "| 符号 ($x_i$) | 特征变量名 |\n| :--- | :--- |",
]

# 添加特征映射表到 Markdown
for i, name in enumerate(feature_names):
    markdown_lines.append(f"| $x_{{{i}}}$ | `{name}` |")
markdown_lines.append("\n")

# 添加变量解包代码到 Python 模块
python_code_lines.append("        # Unpack state variables to match PySR's x_i notation")
for i, name in enumerate(feature_names):
    python_code_lines.append(f"        x{i} = state['{name}']")
python_code_lines.append("\n        # Calculate each gamma component")

# 遍历模型，同时填充所有输出文件的数据
gamma_component_vars = []
for i, target_column in enumerate(gamma_target_columns):
    fitted_model = models.get(target_column)
    component_var_name = f"gamma_{i}"
    gamma_component_vars.append(component_var_name)

    try:
        best_eq = fitted_model.get_best()
        equation_str = best_eq["equation"]
        best_idx = best_eq.name
        latex_equation = fitted_model.latex(index=best_idx, precision=4)
        has_result = True
    except (KeyError, IndexError, RuntimeError, ValueError) as e:
        equation_str = f"ERROR: {e}"
        latex_equation = "无法获取最佳公式"
        has_result = False

    # 1. 准备 CSV 数据
    gamma_formulas_data.append({'gamma_signal': target_column, 'equation': equation_str, 'latex_equation': latex_equation})
    
    # 2. 准备 Python 模块代码
    if has_result:
        py_equation = equation_str.replace('sin(', 'np.sin(').replace('cos(', 'np.cos(').replace('exp(', 'np.exp(').replace('log(', 'np.log(')
        python_code_lines.append(f"        {component_var_name} = {py_equation}  # Formula for {target_column}")
    else:
        python_code_lines.append(f"        {component_var_name} = 0.0  # ERROR for {target_column}")

    # 3. 准备 Markdown 内容
    markdown_lines.append("---\n")
    markdown_lines.append(f"### Gamma 信号: `{target_column}`\n")
    markdown_lines.append("**最佳公式:**\n")
    if has_result:
        markdown_lines.append(f"$$\n{latex_equation}\n$$\n")
    else:
        markdown_lines.append(f"```\n{latex_equation}\n{equation_str}\n```\n")

# --- 完成并写入 gamma_calculator.py ---
python_code_lines.extend([
    "\n        # Combine components into a vector",
    f"        gamma_values = [{', '.join(gamma_component_vars)}]",
    "\n        # Convert to a PyTorch tensor with a batch dimension",
    "        return torch.tensor(gamma_values, dtype=torch.float32).unsqueeze(0)"
])
py_file_path = os.path.join(output_dir, "gamma_calculator.py")
try:
    with open(py_file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(python_code_lines))
    print(f"\n[1/3] 可直接调用的 Python 模块已成功写入: {py_file_path}")
except Exception as e:
    print(f"\n写入 gamma_calculator.py 文件时出错: {e}")

# --- 写入 gamma_formulas.csv ---
gamma_df = pd.DataFrame(gamma_formulas_data)
gamma_csv_path = os.path.join(output_dir, "gamma_formulas.csv")
try:
    gamma_df.to_csv(gamma_csv_path, index=False)
    print(f"[2/3] 公式的 CSV 引用文件已成功写入: {gamma_csv_path}")
except Exception as e:
    print(f"\n写入 gamma_formulas.csv 文件时出错: {e}")

# --- 写入 result.md ---
markdown_file_path = os.path.join(output_dir, "result.md")
try:
    with open(markdown_file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(markdown_lines))
    print(f"[3/3] Markdown 汇总报告已成功写入: {markdown_file_path}")
except Exception as e:
    print(f"\n写入 result.md 文件时出错: {e}")

print("\n分析完成！")