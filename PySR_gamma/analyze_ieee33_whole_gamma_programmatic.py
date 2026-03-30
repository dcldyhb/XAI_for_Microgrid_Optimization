import sys
import os
import pandas as pd
import numpy as np
from pysr import PySRRegressor
import warnings

# ==========================================
# 0. 路径与环境配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# 导入配置
try:
    from config import TRAIN_PATH, TEST_PATH, PYSR_OUTPUT_DIR, DEVICE
except ImportError:
    print("[WARNING] 无法导入 config.py，使用默认设置。")
    DEVICE = 'cpu'
    PYSR_OUTPUT_DIR = os.path.join(current_dir, 'output_vector')
    TRAIN_PATH = None 
    TEST_PATH = None

warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs(PYSR_OUTPUT_DIR, exist_ok=True)
print(f"[INFO] PySR 输出目录: {PYSR_OUTPUT_DIR}")

# ==========================================
# 1. 数据加载与预处理
# ==========================================
if not TRAIN_PATH or not os.path.exists(TRAIN_PATH):
    print(f"[ERROR] 找不到训练数据文件: {TRAIN_PATH}")
    sys.exit(1)

print(f"[INFO] 正在加载训练集: {TRAIN_PATH}")
df_train = pd.read_csv(TRAIN_PATH)
print(f"[INFO] 正在加载测试集: {TEST_PATH}")
df_test = pd.read_csv(TEST_PATH)

df_train.replace({False: 0, True: 1}, inplace=True)
df_test.replace({False: 0, True: 1}, inplace=True)

# ==========================================
# 2. 特征与目标准备
# ==========================================

# --- A. 确定目标 (y) ---
# 优先使用和当前 5 维电池动作一一对应的 signed 动作灵敏度标签。
preferred_sensitivity_targets = [f"sens_action_batt_{i}" for i in range(5)]
legacy_ess_pairs = [(f"yhat_ess{i}_up", f"yhat_ess{i}_down") for i in range(1, 6)]

if all(col in df_train.columns for col in preferred_sensitivity_targets):
    gamma_target_columns = preferred_sensitivity_targets
    print("[INFO] 发现显式动作灵敏度标签，直接用于 PySR 拟合。")
else:
    missing_legacy = [
        (up_col, down_col)
        for up_col, down_col in legacy_ess_pairs
        if up_col not in df_train.columns or down_col not in df_train.columns
    ]
    if missing_legacy:
        print("[ERROR] 未找到动作灵敏度标签，也无法从 yhat_ess*_up/down 回退构造。")
        sys.exit(1)

    print("[INFO] 未找到显式动作灵敏度标签，使用 yhat_ess*_up - yhat_ess*_down 构造 5 维 signed 动作灵敏度。")
    for idx, (up_col, down_col) in enumerate(legacy_ess_pairs):
        target_col = preferred_sensitivity_targets[idx]
        df_train[target_col] = (
            pd.to_numeric(df_train[up_col], errors='coerce').fillna(0.0)
            - pd.to_numeric(df_train[down_col], errors='coerce').fillna(0.0)
        )
        df_test[target_col] = (
            pd.to_numeric(df_test[up_col], errors='coerce').fillna(0.0)
            - pd.to_numeric(df_test[down_col], errors='coerce').fillna(0.0)
        )
    gamma_target_columns = preferred_sensitivity_targets

print(f"[INFO] 分析目标 ({len(gamma_target_columns)} 个): {gamma_target_columns}")

# --- B. 确定特征 (X) ---
# 排除列表：动作、时间、类型、以及 gamma 本身
user_excluded_features = [
    'action_batt_0', 
    'action_batt_1', 
    'action_batt_2', 
    'action_batt_3', 
    'action_batt_4', 
    'gamma',       # [重要] 排除 gamma，既不当y也不当X
    'timestamp', 
    'regime'
]

# 这些变量更像结果告警，会让符号回归退化成“违规检测器”，
# 不利于学习“状态 -> 动作灵敏度”的映射。
leakage_excluded_features = [
    'elec_any_violate',
    'elec_violate_v',
    'elec_violate_line',
]

# 合并所有需要排除的列 (特征 = 所有列 - 排除列 - 目标列)
legacy_target_columns = [col for col in df_train.columns if col.startswith('yhat_')]
cols_to_drop = list(set(
    user_excluded_features
    + leakage_excluded_features
    + gamma_target_columns
    + legacy_target_columns
))

# 构建特征矩阵 X
X_train_df = df_train.drop(columns=cols_to_drop, errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0)
X_test_df = df_test.drop(columns=cols_to_drop, errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0)

X_train = X_train_df.to_numpy()
X_test = X_test_df.to_numpy()
feature_names = X_train_df.columns.to_list()

print(f"[INFO] 使用特征 ({len(feature_names)} 个): {feature_names}")
print(f"[INFO] 样本数量: {len(X_train)} (训练) / {len(X_test)} (测试)")

# ==========================================
# 3. 循环建模 (PySR)
# ==========================================
models = {}
test_scores = {}

for target_col in gamma_target_columns:
    print("-" * 60)
    print(f"[INFO] 正在分析目标: {target_col}")
    print("-" * 60)
    
    y_train = df_train[target_col].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
    y_test = df_test[target_col].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
    
    model = PySRRegressor(
        niterations=50,
        binary_operators=["+", "*", "/", "-"],
        # 动作灵敏度需要保留正负方向，避免 abs/sqrt 过度抹掉符号信息
        unary_operators=["square", "cube"],
        # 防止公式过度嵌套膨胀
        nested_constraints={
            "square": {"square": 0, "cube": 0},
            "cube": {"square": 0, "cube": 0},
        },
        maxsize=25,       # 限制公式长度
        parsimony=0.001,  # 复杂度惩罚
        model_selection="best",
        temp_equation_file=os.path.join(PYSR_OUTPUT_DIR, f"temp_hof_{target_col}.csv"),
        verbosity=1
    )
    
    model.fit(X_train, y_train)
    models[target_col] = model
    
    try:
        score = model.score(X_test, y_test)
        test_scores[target_col] = score
        print(f"[RESULT] 最佳公式: {model.get_best()['equation']}")
        print(f"[RESULT] 测试集 R2 得分: {score:.4f}")
    except:
        print("[WARNING] 无法获取公式或评分。")

# ==========================================
# 4. 生成输出文件
# ==========================================
print("\n" + "#"*60)
print("[INFO] 正在生成输出文件...")
print("#"*60)

# --- 准备 Markdown 和 Python 代码头 ---
gamma_formulas_data = []
markdown_lines = [
    "# PySR 信号公式汇总\n",
    "## 1. 输入特征映射\n",
    "| 符号 ($x_i$) | 物理含义 |\n| :--- | :--- |",
]

for i, name in enumerate(feature_names):
    markdown_lines.append(f"| $x_{{{i}}}$ | `{name}` |")
markdown_lines.append("\n## 2. 挖掘结果\n")

python_code_lines = [
    "# This file is auto-generated by PySR.",
    "import numpy as np", 
    "import torch",
    "import sys",
    "import os",
    "",
    "# --- Math Safety Functions ---",
    "def safe_sqrt(x): return np.sqrt(np.abs(x))",
    "def safe_div(a, b): return np.divide(a, b + 1e-9)",
    "def safe_log(x): return np.log(np.abs(x) + 1e-9)",
    "def cube(x): return np.power(x, 3)",
    "",
    "# --- Path Configuration ---",
    "current_dir = os.path.dirname(os.path.abspath(__file__))",
    "project_root = os.path.dirname(os.path.dirname(current_dir))",
    "if project_root not in sys.path: sys.path.append(project_root)",
    "try: from config import DEVICE",
    "except ImportError: DEVICE = torch.device('cpu')",
    "",
    "class GeneratedGammaCalculator:",
    "    def __init__(self):", 
    "        np.seterr(all='ignore')",
    f"        self.feature_names = {feature_names!r}",
    f"        self.gamma_signals = {gamma_target_columns!r}\n",
    "    def compute(self, state):",
    "        # Unpack features based on training configuration"
]

# Python 解包逻辑
for i, name in enumerate(feature_names):
    python_code_lines.append(f"        x{i} = state[{i}] # {name}") 
python_code_lines.append("\n        # Formulas")

# --- 填充公式 ---
gamma_component_vars = []

for i, target_col in enumerate(gamma_target_columns):
    model = models.get(target_col)
    var_name = f"gamma_{i}"
    gamma_component_vars.append(var_name)
    score = test_scores.get(target_col, -999.0)
    
    try:
        best_eq = model.get_best()
        eq_str = best_eq["equation"]
        latex_eq = model.latex(index=best_eq.name, precision=4)
        
        # 安全函数替换
        py_eq = eq_str.replace('sin(', 'np.sin(')\
                      .replace('cos(', 'np.cos(')\
                      .replace('exp(', 'np.exp(')\
                      .replace('log(', 'safe_log(')\
                      .replace('sqrt(', 'safe_sqrt(')\
                      .replace('abs(', 'np.abs(')\
                      .replace('square(', 'np.square(')
        
        has_result = True
    except:
        eq_str = "0.0"
        latex_eq = "N/A"
        py_eq = "0.0"
        has_result = False

    # 数据记录
    gamma_formulas_data.append({
        'signal': target_col, 
        'r2_score': score,
        'equation': eq_str, 
        'latex': latex_eq
    })

    python_code_lines.append(f"        {var_name} = {py_eq}")

    markdown_lines.append(f"### 目标: `{target_col}`")
    markdown_lines.append(f"- **R2 Score**: {score:.4f}")
    if has_result:
        markdown_lines.append(f"$$\n{latex_eq}\n$$")
    markdown_lines.append("---\n")

# --- 写入文件 ---
python_code_lines.extend([
    "\n        # Output Assembly",
    f"        gamma_values = [{', '.join(gamma_component_vars)}]",
    "        # 数值清洗: 替换 NaN/Inf 为 0",
    "        gamma_array = np.nan_to_num(np.array(gamma_values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)",
    "        return torch.from_numpy(gamma_array).to(DEVICE).unsqueeze(0)"
])

# 1. Python Module
py_path = os.path.join(PYSR_OUTPUT_DIR, "gamma_calculator.py")
with open(py_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(python_code_lines))

# 2. CSV Summary
csv_path = os.path.join(PYSR_OUTPUT_DIR, "gamma_formulas.csv")
pd.DataFrame(gamma_formulas_data).to_csv(csv_path, index=False)

# 3. Markdown Report
md_path = os.path.join(PYSR_OUTPUT_DIR, "result.md")
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(markdown_lines))

print(f"[SUCCESS] 输出文件已保存至: {PYSR_OUTPUT_DIR}")
print("[INFO] 分析结束")
