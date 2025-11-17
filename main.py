import torch
import numpy as np

# --- 1. 从您的文件中导入必要的类 ---

# 从模型定义文件中导入策略网络
from XAI_for_Microgrid_Optimization.model_gamma import GaussianPolicy

# 从 PySR 自动生成的模块中导入 gamma 计算器
# 注意：确保 PySR_gamma/output_vector 目录在 Python 的搜索路径中，或者像下面这样调整路径
try:
    from PySR_gamma.output_vector.gamma_calculator import GeneratedGammaCalculator
except ImportError:
    print("错误: 无法导入 GeneratedGammaCalculator。")
    print("请确保 'PySR_gamma/output_vector/gamma_calculator.py' 文件存在且路径正确。")
    exit()

# --- 2. 设置模型和环境的参数 (使用示例值) ---

# 这些参数需要与您的实际强化学习环境相匹配
NUM_INPUTS = 22      # 状态空间的维度 (等于 PySR 使用的特征数量)
NUM_ACTIONS = 6      # 动作空间的维度 (请根据您的环境修改)
HIDDEN_DIM = 256     # 神经网络隐藏层的大小
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"使用的设备: {DEVICE}")

# --- 3. 初始化模型和 Gamma 计算器 ---

# 初始化策略网络 (Actor)
# 这是您 model_gamma.py 中的核心部分
policy = GaussianPolicy(NUM_INPUTS, NUM_ACTIONS, HIDDEN_DIM, action_space=None).to(DEVICE)

# 初始化我们自动生成的 Gamma 计算器
gamma_calculator = GeneratedGammaCalculator()

print("\n模型和 Gamma 计算器已成功初始化。")


# --- 4. 模拟一次环境交互 ---

# a) 创建一个模拟的当前状态 (state)
#    这通常由 `env.reset()` 或 `env.step()` 返回
#    这里的键必须与 PySR 训练时使用的特征名完全一致！
#    我从您的日志中复制了这些特征名。
current_state_dict = {
    'temperature_C': 35.0,
    'wind_speed_ms': 5.5,
    'illuminance_lux': 15000.0,
    'precipitation_mm': 0.5,
    'elec_vmin': 0.96,
    'elec_vmax': 1.0,
    'elec_vavg': 0.98,
    'elec_line_loading_max': 75.0,
    'elec_line_loading_top1': 74.0,
    'elec_line_loading_top2': 65.0,
    'elec_line_loading_top3': 50.0,
    'elec_vmax_nonslack': 1.0,
    'elec_line_i_ka_max': 0.18,
    'elec_p_loss_mw': 0.15,
    'elec_soc_1': 50.0,
    'elec_soc_2': 50.0,
    'elec_soc_3': 50.0,
    'elec_soc_4': 50.0,
    'elec_soc_5': 50.0,
    'elec_any_violate': 1,  # 假设有违规
    'elec_violate_v': 1,    # 假设有电压违规
    'elec_violate_line': 0
}
print(f"\n创建了一个模拟状态，例如温度为: {current_state_dict['temperature_C']}°C")

# b) 将状态字典转换为 PyTorch 张量，用于输入策略网络
#    注意：需要添加一个批次维度 (batch dimension)，所以形状是 [1, num_inputs]
state_values = list(current_state_dict.values())
state_tensor = torch.FloatTensor(state_values).unsqueeze(0).to(DEVICE)


# ==========================================================
# vvvvvv              【核心调用步骤】              vvvvvv
#
# 这就是将所有部分连接起来的地方
#
# ==========================================================

# c) 使用 Gamma 计算器，根据当前状态字典计算出 gamma 张量
print("正在使用 gamma_calculator.compute() 计算 gamma 向量...")
gamma_tensor = gamma_calculator.compute(current_state_dict).to(DEVICE)

# d) 将状态张量和 gamma 张量同时传入策略网络的 sample 方法
print("正在调用 policy.sample(state_tensor, gamma_tensor)...")
# 将模型设置为评估模式
policy.eval()
with torch.no_grad(): # 在推理时不需要计算梯度
    action, log_prob, mean = policy.sample(state_tensor, gamma_tensor)

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ==========================================================


# --- 5. 查看输出结果 ---

print("\n--- 调用完成，结果如下 ---")
print(f"Gamma 向量 (形状: {gamma_tensor.shape}):\n{gamma_tensor.cpu().numpy()}")
print(f"\n策略网络输出的动作 (Action) (形状: {action.shape}):\n{action.cpu().numpy()}")
print(f"该动作的对数概率 (Log Probability) (形状: {log_prob.shape}):\n{log_prob.cpu().numpy()}")