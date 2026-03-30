import pandas as pd
import numpy as np
import pandapower as pp
import os
import sys
from tqdm import tqdm

# ==========================================
# 1. 配置与路径管理 (适配 config.py)
# ==========================================
try:
    from config import DATASET_PATH
    # 尝试从 config 导入扩充后的路径，如果没有定义则自动生成
    try:
        from config import AUGMENTED_DATASET_PATH
        OUTPUT_FILE = AUGMENTED_DATASET_PATH
    except ImportError:
        # 如果 config.py 没定义输出路径，则基于输入路径自动生成后缀
        base, ext = os.path.splitext(DATASET_PATH)
        OUTPUT_FILE = f"{base}_augmented_phys{ext}"
except ImportError:
    print("错误: 无法找到 config.py。请确保脚本在项目根目录运行，且 config.py 存在。")
    sys.exit(1)

# 检查输入文件是否存在
if not os.path.exists(DATASET_PATH):
    print(f"错误: 输入数据文件不存在: {DATASET_PATH}")
    sys.exit(1)

print(f"📂 输入数据: {DATASET_PATH}")
print(f"💾 输出目标: {OUTPUT_FILE}")

# ==========================================
# 2. 扩充参数配置
# ==========================================
AUGMENT_FACTOR = 20  # 扩充倍数：960 -> ~19200
NOISE_LEVEL = 0.05   # 环境参数扰动 5%
# 电池动作采样范围 (-1.0 ~ 1.0)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
TAIL_RATIO = 0.30
SENSITIVITY_DELTA_ACTION = 0.05

# ==========================================
# 3. IEEE 33 物理仿真器类
# ==========================================
class IEEE33Simulator:
    def __init__(self):
        self.net = pp.create_empty_network()
        self._create_network()
        
        # 参数需与 microgrid_env_complex_v11.py 保持一致
        self.pv_rated_kw = np.array([100, 50, 50, 20, 20, 30]) * 1.0
        self.wind_rated_kw = np.array([50, 50, 100, 100, 50, 50]) * 1.0
        self.batt_power_mw = np.array([0.15, 0.1, 0.25, 0.2, 0.2]) # MW
        
        self.v_ci, self.v_r, self.v_co = 3.0, 12.0, 25.0

    def _create_network(self):
        # 建立拓扑
        self.bus_idx = [pp.create_bus(self.net, vn_kv=12.66) for _ in range(33)]
        pp.create_ext_grid(self.net, bus=self.bus_idx[0], vm_pu=1.0)
        
        for i in range(32):
            pp.create_line_from_parameters(
                self.net, from_bus=self.bus_idx[i], to_bus=self.bus_idx[i+1],
                length_km=1.0, r_ohm_per_km=0.0922, x_ohm_per_km=0.047,
                c_nf_per_km=0.0, max_i_ka=1.0)
        
        self.n_pv, self.n_wind, self.n_batt = 6, 6, 5
        
        # 创建组件
        self.pv_sgen_idx = [pp.create_sgen(self.net, bus=self.bus_idx[min(i, 32)], p_mw=0.0) for i in range(self.n_pv)]
        self.wind_sgen_idx = [pp.create_sgen(self.net, bus=self.bus_idx[min(i+6, 32)], p_mw=0.0) for i in range(self.n_wind)]
        self.batt_sgen_idx = [pp.create_sgen(self.net, bus=self.bus_idx[min(i+12, 32)], p_mw=0.0) for i in range(self.n_batt)]
        self.load_idx = [pp.create_load(self.net, bus=self.bus_idx[32], p_mw=0.1)]

    def calc_pv(self, G_raw, T_amb, rated_kw):
        # 物理公式 (与环境一致)
        G = np.clip(G_raw / 20.0, 0, 1000) 
        if G <= 0: return 0.0
        G_eff = G / 1000.0
        T_cell = T_amb + (G / 800.0) * 25.0
        temp_factor = max(0.0, 1 - 0.004 * (T_cell - 25.0))
        return max(rated_kw * G_eff * temp_factor, 0.0)

    def calc_wind(self, v, rated_kw):
        if v < self.v_ci: return 0.0
        if self.v_ci <= v < self.v_r:
            return rated_kw * ((v - self.v_ci) / (self.v_r - self.v_ci)) ** 3
        if self.v_r <= v < self.v_co:
            return rated_kw
        return 0.0

    def run_power_flow(self, load_kw, pv_p_kw, wind_p_kw, batt_p_kw):
        # 写入 Pandapower 模型
        # 注意单位转换 kW -> MW
        self.net.load.at[self.load_idx[0], "p_mw"] = load_kw / 1000.0
        
        # 发电单元注入功率，在 Pandapower 中 sgen p_mw 通常为负表示发电注入(视具体约定)，
        # 但在你的环境中使用了正值注入逻辑(p_mw=positive)，Pandapower 会将其视为负荷的相反数(负负荷=发电)
        # 这里为了与你的环境代码 microgrid_env_complex_v11.py 保持完全一致：
        # 环境代码: self.net.sgen.at[idx, "p_mw"] = val (正值)
        
        for i, idx in enumerate(self.pv_sgen_idx):
            self.net.sgen.at[idx, "p_mw"] = pv_p_kw[i] / 1000.0

        for i, idx in enumerate(self.wind_sgen_idx):
            self.net.sgen.at[idx, "p_mw"] = wind_p_kw[i] / 1000.0

        for i, idx in enumerate(self.batt_sgen_idx):
            # batt_p_kw 正值代表放电注入电网，负值代表充电
            self.net.sgen.at[idx, "p_mw"] = batt_p_kw[i] / 1000.0

        try:
            pp.runpp(self.net)
            return True
        except pp.LoadflowNotConverged:
            return False
        except Exception as e:
            return False

    def get_bus_voltages(self):
        if len(self.net.res_bus) == 0:
            return np.ones((len(self.bus_idx),), dtype=np.float32)
        return self.net.res_bus.vm_pu.to_numpy(dtype=np.float32)


def compute_tail_voltage(voltages, tail_ratio=TAIL_RATIO):
    voltages = np.asarray(voltages, dtype=np.float32).flatten()
    if voltages.size == 0:
        return 1.0
    tail_count = max(1, int(np.ceil(voltages.size * float(tail_ratio))))
    return float(np.mean(np.sort(voltages)[:tail_count]))


def compute_action_sensitivities(sim, load_kw, pv_p_kw, wind_p_kw):
    """
    在零电池动作的基准工作点附近，对每个电池动作做中心差分，
    得到 tail-voltage 对归一化动作 a_i in [-1, 1] 的局部灵敏度。
    """
    sensitivities = []
    for i in range(sim.n_batt):
        delta_action = float(SENSITIVITY_DELTA_ACTION)
        delta_kw = float(sim.batt_power_mw[i] * 1000.0 * delta_action)

        batt_p_plus = np.zeros(sim.n_batt, dtype=np.float32)
        batt_p_minus = np.zeros(sim.n_batt, dtype=np.float32)
        batt_p_plus[i] = delta_kw
        batt_p_minus[i] = -delta_kw

        ok_plus = sim.run_power_flow(load_kw, pv_p_kw, wind_p_kw, batt_p_plus)
        if ok_plus:
            vtail_plus = compute_tail_voltage(sim.get_bus_voltages())
        else:
            vtail_plus = np.nan

        ok_minus = sim.run_power_flow(load_kw, pv_p_kw, wind_p_kw, batt_p_minus)
        if ok_minus:
            vtail_minus = compute_tail_voltage(sim.get_bus_voltages())
        else:
            vtail_minus = np.nan

        if np.isfinite(vtail_plus) and np.isfinite(vtail_minus):
            sensitivity = (vtail_plus - vtail_minus) / (2.0 * delta_action)
        else:
            sensitivity = 0.0
        sensitivities.append(float(sensitivity))

    return np.asarray(sensitivities, dtype=np.float32)

# ==========================================
# 4. 主执行逻辑
# ==========================================
def main():
    sim = IEEE33Simulator()
    df = pd.read_csv(DATASET_PATH)
    
    # 自动识别光照列
    irr_col = None
    for c in ["solar_irradiance", "irradiance", "G"]:
        if c in df.columns:
            irr_col = c
            break
    if irr_col is None:
        print("警告: 未找到光照列，默认光照为 0")
    
    new_rows = []
    
    print(f"开始扩充... (原始数据: {len(df)} 条, 倍数: {AUGMENT_FACTOR}x)")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 基础数据提取
        base_load = row.get("grid_load_demand", 800.0)
        base_temp = row.get("temperature", 25.0)
        base_wind = row.get("wind_speed", 5.0)
        base_irr = row.get(irr_col, 0.0) if irr_col else 0.0
        
        for _ in range(AUGMENT_FACTOR):
            aug_row = row.copy()
            
            # --- 1. 环境扰动 ---
            aug_load = max(0, base_load * np.random.normal(1, NOISE_LEVEL))
            aug_temp = base_temp + np.random.normal(0, 2.0)
            aug_wind = max(0, base_wind * np.random.normal(1, NOISE_LEVEL))
            aug_irr = max(0, base_irr * np.random.normal(1, NOISE_LEVEL))
            
            # --- 2. 动作采样 (随机生成电池策略) ---
            # 这对于 PySR 寻找控制规律至关重要
            batt_actions = [] # 记录动作值 (-1 ~ 1)
            batt_p_kw = []    # 记录实际功率 (kW)
            
            for i in range(sim.n_batt):
                act = np.random.uniform(ACTION_LOW, ACTION_HIGH)
                batt_actions.append(act)
                # 动作 -> 功率: 假设 action 1.0 对应最大功率放电
                p_max_kw = sim.batt_power_mw[i] * 1000.0
                p_real = act * p_max_kw
                batt_p_kw.append(p_real)
            
            # --- 3. 物理计算 ---
            pv_p_kw = [sim.calc_pv(aug_irr, aug_temp, cap) for cap in sim.pv_rated_kw]
            wind_p_kw = [sim.calc_wind(aug_wind, cap) for cap in sim.wind_rated_kw]

            # --- 4. 运行 Pandapower ---
            success = sim.run_power_flow(aug_load, pv_p_kw, wind_p_kw, batt_p_kw)

            if success:
                base_res_bus = sim.net.res_bus.copy()
                v_min = float(base_res_bus.vm_pu.min())
                v_max = float(base_res_bus.vm_pu.max())
                action_sensitivities = compute_action_sensitivities(sim, aug_load, pv_p_kw, wind_p_kw)

                # --- 5. 保存结果 ---
                # 更新输入特征
                aug_row["grid_load_demand"] = aug_load
                aug_row["temperature"] = aug_temp
                aug_row["wind_speed"] = aug_wind
                if irr_col:
                    aug_row[irr_col] = aug_irr

                # 记录动作 (这是 PySR 的重要输入特征)
                for i, act in enumerate(batt_actions):
                    aug_row[f"action_batt_{i}"] = act
                    aug_row[f"sens_action_batt_{i}"] = float(action_sensitivities[i])

                # 显式计算 Gamma (电压安全裕度)
                # 定义: 距离安全边界(0.95, 1.05)的最小距离
                # 正值表示安全，负值表示越限
                # 公式: min(V - 0.95, 1.05 - V)
                gamma_val = min(v_min - 0.95, 1.05 - v_max)

                aug_row["elec_vmin"] = v_min
                aug_row["elec_vmax"] = v_max
                aug_row["gamma"] = gamma_val

                new_rows.append(aug_row)

    # 保存
    df_aug = pd.DataFrame(new_rows)
    # 这里我们只保存扩充数据，或者你可以选择 pd.concat([df, df_aug])
    # 为了 PySR 训练纯净，通常建议混合原始数据和扩充数据，但要确保列对齐
    # 由于原始数据没有 action_batt 列，我们这里只保存扩充后的数据用于 PySR
    # 或者给原始数据填补默认 action (例如 0)
    
    df_aug.to_csv(OUTPUT_FILE, index=False)
    print(f"成功! 已生成 {len(df_aug)} 条扩充数据。")
    print(f"文件保存在: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
