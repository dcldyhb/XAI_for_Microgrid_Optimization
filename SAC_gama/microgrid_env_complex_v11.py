# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 16:00:44 2025

@author: 12392
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 09:02:39 2025

@author: 12392
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:58:54 2025

@author: 12392
"""

# -*- coding: utf-8 -*-
"""
microgrid_env_complex_v8.py

改进内容：
- 自动检测太阳辐照度单位（W/m² 或 kW/m²），自动缩放。
- 光伏模型采用按额定容量比例模型：P = rated_kw * (G_eff / 1.0) * η * 温度修正。
- 保留 SOC 严格约束与能量守恒逻辑。
- 修复风机与储能方向一致性。
- 增强 info 输出以便绘图。
- 加入潮流计算并获取各节点电压
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
import pandapower as pp


class IEEE33Env(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, data: pd.DataFrame):
        super(IEEE33Env, self).__init__()

        # ========== 数据 ==========
        self.data = data.copy().reset_index(drop=True)
        if "timestamp" in self.data.columns:
            self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], errors="coerce")

        # ========== 环境时序 ==========
        self.timestep = 1.0  # 1小时
        self._max_episode_steps = 24
        
        self.hours_per_day = 24
        self.num_days = len(self.data) // self.hours_per_day
        self.episode_counter = 0  # 记录第几天

        # ========== 拓扑结构 ==========
        self.n_batt = 5
        self.n_pv = 6
        self.n_wind = 6
        self.battery_nodes = list(range(self.n_batt))
        self.pv_nodes = list(range(self.n_pv))
        self.wind_nodes = list(range(self.n_wind))

        # ========== 储能参数 ==========
        self.batt_capacity_mwh = np.array([0.3, 0.2, 0.5, 0.4, 0.4])  # MWh
        self.batt_power_mw = np.array([0.15, 0.1, 0.25, 0.2, 0.2])
        self.batt_capacity = self.batt_capacity_mwh * 1000.0  # kWh
        self.max_batt_power = self.batt_power_mw * 1000.0     # kW
        self.eta_ch = 0.95
        self.eta_dis = 0.95
        self.current_soc = np.ones(self.n_batt, dtype=np.float32) * 0.5

        # ========== 风机参数 ==========
        self.v_ci = 3.0
        self.v_r = 12.0
        self.v_co = 25.0
        self.wind_rated_kw = np.array([50, 50, 100, 100, 50, 50]) * 1.0

        # ========== 光伏参数 ==========
        self.pv_rated_kw = np.array([100, 50, 50, 20, 20, 30]) * 1.0
        self.eta_pv = 0.18
        self.beta = 0.004
        self.T_ref = 25.0

        # ===== 自动检测光照强度单位 =====
        col_candidates = ["solar_irradiance", "irradiance", "G"]
        found_col = None
        for c in col_candidates:
            if c in self.data.columns:
                found_col = c
                break
        if found_col is None:
            self.G_scale = 1.0
        else:
            med = float(np.nanmedian(self.data[found_col].fillna(0.0)))
            if med > 50.0:  # 单位是 W/m²
                self.G_scale = 1.0 / 1000.0
            else:
                self.G_scale = 1.0
        self._irr_col = found_col

        # ========== 电价结构 ==========
        base_price = np.array([
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.65, 0.65,
            0.45, 0.45, 0.45,
            0.35, 0.35,
            0.70, 0.70, 0.70, 0.70, 0.70,
            0.80, 0.80, 0.80,
            0.40, 0.40,
            0.30
        ], dtype=np.float32)
        self.price_modes = {
            "base": base_price,
            "summer": base_price + np.array([
                0.0,0.0,0.0,0.0,0.0,0.0,
                0.02,0.04,0.06,0.08,0.10,0.12,
                0.14,0.16,0.18,0.16,0.12,0.08,
                0.06,0.04,0.02,0.0,0.0,0.0
            ], dtype=np.float32),
            "weekend": base_price * np.array([
                0.9,0.9,0.9,0.9,0.9,0.9,
                0.95,0.95,1.0,1.0,1.0,1.0,
                1.0,1.0,1.0,0.95,0.95,0.9,
                0.9,0.9,0.9,0.9,0.9,0.9
            ], dtype=np.float32)
        }
        self.price_mode = "base"
        self.price_vol_factor = 1.5
        self.price_daily_jitter = 0.05
        self.price_per_hour = self._build_price_from_mode(self.price_mode, self.price_vol_factor)

        # ========== 经济与约束参数 ==========
        self.sell_price_ratio = 0.8
        self.arbitrage_weight = 1000.0
        self.balance_lambda = 0.05
        self.lambda_cyc = 2e-3
        self.lambda_sw = 1e-3
        self.lambda_edge = 0.2
        self.lambda_soc_violation = 2000.0
        self.vmin, self.vmax = 0.95, 1.05

        # ========== 电网模型 ==========
        self.net = pp.create_empty_network()
        self._create_simple_network()

        # ========== 空间定义 ==========
        obs_dim = 3 * self.n_batt + self.n_pv + self.n_wind + 2 + len(self.net.bus) + 9
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_batt,), dtype=np.float32)

        # ========== 时间索引 ==========
        ts = self.data["timestamp"] if "timestamp" in self.data.columns else pd.Series([pd.NaT]*len(self.data))
        is_midnight = (ts.dt.hour == 0) & (ts.dt.minute == 0) if not ts.isnull().all() else pd.Series([False]*len(self.data))
        idxs = np.flatnonzero(is_midnight.to_numpy()) if len(is_midnight) == len(self.data) else np.arange(0, len(self.data), 24)
        self.day_start_indices = [int(i) for i in idxs if i + 24 <= len(self.data)]
        if len(self.day_start_indices) == 0:
            self.day_start_indices = list(range(0, max(1, len(self.data) - 24 + 1)))
        self.episode_start_idx = int(self.day_start_indices[0])
        self.current_step = self.episode_start_idx
        self.episode_step_count = 0

        self.np_random = np.random.RandomState()

    # ===== Helper =====
    def _build_price_from_mode(self, mode, vol_factor=1.0):
        base = self.price_modes.get(mode, self.price_modes["base"]).astype(np.float32)
        mean_base = float(np.mean(base))
        return mean_base + (base - mean_base) * float(vol_factor)

    def _create_simple_network(self):
        self.bus_idx = [pp.create_bus(self.net, vn_kv=12.66) for _ in range(33)]
        pp.create_ext_grid(self.net, bus=self.bus_idx[0], vm_pu=1.0)
        for i in range(32):
            pp.create_line_from_parameters(
                self.net, from_bus=self.bus_idx[i], to_bus=self.bus_idx[i+1],
                length_km=1.0, r_ohm_per_km=0.0922, x_ohm_per_km=0.047,
                c_nf_per_km=0.0, max_i_ka=1.0)
        self.pv_sgen_idx = [pp.create_sgen(self.net, bus=self.bus_idx[min(i, 32)], p_mw=0.0) for i in range(self.n_pv)]
        self.wind_sgen_idx = [pp.create_sgen(self.net, bus=self.bus_idx[min(i+6, 32)], p_mw=0.0) for i in range(self.n_wind)]
        self.batt_sgen_idx = [pp.create_sgen(self.net, bus=self.bus_idx[min(i+12, 32)], p_mw=0.0) for i in range(self.n_batt)]
        self.load_idx = [pp.create_load(self.net, bus=self.bus_idx[32], p_mw=0.1)]

    def _run_power_flow(self, pv_p, wind_p, batt_p_list, load_kW):
        """执行潮流计算并返回节点电压"""
        try:
            # 更新光伏发电功率
            for i, sgen_idx in enumerate(self.pv_sgen_idx):
                self.net.sgen.at[sgen_idx, "p_mw"] = pv_p[i] / 1000.0  # kW -> MW
            
            # 更新风电发电功率
            for i, sgen_idx in enumerate(self.wind_sgen_idx):
                self.net.sgen.at[sgen_idx, "p_mw"] = wind_p[i] / 1000.0  # kW -> MW
            
            # 更新储能功率（正值放电，负值充电）
            for i, sgen_idx in enumerate(self.batt_sgen_idx):
                self.net.sgen.at[sgen_idx, "p_mw"] = batt_p_list[i] / 1000.0  # kW -> MW
            
            # 更新负荷功率
            for load_idx in self.load_idx:
                self.net.load.at[load_idx, "p_mw"] = load_kW / 1000.0  # kW -> MW
            
            # 执行潮流计算
            pp.runpp(self.net)
            
            # 获取节点电压（标幺值）
            voltages = self.net.res_bus.vm_pu.values.astype(np.float32)
            
            # 获取线路负载率等信息（可选）
            line_loading = self.net.res_line.loading_percent.values.astype(np.float32) if len(self.net.line) > 0 else np.array([])
            
            return voltages, line_loading
            
        except Exception as e:
            print(f"潮流计算失败: {e}")
            # 返回默认电压值
            return np.ones(len(self.net.bus), dtype=np.float32), np.array([])

    # ===== Gym API =====
    def seed(self, seed=None):
        if seed is not None:
            self.np_random.seed(seed)
        return [seed]

    def reset(self):
        self.episode_counter += 1  # ✅ 每次reset时自增1
        #self.episode_start_idx = int(self.np_random.choice(self.day_start_indices))
        # 每集固定从 0 点开始（保证 0~23 小时完整覆盖）
        self.episode_start_idx = (self.episode_counter % self.num_days) * 24

        self.current_step = self.episode_start_idx
        # 👇 验证当天光照数据（打印24小时序列）
        #print("[DEBUG] 今日光照（W/m2）:", 
              #self.data["solar_irradiance"].iloc[self.episode_start_idx:self.episode_start_idx+24].to_list())

        self.episode_step_count = 0
        self.current_soc = self.np_random.uniform(0.2, 0.8, size=self.n_batt).astype(np.float32)
        self.prev_ctrl = np.zeros(self.n_batt, dtype=np.float32)

        self.price_mode = str(self.np_random.choice(list(self.price_modes.keys())))
        self.price_per_hour = self._build_price_from_mode(self.price_mode, self.price_vol_factor)
        jitter = float(self.np_random.uniform(-self.price_daily_jitter, self.price_daily_jitter))
        mean_p = float(np.mean(self.price_per_hour))
        self.price_per_hour = mean_p + (self.price_per_hour - mean_p) * (1.0 + jitter)
        return self._get_state(self.current_step)

    def step(self, action):
        if self.current_step >= len(self.data):
            self.current_step = 0
        row = self.data.iloc[self.current_step]
        wind_speed = float(row.get("wind_speed", 5.0))
        G_raw = float(row.get(self._irr_col, 0.0)) if self._irr_col is not None else 0.0
        T_amb = float(row.get("temperature", self.T_ref))
        load_kW = float(row.get("grid_load_demand", row.get("load_kW", 800.0)))
        
        #if self.current_step % 24 == 0:  # 每天首小时打印一次
            #print(f"[DEBUG-step] step={self.current_step}, G_raw={G_raw}, T_amb={T_amb}, hour={row.get('hour_of_day', -1)}")

        # --- PV & Wind ---
        pv_p = np.array([self._calc_pv(G_raw, T_amb, float(self.pv_rated_kw[i])) for i in range(self.n_pv)], dtype=np.float32)
        wind_p = np.array([self._calc_wind(wind_speed, float(self.wind_rated_kw[i])) for i in range(self.n_wind)], dtype=np.float32)

        # --- Battery ---
        a = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
        p_dis_des = np.maximum(a, 0.0) * self.max_batt_power
        p_ch_des = np.maximum(-a, 0.0) * self.max_batt_power

        p_dis_list, p_ch_list, batt_p_list = [], [], []
        for i in range(self.n_batt):
            soc = float(self.current_soc[i])
            Cap = float(self.batt_capacity[i])
            Pmax = float(self.max_batt_power[i])

            max_dis_energy = soc * Cap
            p_dis_max = (max_dis_energy * self.eta_dis) / self.timestep
            p_dis = min(p_dis_des[i], p_dis_max, Pmax)

            max_ch_energy = (1.0 - soc) * Cap
            p_ch_max = (max_ch_energy) / (self.eta_ch * self.timestep)
            p_ch = min(p_ch_des[i], p_ch_max, Pmax)

            if soc <= 1e-8 and p_dis > 0:
                p_dis = 0.0
            if soc >= 1.0 - 1e-8 and p_ch > 0:
                p_ch = 0.0

            delta_e = (p_ch * self.eta_ch - p_dis / self.eta_dis) * self.timestep
            new_soc = float(np.clip(soc + delta_e / Cap, 0.0, 1.0))
            self.current_soc[i] = new_soc
            p_inj = p_dis - p_ch
            p_dis_list.append(p_dis)
            p_ch_list.append(p_ch)
            batt_p_list.append(p_inj)

        batt_p_list = np.array(batt_p_list, dtype=np.float32)

        # --- 潮流计算 ---
        voltages, line_loading = self._run_power_flow(pv_p, wind_p, batt_p_list, load_kW)
        
        # 电压越限惩罚
        voltage_violation = np.sum(np.maximum(0, self.vmin - voltages) + np.maximum(0, voltages - self.vmax))
        voltage_penalty = self.lambda_edge * voltage_violation

        # --- 电网功率平衡 ---
        total_gen = np.sum(pv_p) + np.sum(wind_p) + np.sum(batt_p_list)
        grid_kW = load_kW - total_gen

        # --- 经济 ---
        hour = int(row.get("hour_of_day", self.current_step % 24))
        price = float(self.price_per_hour[hour])
        sell_price = self.sell_price_ratio * price
        import_kW = max(grid_kW, 0.0)
        export_kW = max(-grid_kW, 0.0)
        energy_cost = import_kW * price - export_kW * sell_price

        avg_price = float(np.mean(self.price_per_hour))
        price_signal = (price - avg_price) / (avg_price + 1e-9)
        batt_arb_reward = float(np.sum(a * price_signal) * self.arbitrage_weight)

        reward = -energy_cost + batt_arb_reward - voltage_penalty
        mismatch = total_gen + grid_kW - load_kW

        info = {
            "p_ch": np.array(p_ch_list),
            "p_dis": np.array(p_dis_list),
            "batt_p": np.array(batt_p_list),
            "pv_p": np.array(pv_p),
            "wind_p": np.array(wind_p),
            "grid_kW": float(grid_kW),
            "price": float(price),
            "mismatch": float(mismatch),
            "soc": np.array(self.current_soc),
            "load_kW": float(load_kW),
            "voltages": np.array(voltages),  # 添加电压信息
            "voltage_violation": float(voltage_violation),
            "line_loading": np.array(line_loading) if len(line_loading) > 0 else np.array([])
        }

        self.prev_ctrl = np.array(a)
        self.current_step += 1
        self.episode_step_count += 1
        done = self.episode_step_count >= self._max_episode_steps
        state = self._get_state(self.current_step)
        return state, float(reward), bool(done), info

    def _get_state(self, step_idx):
        idx = step_idx % len(self.data)
        row = self.data.iloc[idx]
        G_raw = float(row.get(self._irr_col, 0.0)) if self._irr_col else 0.0
        T = float(row.get("temperature", self.T_ref))
        wind_speed = float(row.get("wind_speed", 5.0))
        load_kW = float(row.get("grid_load_demand", 800.0))
        pv_p = np.array([self._calc_pv(G_raw, T, float(self.pv_rated_kw[i])) for i in range(self.n_pv)], dtype=np.float32)
        wind_p = np.array([self._calc_wind(wind_speed, float(self.wind_rated_kw[i])) for i in range(self.n_wind)], dtype=np.float32)
        
        # 获取当前储能状态（假设无动作时的功率）
        batt_p_list = np.zeros(self.n_batt, dtype=np.float32)
        voltages, _ = self._run_power_flow(pv_p, wind_p, batt_p_list, load_kW)
        
        grid_kW_pf = load_kW - (np.sum(pv_p) + np.sum(wind_p))
        ctx = self._time_ctx(idx)
        state = np.concatenate([
            np.concatenate([self.current_soc, np.zeros_like(self.current_soc), np.zeros_like(self.current_soc)]),
            np.array(list(wind_p) + list(pv_p) + [load_kW, grid_kW_pf], dtype=np.float32),
            voltages, ctx
        ], dtype=np.float32)
        return state

    def _time_ctx(self, step_idx):
        hour = int(self.data.iloc[step_idx % len(self.data)].get("hour_of_day", step_idx % 24))
        dow = int(self.data.iloc[step_idx % len(self.data)].get("day_of_week", 0))
        hour_ang = 2 * np.pi * hour / 24
        day_ang = 2 * np.pi * dow / 7
        return np.array([np.sin(hour_ang), np.cos(hour_ang), float(self.price_per_hour[hour]),
                         np.sin(day_ang), np.cos(day_ang), 1.0 if dow >= 5 else 0.0, 0, 0, 0], dtype=np.float32)

    def _calc_wind(self, v, rated_kw):
        if v < self.v_ci: return 0.0
        if self.v_ci <= v < self.v_r:
            return rated_kw * ((v - self.v_ci) / (self.v_r - self.v_ci)) ** 3
        if self.v_r <= v < self.v_co:
            return rated_kw
        return 0.0

    def _calc_pv(self, G_raw, T_amb, rated_kw):
        # 将辐照度限定在物理合理范围
        G = np.clip(G_raw / 20.0, 0, 1000)  # 🔧 若你的数据约在0~20000，除以20
        if G <= 0:
            return 0.0
        G_eff = G / 1000.0
        T_cell = T_amb + (G / 800.0) * 25.0
        temp_factor = max(0.0, 1 - 0.004 * (T_cell - 25.0))
        P_out = rated_kw * G_eff * temp_factor
        return max(P_out, 0.0)



    def close(self):
        pass