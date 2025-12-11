# -*- coding: utf-8 -*-
# main_sac_only.py (不依赖 PySR 的独立版本)
import argparse
import datetime
import itertools
import os
import warnings
import time
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import sys
import os
from config import DEVICE, DATASET_PATH, OUTPUTS_ROOT, FONT_SANS_SERIF

# [MODIFIED] 引入项目模块
from SAC_gama.replay_memory import ReplayMemory
from SAC_gama.sac import SAC
from SAC_gama.microgrid_env_complex_v11 import IEEE33Env

# --- 环境设置 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

matplotlib.rcParams["font.sans-serif"] = FONT_SANS_SERIF
matplotlib.rcParams["axes.unicode_minus"] = False

print(f"当前使用的计算设备: {DEVICE}")

# ========== 参数解析 ==========
parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Args")
parser.add_argument("--env-name", default="IEEE33Env")
parser.add_argument("--policy", default="Gaussian")
parser.add_argument("--eval", type=bool, default=True)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--lr", type=float, default=0.0003)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--automatic_entropy_tuning", type=bool, default=True)
parser.add_argument("--seed", type=int, default=123456)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=5000)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--updates_per_step", type=int, default=3)
parser.add_argument("--start_steps", type=int, default=200)
parser.add_argument("--target_update_interval", type=int, default=1)
parser.add_argument("--replay_size", type=int, default=1000000)
parser.add_argument("--eval_every", type=int, default=10)
parser.add_argument("--eval_episodes", type=int, default=1)
args = parser.parse_args()
args.device = DEVICE

# ========== 数据集读取 ==========
print(f"读取数据集: {DATASET_PATH}")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"未找到数据集文件: {DATASET_PATH}")
data = pd.read_csv(DATASET_PATH)

# ========== 随机种子 ==========
if args.seed is None or args.seed < 0:
    args.seed = (int(time.time() * 1e6) ^ os.getpid() ^ random.getrandbits(32)) & 0xFFFFFFFF
print(f"使用随机种子: {args.seed}")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# ========== 初始化环境与 Agent ==========
env = IEEE33Env(data)
env.seed(None)
env.G_scale = 1.0 / 1000.0

agent = SAC(env.observation_space.shape[0], env.action_space, args)
try:
    if hasattr(agent, "automatic_entropy_tuning"):
        agent.automatic_entropy_tuning = False
    if hasattr(agent, "alpha"):
        agent.alpha = torch.tensor(0.2, dtype=torch.float32).to(DEVICE)
except Exception as _e:
    print(f"警告: alpha 参数微调跳过: {_e}")

# ========== 输出目录配置 ==========
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"{timestamp}_{args.env_name}_{args.policy}_SAC_ONLY" # 文件夹名加后缀以区分

save_dir = os.path.join(OUTPUTS_ROOT, run_name)
os.makedirs(save_dir, exist_ok=True)
print(f"本次运行的所有结果将保存至: {save_dir}")

log_dir = os.path.join(save_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

memory = ReplayMemory(args.replay_size, args.seed)

# [MODIFIED] 只保留随机 gamma 生成函数
def generate_random_gamma(batch_size, n_heads):
    return torch.rand(batch_size, n_heads).to(DEVICE)

# ========== 训练与收集 info ==========
rewards = []
infos_all = []  

total_numsteps = 0
updates = 0

print("开始训练 (仅 SAC)...")

for i_episode in itertools.count(1):
    env.seed(args.seed + i_episode)
    state = env.reset()
    done = False
    episode_reward = 0.0
    episode_steps = 0
    infos = []

    while not done:
        # [MODIFIED] 始终使用随机 gamma
        gamma = generate_random_gamma(1, n_heads=10)

        if args.start_steps > total_numsteps:
            action = env.action_space.sample()
        else:
            action, log_prob, mean = agent.select_action(state, gamma=gamma, evaluate=False)

        next_state, reward, done, info = env.step(action)
        infos.append(info)
        
        episode_steps += 1
        total_numsteps += 1
        episode_reward += float(reward)

        state = np.asarray(state, dtype=np.float32).flatten()
        next_state = np.asarray(next_state, dtype=np.float32).flatten()
        action = np.asarray(action, dtype=np.float32).flatten()
        reward = float(reward)

        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        memory.push(state, action, reward / 100.0, next_state, mask)
        state = next_state

        if len(memory) > args.batch_size:
            for _ in range(args.updates_per_step):
                c1, c2, p, ent, alpha = agent.update_parameters(memory, args.batch_size, updates)
                writer.add_scalar("loss/critic_1", c1, updates)
                writer.add_scalar("loss/critic_2", c2, updates)
                writer.add_scalar("loss/policy", p, updates)
                updates += 1

    rewards.append(episode_reward)
    infos_all.append(infos)
    writer.add_scalar("reward/train", float(episode_reward), i_episode)
    print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, steps: {episode_steps}, reward: {round(episode_reward, 2)}")

    if total_numsteps >= args.num_steps:
        break

env.close()
writer.close()

# ==============================================================================
# 绘图与可视化 (基于 main_origin.py 逻辑，已集成保存功能)
# ==============================================================================

def plot_all_node_voltages(voltages, hours, env, save_dir=None):
    """
    绘制所有33个节点的电压变化曲线
    """
    try:
        fig, ax = plt.subplots(figsize=(16, 10))
        n_nodes = voltages.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, n_nodes))
        
        for node in range(n_nodes):
            alpha = 0.7 if node in [0, 10, 20, 32] else 0.4
            linewidth = 2.0 if node in [0, 10, 20, 32] else 1.0
            
            ax.plot(hours, voltages[:, node], 
                   color=colors[node], 
                   alpha=alpha, 
                   linewidth=linewidth,
                   label=f'Node {node}' if node in [0, 10, 20, 32] else "")
        
        ax.axhline(env.vmax, color="red", linestyle="--", linewidth=2, alpha=0.8, label='电压上限')
        ax.axhline(env.vmin, color="red", linestyle="--", linewidth=2, alpha=0.8, label='电压下限')
        
        ax.set_xlabel("时间 (小时)")
        ax.set_ylabel("电压 (p.u.)")
        ax.set_title("所有节点电压变化曲线")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                  norm=plt.Normalize(vmin=0, vmax=n_nodes-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('节点编号')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, "plot_all_nodes_voltage.png")
            plt.savefig(save_path, dpi=300)
            print(f"图片已保存: {save_path}")
            
        plt.show()
        plt.close()
        
    except Exception as e:
        print(f"绘制所有节点电压曲线时发生错误: {e}")
        import traceback
        traceback.print_exc()

def plot_voltage_heatmap(voltages, hours, env, save_dir=None):
    """
    绘制节点电压热力图，直观显示电压分布
    """
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        heatmap_data = voltages.T
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r', 
                      extent=[hours[0], hours[-1], 0, voltages.shape[1]-1],
                      vmin=env.vmin, vmax=env.vmax)
        
        ax.set_xlabel("时间 (小时)")
        ax.set_ylabel("节点编号")
        ax.set_title("节点电压热力图")
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('电压 (p.u.)')
        
        ax.set_yticks(range(voltages.shape[1]))
        ax.set_yticklabels([f'Node {i}' for i in range(voltages.shape[1])])
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, "plot_voltage_heatmap.png")
            plt.savefig(save_path, dpi=300)
            print(f"图片已保存: {save_path}")
            
        plt.show()
        plt.close()
        
    except Exception as e:
        print(f"绘制电压热力图时发生错误: {e}")
        import traceback
        traceback.print_exc()

def save_voltage_data(voltages, hours, env, save_dir=None):
    """
    保存24个时间点的各节点电压数据到CSV文件 (已修复Bug)
    """
    try:
        voltage_df = pd.DataFrame(voltages, 
                                 index=[f"Hour_{h}" for h in hours],
                                 columns=[f"Node_{i}" for i in range(len(env.net.bus))])
        
        voltage_df['Hour'] = hours
        voltage_df['Timestamp'] = pd.date_range(start='2025-01-01', periods=len(hours), freq='h')
        
        cols = ['Hour', 'Timestamp'] + [f"Node_{i}" for i in range(len(env.net.bus))]
        voltage_df = voltage_df[cols]
        
        if save_dir:
            filename = "node_voltages.csv"
            file_path = os.path.join(save_dir, filename)
            voltage_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"数据已保存到: {file_path}")

        print("\n电压统计信息:")
        min_val, max_val = np.min(voltages), np.max(voltages)
        print(f"   平均电压范围: {min_val:.4f} - {max_val:.4f} p.u.")
        print(f"   电压越限次数: {np.sum((voltages < env.vmin) | (voltages > env.vmax))}")
        
        min_hour_idx, min_node_idx = np.unravel_index(np.argmin(voltages), voltages.shape)
        max_hour_idx, max_node_idx = np.unravel_index(np.argmax(voltages), voltages.shape)
        
        print(f"   最低电压出现: 节点{min_node_idx} (值: {min_val:.4f} p.u.) at Hour {hours[min_hour_idx]}")
        print(f"   最高电压出现: 节点{max_node_idx} (值: {max_val:.4f} p.u.) at Hour {hours[max_hour_idx]}")
        
        return voltage_df
        
    except Exception as e:
        print(f"保存电压数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_voltage_analysis_report(infos, env, save_dir=None):
    """
    生成并保存详细的电压分析报告
    """
    try:
        voltages = np.array([info.get("voltages", np.zeros(len(env.net.bus))) for info in infos])
        hours = np.arange(len(infos))
        
        report_data = []
        for hour in hours:
            hour_voltages = voltages[hour]
            report_data.append({
                'Hour': hour,
                'Min_Voltage': np.min(hour_voltages),
                'Max_Voltage': np.max(hour_voltages),
                'Min_Node': np.argmin(hour_voltages),
                'Max_Node': np.argmax(hour_voltages),
                'Violation_Count': np.sum((hour_voltages < env.vmin) | (hour_voltages > env.vmax)),
                'Avg_Voltage': np.mean(hour_voltages)
            })
        
        report_df = pd.DataFrame(report_data)
        
        if save_dir:
            report_filename = "voltage_analysis_report.csv"
            file_path = os.path.join(save_dir, report_filename)
            report_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"分析报告已保存到: {file_path}")
        
        return report_df
        
    except Exception as e:
        print(f"生成电压分析报告时发生错误: {e}")
        return None

def plot_microgrid_power_from_info(infos, env, save_dir=None):
    """
    主绘图函数 (源自 main_origin.py)
    """
    try:
        # 提取信息
        n_ess, n_pv, n_wind = env.n_batt, env.n_pv, env.n_wind
        batt_p = np.array([info.get("batt_p", np.zeros(n_ess)) for info in infos])
        pv_p = np.array([info.get("pv_p", np.zeros(n_pv)) for info in infos])
        wind_p = np.array([info.get("wind_p", np.zeros(n_wind)) for info in infos])
        grid_p = np.array([info.get("grid_kW", 0.0) for info in infos])
        socs = np.array([info.get("soc", np.zeros(n_ess)) for info in infos])
        prices = np.array([info.get("price", 0.0) for info in infos])
        load_p = np.array([info.get("load_kW", 0.0) for info in infos])
        voltages = np.array([info.get("voltages", np.zeros(len(env.net.bus))) for info in infos])
        T = len(infos)
        hours = np.arange(T)

        total_ess_p = batt_p.sum(axis=1)
        total_pv = pv_p.sum(axis=1)
        total_wind = wind_p.sum(axis=1)
        total_gen = total_ess_p + total_pv + total_wind

        # 主图绘制 (四合一)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
        
        # 子图1: 功率堆叠
        bottom_pos, bottom_neg = np.zeros(T), np.zeros(T)
        ess_colors = ["#203864", "#305496", "#4472C4", "#5B9BD5", "#8EA9DB",
                      "#A9D18E", "#548235", "#BF9000", "#7F6000", "#7030A0"]

        for i in range(n_ess):
            dis = np.where(batt_p[:, i] > 0, batt_p[:, i], 0)
            chg = np.where(batt_p[:, i] < 0, batt_p[:, i], 0)
            color = ess_colors[i % len(ess_colors)]
            ax1.bar(hours, dis, bottom=bottom_pos, color=color, width=0.6, label=f"ESS{i+1}")
            ax1.bar(hours, chg, bottom=bottom_neg, color=color, width=0.6)
            bottom_pos += dis
            bottom_neg += chg

        ax1.bar(hours, total_wind, bottom=bottom_pos, color="#00B0F0", width=0.6, label="风电")
        ax1.bar(hours, total_pv, bottom=bottom_pos + total_wind, color="#FFD966", width=0.6, label="光伏")
        ax1.plot(hours, grid_p, color="red", linewidth=2.2, marker="x", label="并网功率 Grid Power")

        ax1b = ax1.twinx()
        ax1b.plot(hours, prices, color="black", linestyle="--", linewidth=2, label="电价 (RMB/kWh)")
        ax1b.set_ylabel("电价 (RMB/kWh)")

        ax1.axhline(0, color="k", linewidth=0.8)
        ax1.set_ylabel("功率 / kW")
        ax1.set_title("微电网多源功率分布")
        ax1.grid(True, linestyle="--", alpha=0.5)

        lines_labels = [ax1.get_legend_handles_labels() for ax1 in [ax1, ax1b]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        ax1.legend(lines, labels, ncol=3, fontsize=8, loc="upper left")

        # 子图2: 系统能量平衡
        total_load = load_p
        mismatch = (total_gen + grid_p) - total_load
        ax2.plot(hours, total_gen, color="green", linewidth=2, label="可再生 + 储能总出力")
        ax2.plot(hours, grid_p, color="red", linewidth=2, label="并网功率")
        ax2.plot(hours, total_load, color="purple", linewidth=2, label="负荷需求")
        ax2.plot(hours, mismatch, color="black", linestyle="--", linewidth=1.5, label="系统能量平衡（mismatch≈0理想）")

        ax2.axhline(0, color="k", linewidth=0.8)
        ax2.set_xlabel("时间步（小时）")
        ax2.set_ylabel("功率 / kW")
        ax2.set_title("系统能量平衡图")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(fontsize=9, loc="upper left")

        # 子图3: SOC曲线
        for i in range(n_ess):
            ax3.plot(hours, socs[:, i], label=f"ESS{i+1} SOC", linewidth=2)
            
        soc_max = getattr(env, 'SOC_max', 1.0)
        soc_min = getattr(env, 'SOC_min', 0.0)

        ax3.axhline(soc_max, color="r", linestyle="--", alpha=0.7, label='SOC_max')
        ax3.axhline(soc_min, color="r", linestyle="--", alpha=0.7, label='SOC_min')
        ax3.set_title("各储能设备 SOC 变化曲线")
        ax3.set_xlabel("时间步")
        ax3.set_ylabel("SOC")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend()

        # 子图4: 关键节点电压曲线
        key_nodes = [0, 10, 20, 32]
        node_labels = ['节点0 (首端)', '节点10', '节点20', '节点32 (末端)']
        
        for i, node_idx in enumerate(key_nodes):
            if node_idx < voltages.shape[1]:
                ax4.plot(hours, voltages[:, node_idx], label=node_labels[i], linewidth=2)
        
        ax4.axhline(env.vmax, color="r", linestyle="--", alpha=0.7, label='电压上限')
        ax4.axhline(env.vmin, color="r", linestyle="--", alpha=0.7, label='电压下限')
        ax4.set_title("关键节点电压变化曲线")
        ax4.set_xlabel("时间步")
        ax4.set_ylabel("电压 (p.u.)")
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend()

        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, "plot_summary.png")
            plt.savefig(save_path, dpi=300)
            print(f"图片已保存: {save_path}")
            
        plt.show()
        plt.close()

        # 调用其他函数
        plot_all_node_voltages(voltages, hours, env, save_dir)
        plot_voltage_heatmap(voltages, hours, env, save_dir)
        save_voltage_data(voltages, hours, env, save_dir)

    except Exception as e:
        print(f"主绘图函数发生错误: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# 最终执行
# ==============================================================================
if len(infos_all) > 0:
    print("\n开始生成可视化图表和报告...")
    infos = infos_all[-1]
    
    # 1. 保存分析报告
    save_voltage_analysis_report(infos, env, save_dir)

    # 2. 调用主绘图函数 (它会负责所有绘图和数据保存)
    plot_microgrid_power_from_info(infos, env, save_dir)
    
    print(f"\n所有结果已归档至: {save_dir}")