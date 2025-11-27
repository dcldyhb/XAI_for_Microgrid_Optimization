# -*- coding: utf-8 -*-
# main.py
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

from SAC_gama.replay_memory import ReplayMemory
from SAC_gama.sac import SAC
from SAC_gama.microgrid_env_complex_v11 import IEEE33Env  # 使用 v11 环境（含 info 输出）

# --- 导入配置 ---
from config import DEVICE, DATASET_PATH, OUTPUTS_ROOT, FONT_SANS_SERIF

# --- 环境设置 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# 显示中文 (使用 config 配置)
matplotlib.rcParams["font.sans-serif"] = FONT_SANS_SERIF
matplotlib.rcParams["axes.unicode_minus"] = False

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
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--eval_every", type=int, default=10)
parser.add_argument("--eval_episodes", type=int, default=1)
args = parser.parse_args()

# --- 依赖注入 ---
args.device = DEVICE

# ========== 数据集读取 ==========
print(f"读取数据集: {DATASET_PATH}")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"未找到数据集文件: {DATASET_PATH}")
data = pd.read_csv(DATASET_PATH)

# ========== 随机种子 ==========
if args.seed is None or args.seed < 0:
    args.seed = (int(time.time() * 1e6) ^ os.getpid() ^ random.getrandbits(32)) & 0xFFFFFFFF
print(f"[seed] using seed = {args.seed}")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# ========== 初始化环境 ==========
env = IEEE33Env(data)
env.seed(None)
env.G_scale = 1.0 / 1000.0
#print(data.loc[0:24, ["timestamp", "solar_irradiance"]])


# ========== 初始化 agent ==========
agent = SAC(env.observation_space.shape[0], env.action_space, args)
try:
    if hasattr(agent, "automatic_entropy_tuning"):
        agent.automatic_entropy_tuning = False
    if hasattr(agent, "alpha"):
        agent.alpha = torch.tensor(0.2, dtype=torch.float32).to(DEVICE)
except Exception as _e:
    print("[warn] alpha fix tweak skipped:", _e)

# 1. 使用 config 中的 OUTPUTS_ROOT
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"{timestamp}_{args.env_name}_{args.policy}"
save_dir = os.path.join(OUTPUTS_ROOT, run_name) # <--- 这里用了 config

# 2. 定义本次实验的独立文件夹: outputs/时间戳_环境名_策略/
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"{timestamp}_{args.env_name}_{args.policy}"
save_dir = os.path.join(OUTPUTS_ROOT, run_name)

# 3. 创建目录
os.makedirs(save_dir, exist_ok=True)
print(f"[Output] 本次运行的所有结果将保存至: {save_dir}")

# 4. 设置 TensorBoard 日志目录到子文件夹 logs/
log_dir = os.path.join(save_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir)


memory = ReplayMemory(args.replay_size, args.seed)

# main.py
# 假设你已经有了一个预训练的模型或结构调制网络生成 gamma
# 在初始化 agent 前，添加如下代码生成 gamma 向量

# 假设你有一个 VSRDPGHead 模型来生成 gamma 向量
# 你可以根据状态来生成对应的 gamma

# 生成一个随机的 gamma 示例（根据实际需求可以更改）
def generate_gamma(batch_size, n_heads):
    # 使用全局 DEVICE (自动适配 M4/CUDA)
    return torch.rand(batch_size, n_heads).to(DEVICE)

# 在训练过程中的每一步，生成 gamma 向量并传递给 policy

# ========== 训练与收集 info ==========
rewards = []
infos_all = []  # 存放每个 episode 的 infos（每步 info）

total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    env.seed(args.seed + i_episode)
    state = env.reset()
    done = False
    episode_reward = 0.0
    episode_steps = 0
    infos = []

    while not done:
        # 生成 gamma 向量
        gamma = generate_gamma(1, n_heads=10)  # 假设 gamma 有 10 个头

        if args.start_steps > total_numsteps:
            action = env.action_space.sample()
        else:
            # 传递 gamma 向量到 policy
            action, log_prob, mean = agent.select_action(state,  gamma=gamma, evaluate=False)


        next_state, reward, done, info = env.step(action)
        infos.append(info)  # 保存 info（v5 返回 batt_p, pv_p, wind_p, grid_kW, price, mismatch）
        
        #if episode_steps % 6 == 0:  # 每隔6步打印一次
            #print(f"[Episode {i_episode} | Step {episode_steps}] PV_P = {info['pv_p']}")

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
    print(f"✅ Episode: {i_episode}, total numsteps: {total_numsteps}, steps: {episode_steps}, reward: {round(episode_reward, 2)}")

    if total_numsteps >= args.num_steps:
        break

env.close()
writer.close()

# ========== 可视化（选取最后一个 episode 的 infos） ==========
infos = infos_all[-1]  # 使用最后一个 episode 的数据

# ==============================================================================
# 绘图与可视化函数群 (优化版：支持输出到 save_dir)
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
            is_key_node = node in [0, 10, 20, 32]
            alpha = 0.8 if is_key_node else 0.3
            linewidth = 2.5 if is_key_node else 1.0
            label = f'Node {node}' if is_key_node else ""
            
            ax.plot(hours, voltages[:, node], color=colors[node], 
                   alpha=alpha, linewidth=linewidth, label=label)
        
        ax.axhline(env.vmax, color="red", linestyle="--", linewidth=2, label='Limit')
        ax.axhline(env.vmin, color="red", linestyle="--", linewidth=2)
        
        ax.set_xlabel("Time (Hour)")
        ax.set_ylabel("Voltage (p.u.)")
        ax.set_title("All 33 Nodes Voltage Profile")
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加Colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, n_nodes-1))
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Node Index')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, "plot_all_nodes_voltage.png"), dpi=300)
            print(f"[Plot] 全节点电压图已保存")
            
        plt.show()
        plt.close()
    except Exception as e:
        print(f"绘制全节点电压图失败: {e}")

def plot_voltage_heatmap(voltages, hours, env, save_dir=None):
    """
    绘制节点电压热力图
    """
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        heatmap_data = voltages.T
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r', 
                      extent=[hours[0], hours[-1], 0, voltages.shape[1]-1],
                      vmin=env.vmin, vmax=env.vmax)
        
        ax.set_xlabel("Time (Hour)")
        ax.set_ylabel("Node Index")
        ax.set_title("Node Voltage Heatmap")
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Voltage (p.u.)')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, "plot_voltage_heatmap.png"), dpi=300)
            print(f"[Plot] 电压热力图已保存")
            
        plt.show()
        plt.close()
    except Exception as e:
        print(f"绘制热力图失败: {e}")

def plot_microgrid_power_from_info(infos, env, save_dir=None):
    """
    主绘图函数：微电网多源功率 + 能量平衡 + SOC + 关键节点电压
    """
    try:
        # ======= 1. 数据提取 =======
        n_ess = env.n_batt
        n_pv = env.n_pv
        n_wind = env.n_wind

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

        # ======= 2. 绘制四子图 =======
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
        
        # --- 子图1: 功率堆叠 ---
        bottom_pos = np.zeros(T)
        bottom_neg = np.zeros(T)
        ess_colors = ["#203864", "#305496", "#4472C4", "#5B9BD5", "#8EA9DB"]

        for i in range(n_ess):
            dis = np.where(batt_p[:, i] > 0, batt_p[:, i], 0)
            chg = np.where(batt_p[:, i] < 0, batt_p[:, i], 0)
            color = ess_colors[i % len(ess_colors)]
            ax1.bar(hours, dis, bottom=bottom_pos, color=color, width=0.6, label=f"ESS{i}")
            ax1.bar(hours, chg, bottom=bottom_neg, color=color, width=0.6)
            bottom_pos += dis
            bottom_neg += chg

        ax1.bar(hours, total_wind, bottom=bottom_pos, color="#00B0F0", width=0.6, label="Wind")
        ax1.bar(hours, total_pv, bottom=bottom_pos + total_wind, color="#FFD966", width=0.6, label="PV")
        ax1.plot(hours, grid_p, color="red", linewidth=2, marker="x", label="Grid")

        ax1b = ax1.twinx()
        ax1b.plot(hours, prices, "k--", alpha=0.5, label="Price")
        ax1.set_title("Source Power Dispatch")
        ax1.legend(loc="upper left", ncol=3, fontsize=8)

        # --- 子图2: 能量平衡 ---
        mismatch = (total_gen + grid_p) - load_p
        ax2.plot(hours, total_gen, "g", label="Renewable+ESS")
        ax2.plot(hours, grid_p, "r", label="Grid")
        ax2.plot(hours, load_p, "purple", label="Load")
        ax2.plot(hours, mismatch, "k--", label="Mismatch")
        ax2.set_title("Power Balance")
        ax2.legend()

        # --- 子图3: SOC ---
        for i in range(n_ess):
            ax3.plot(hours, socs[:, i], label=f"ESS{i}")
        soc_max = getattr(env, 'SOC_max', 1.0)
        soc_min = getattr(env, 'SOC_min', 0.0)
        ax3.axhline(soc_max, color='r', linestyle='--')
        ax3.axhline(soc_min, color='r', linestyle='--')
        ax3.set_title("Battery SOC")
        ax3.legend()

        # --- 子图4: 关键节点电压 ---
        key_nodes = [0, 10, 20, 32]
        for node in key_nodes:
            if node < voltages.shape[1]:
                ax4.plot(hours, voltages[:, node], label=f"Node {node}")
        ax4.axhline(env.vmax, color='r', linestyle='--')
        ax4.axhline(env.vmin, color='r', linestyle='--')
        ax4.set_title("Key Nodes Voltage")
        ax4.legend()

        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, "plot_summary.png"), dpi=300)
            print(f"[Plot] 综合分析图已保存")

        plt.show()
        plt.close()

        # 调用其他绘图函数
        plot_all_node_voltages(voltages, hours, env, save_dir)
        plot_voltage_heatmap(voltages, hours, env, save_dir)

    except Exception as e:
        print(f"绘图主程序出错: {e}")
        import traceback
        traceback.print_exc()

def save_voltage_data(voltages, hours, env, save_dir):
    """
    保存节点电压数据 (优化版：修复索引BUG，输出到指定目录)
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
            file_path = os.path.join(save_dir, "node_voltages.csv")
            voltage_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f" [CSV] 节点电压数据已保存: {file_path}")
        
        # 统计信息与Debug修复
        min_val = np.min(voltages)
        max_val = np.max(voltages)
        
        # 使用 unravel_index 修复之前的 IndexError
        min_hour_idx, min_node_idx = np.unravel_index(np.argmin(voltages), voltages.shape)
        max_hour_idx, max_node_idx = np.unravel_index(np.argmax(voltages), voltages.shape)
        
        print("\n 电压统计信息:")
        print(f"  平均电压范围: {min_val:.4f} - {max_val:.4f} p.u.")
        print(f"  最低电压: {min_val:.4f} (Hour {hours[min_hour_idx]}, Node {min_node_idx})")
        print(f"  最高电压: {max_val:.4f} (Hour {hours[max_hour_idx]}, Node {max_node_idx})")
        
        return voltage_df
    except Exception as e:
        print(f" 保存电压数据失败: {e}")
        return None

def save_voltage_analysis_report(infos, env, save_dir):
    """
    保存分析报告 (优化版：输出到指定目录)
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
            file_path = os.path.join(save_dir, "voltage_analysis_report.csv")
            report_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"[CSV] 电压分析报告已保存: {file_path}")
            
        return report_df
    except Exception as e:
        print(f"保存分析报告失败: {e}")
        return None

# ==============================================================================
# 最终执行 (使用新的 save_dir)
# ==============================================================================
if len(infos_all) > 0:
    print("\n 开始生成可视化图表...")
    infos = infos_all[-1]
    
    # 1. 保存 CSV
    save_voltage_data(
        voltages=np.array([i.get("voltages") for i in infos]), 
        hours=np.arange(len(infos)), 
        env=env, 
        save_dir=save_dir 
    )
    
    save_voltage_analysis_report(infos, env, save_dir)

    # 2. 绘图并自动保存
    plot_microgrid_power_from_info(infos, env, save_dir)
    
    print(f"\n 所有结果已归档至: {save_dir}")