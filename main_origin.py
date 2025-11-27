# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 16:39:14 2025

@author: 12392
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 22:13:39 2025

@author: 12392
"""

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

# --- 环境设置 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# 显示中文
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
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

# ========== 数据集读取 ==========
csv_path = r"dataset\dataset_ieee33_extreme.csv"
data = pd.read_csv(csv_path)

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
        agent.alpha = torch.tensor(0.2, dtype=torch.float32)
except Exception as _e:
    print("[warn] alpha fix tweak skipped:", _e)

log_dir = r"outputs"
if os.path.exists(log_dir):
    print('找到对应文件夹runs')
    if not os.path.isdir(log_dir):
        print('runs不是个文件夹')
        os.remove(log_dir)  # 删除同名文件
        os.makedirs(log_dir)
else:
    os.makedirs(log_dir)

writer = SummaryWriter(
    os.path.join(
        log_dir,
        "{}_SAC_{}_{}_{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            args.env_name,
            args.policy,
            "autotune" if args.automatic_entropy_tuning else "",
        )
    )
)


memory = ReplayMemory(args.replay_size, args.seed)

# main.py
# 假设你已经有了一个预训练的模型或结构调制网络生成 gamma
# 在初始化 agent 前，添加如下代码生成 gamma 向量

# 假设你有一个 VSRDPGHead 模型来生成 gamma 向量
# 你可以根据状态来生成对应的 gamma

# 生成一个随机的 gamma 示例（根据实际需求可以更改）
def generate_gamma(batch_size, n_heads):
    return torch.rand(batch_size, n_heads).to(torch.device('cuda' if args.cuda else 'cpu'))

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

def plot_microgrid_power_from_info(infos, env):
    """
    绘制微电网多源功率分布 + 系统能量平衡图 + 各储能SOC变化曲线 + 节点电压曲线
    （修改：同一储能设备充放电颜色保持一致）
    """
    try:
        # ======= 提取信息 =======
        n_ess = env.n_batt
        n_pv = env.n_pv
        n_wind = env.n_wind

        batt_p = np.array([info.get("batt_p", np.zeros(n_ess)) for info in infos])  # [T, n_ess]
        p_ch = np.array([info.get("p_ch", np.zeros(n_ess)) for info in infos])
        p_dis = np.array([info.get("p_dis", np.zeros(n_ess)) for info in infos])
        pv_p = np.array([info.get("pv_p", np.zeros(n_pv)) for info in infos])
        wind_p = np.array([info.get("wind_p", np.zeros(n_wind)) for info in infos])
        grid_p = np.array([info.get("grid_kW", 0.0) for info in infos])
        socs = np.array([info.get("soc", np.zeros(n_ess)) for info in infos])
        prices = np.array([info.get("price", 0.0) for info in infos])
        load_p = np.array([info.get("load_kW", 0.0) for info in infos])
        voltages = np.array([info.get("voltages", np.zeros(len(env.net.bus))) for info in infos])  # 节点电压

        T = len(infos)
        hours = np.arange(T)

        # 汇总
        total_ess_p = batt_p.sum(axis=1)
        total_pv = pv_p.sum(axis=1)
        total_wind = wind_p.sum(axis=1)
        total_gen = total_ess_p + total_pv + total_wind

        # ==================== 图1：多源功率堆叠 ====================
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
        
        # 储能功率堆叠：正为放电，负为充电（充放电同色）
        bottom_pos = np.zeros(T)
        bottom_neg = np.zeros(T)
        ess_colors = ["#203864", "#305496", "#4472C4", "#5B9BD5", "#8EA9DB",
                      "#A9D18E", "#548235", "#BF9000", "#7F6000", "#7030A0"]

        for i in range(n_ess):
            dis = np.where(batt_p[:, i] > 0, batt_p[:, i], 0)
            chg = np.where(batt_p[:, i] < 0, batt_p[:, i], 0)
            color = ess_colors[i % len(ess_colors)]  # 同一ESS同色
            ax1.bar(hours, dis, bottom=bottom_pos, color=color, width=0.6, label=f"ESS{i+1}")
            ax1.bar(hours, chg, bottom=bottom_neg, color=color, width=0.6)
            bottom_pos += dis
            bottom_neg += chg

        # 加上风电与光伏
        ax1.bar(hours, total_wind, bottom=bottom_pos, color="#00B0F0", width=0.6, label="风电")
        ax1.bar(hours, total_pv, bottom=bottom_pos + total_wind, color="#FFD966", width=0.6, label="光伏")

        # 并网功率
        ax1.plot(hours, grid_p, color="red", linewidth=2.2, marker="x", label="并网功率 Grid Power")

        # 电价曲线（右轴）
        ax1b = ax1.twinx()
        ax1b.plot(hours, prices, color="black", linestyle="--", linewidth=2, label="电价 (RMB/kWh)")
        ax1b.set_ylabel("电价 (RMB/kWh)")

        ax1.axhline(0, color="k", linewidth=0.8)
        ax1.set_ylabel("功率 / kW")
        ax1.set_title("微电网多源功率分布")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 合并主轴与副轴图例
        lines_labels = [ax1.get_legend_handles_labels() for ax1 in [ax1, ax1b]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        ax1.legend(lines, labels, ncol=3, fontsize=8, loc="upper left")

        # ==================== 图2：系统能量平衡 ====================
        total_load = load_p
        total_gen_plus_grid = total_gen + grid_p
        mismatch = total_gen_plus_grid - total_load

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

        # ==================== 图3：SOC曲线 ====================
        for i in range(n_ess):
            ax3.plot(hours, socs[:, i], label=f"ESS{i+1} SOC", linewidth=2)

        # 检查 SOC 限制
        if hasattr(env, 'SOC_max') and hasattr(env, 'SOC_min'):
            soc_max = env.SOC_max
            soc_min = env.SOC_min
        else:
            soc_max = 1.0
            soc_min = 0.0
            print("使用默认 SOC 限制: [0.0, 1.0]")

        ax3.axhline(soc_max, color="r", linestyle="--", alpha=0.7, label='SOC_max')
        ax3.axhline(soc_min, color="r", linestyle="--", alpha=0.7, label='SOC_min')

        ax3.set_title("各储能设备 SOC 变化曲线")
        ax3.set_xlabel("时间步")
        ax3.set_ylabel("SOC")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend()

        # ==================== 图4：关键节点电压曲线 ====================
        # 绘制关键节点的电压（例如：首节点、中间节点、末端节点）
        key_nodes = [0, 10, 20, 32]  # 关键节点索引
        node_labels = ['节点0 (首端)', '节点10', '节点20', '节点32 (末端)']
        
        for i, node_idx in enumerate(key_nodes):
            if node_idx < voltages.shape[1]:
                ax4.plot(hours, voltages[:, node_idx], label=node_labels[i], linewidth=2)
        
        # 添加电压上下限
        ax4.axhline(env.vmax, color="r", linestyle="--", alpha=0.7, label='电压上限')
        ax4.axhline(env.vmin, color="r", linestyle="--", alpha=0.7, label='电压下限')
        
        ax4.set_title("关键节点电压变化曲线")
        ax4.set_xlabel("时间步")
        ax4.set_ylabel("电压 (p.u.)")
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend()

        plt.tight_layout()
        plt.show()

        # ==================== 新增：所有节点电压曲线图 ====================
        plot_all_node_voltages(voltages, hours, env)
        
        # ==================== 新增：电压热力图 ====================
        plot_voltage_heatmap(voltages, hours, env)

        # ==================== 保存节点电压数据 ====================
        save_voltage_data(voltages, hours, env)

    except Exception as e:
        print(f"绘图时发生错误: {e}")
        import traceback
        traceback.print_exc()

def plot_all_node_voltages(voltages, hours, env):
    """
    绘制所有33个节点的电压变化曲线
    """
    try:
        # 创建一个大图来显示所有节点
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 使用颜色映射，根据节点位置分配颜色
        n_nodes = voltages.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, n_nodes))
        
        # 绘制所有节点的电压曲线
        for node in range(n_nodes):
            # 使用透明度区分不同节点，避免过于混乱
            alpha = 0.7 if node in [0, 10, 20, 32] else 0.4  # 关键节点更明显
            linewidth = 2.0 if node in [0, 10, 20, 32] else 1.0
            
            ax.plot(hours, voltages[:, node], 
                   color=colors[node], 
                   alpha=alpha, 
                   linewidth=linewidth,
                   label=f'Node {node}' if node in [0, 10, 20, 32] else "")
        
        # 添加电压上下限
        ax.axhline(env.vmax, color="red", linestyle="--", linewidth=2, alpha=0.8, label='电压上限')
        ax.axhline(env.vmin, color="red", linestyle="--", linewidth=2, alpha=0.8, label='电压下限')
        
        # 设置图表属性
        ax.set_xlabel("时间 (小时)")
        ax.set_ylabel("电压 (p.u.)")
        ax.set_title("所有节点电压变化曲线")
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # 只显示关键节点的图例，避免图例过多
        ax.legend(loc='upper right', fontsize=10)
        
        # 添加颜色条表示节点编号
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                  norm=plt.Normalize(vmin=0, vmax=n_nodes-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('节点编号')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ 已绘制所有 {n_nodes} 个节点的电压变化曲线")
        
    except Exception as e:
        print(f"绘制所有节点电压曲线时发生错误: {e}")
        import traceback
        traceback.print_exc()

def plot_voltage_heatmap(voltages, hours, env):
    """
    绘制节点电压热力图，直观显示电压分布
    """
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 创建热力图数据
        heatmap_data = voltages.T  # 转置，使节点在y轴，时间在x轴
        
        # 绘制热力图
        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r', 
                      extent=[hours[0], hours[-1], 0, voltages.shape[1]-1],
                      vmin=env.vmin, vmax=env.vmax)
        
        # 设置坐标轴
        ax.set_xlabel("时间 (小时)")
        ax.set_ylabel("节点编号")
        ax.set_title("节点电压热力图")
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('电压 (p.u.)')
        
        # 设置y轴刻度，显示所有节点
        ax.set_yticks(range(voltages.shape[1]))
        ax.set_yticklabels([f'Node {i}' for i in range(voltages.shape[1])])
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ 已生成节点电压热力图")
        
    except Exception as e:
        print(f"绘制电压热力图时发生错误: {e}")
        import traceback
        traceback.print_exc()

def save_voltage_data(voltages, hours, env):
    """
    保存24个时间点的各节点电压数据到CSV文件
    """
    try:
        # 创建DataFrame，行为时间点，列为节点
        voltage_df = pd.DataFrame(voltages, 
                                 index=[f"Hour_{h}" for h in hours],
                                 columns=[f"Node_{i}" for i in range(len(env.net.bus))])
        
        # 添加时间戳列
        voltage_df['Hour'] = hours
        voltage_df['Timestamp'] = pd.date_range(start='2025-01-01', periods=len(hours), freq='H')
        
        # 重新排列列顺序，将时间和节点分开
        cols = ['Hour', 'Timestamp'] + [f"Node_{i}" for i in range(len(env.net.bus))]
        voltage_df = voltage_df[cols]
        
        # 保存到CSV文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"node_voltages_{timestamp}.csv"
        voltage_df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"✅ 节点电压数据已保存到: {filename}")
        print(f"   数据包含 {len(hours)} 个时间点和 {len(env.net.bus)} 个节点")
        
        # 打印统计信息
        print("\n📊 电压统计信息:")
        print(f"   平均电压范围: {np.min(voltages):.4f} - {np.max(voltages):.4f} p.u.")
        print(f"   电压越限次数: {np.sum((voltages < env.vmin) | (voltages > env.vmax))}")
        
        # 找出电压最低和最高的节点
        min_voltage_node = np.argmin(voltages, axis=1)
        max_voltage_node = np.argmax(voltages, axis=1)
        
        print(f"   最低电压出现: 节点{min_voltage_node[np.argmin(voltages)]} (值: {np.min(voltages):.4f} p.u.)")
        print(f"   最高电压出现: 节点{max_voltage_node[np.argmax(voltages)]} (值: {np.max(voltages):.4f} p.u.)")
        
        return voltage_df
        
    except Exception as e:
        print(f"保存电压数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# 调用绘图函数
voltage_data = plot_microgrid_power_from_info(infos, env)

# 额外保存一份详细的分析报告
def save_voltage_analysis_report(infos, env):
    """
    生成并保存详细的电压分析报告
    """
    try:
        voltages = np.array([info.get("voltages", np.zeros(len(env.net.bus))) for info in infos])
        hours = np.arange(len(infos))
        
        # 创建详细的分析报告
        report_data = []
        
        for hour in hours:
            hour_voltages = voltages[hour]
            min_voltage = np.min(hour_voltages)
            max_voltage = np.max(hour_voltages)
            min_node = np.argmin(hour_voltages)
            max_node = np.argmax(hour_voltages)
            violation_count = np.sum((hour_voltages < env.vmin) | (hour_voltages > env.vmax))
            
            report_data.append({
                'Hour': hour,
                'Min_Voltage': min_voltage,
                'Max_Voltage': max_voltage,
                'Min_Node': min_node,
                'Max_Node': max_node,
                'Violation_Count': violation_count,
                'Avg_Voltage': np.mean(hour_voltages)
            })
        
        report_df = pd.DataFrame(report_data)
        
        # 保存分析报告
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"voltage_analysis_report_{timestamp}.csv"
        report_df.to_csv(report_filename, index=False, encoding='utf-8-sig')
        
        print(f"✅ 电压分析报告已保存到: {report_filename}")
        
        return report_df
        
    except Exception as e:
        print(f"生成电压分析报告时发生错误: {e}")
        return None

# 生成电压分析报告
voltage_report = save_voltage_analysis_report(infos, env)