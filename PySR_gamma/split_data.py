import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import StratifiedShuffleSplit

# ==========================================
# 1. 路径配置 (全自动绝对路径适配)
# ==========================================

# 获取当前脚本所在目录 (.../PySR_gamma/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (.../XAI_for_Microgrid_Optimization/)
project_root = os.path.dirname(current_dir)

# 将根目录加入系统路径，以便能导入 config
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config import PROJECT_ROOT, AUGMENTED_DATASET_PATH
    # 方案 A: Config 存在，使用 Config 定义的根路径
    # 注意：PROJECT_ROOT 来自 config，应该是绝对路径
    TRAIN_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_train.csv')
    TEST_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_test.csv')
    INPUT_FILE = AUGMENTED_DATASET_PATH
    print("已加载 config.py 配置")

except ImportError:
    # 方案 B: Config 导入失败，使用手动计算的 project_root 构建绝对路径
    print("未找到 config.py，正在使用基于脚本位置推算的绝对路径...")
    
    # 核心修改：使用 os.path.join(project_root, ...) 确保指向根目录下的 dataset
    dataset_dir = os.path.join(project_root, 'dataset')
    
    INPUT_FILE = os.path.join(dataset_dir, 'dataset_ieee33_extreme_full_augmented_phys.csv')
    TRAIN_PATH = os.path.join(dataset_dir, 'dataset_ieee33_extreme_train.csv')
    TEST_PATH = os.path.join(dataset_dir, 'dataset_ieee33_extreme_test.csv')

# 调试打印，让你确信路径是对的
print("-" * 30)
print(f"项目根目录: {project_root}")
print(f"输入文件:   {INPUT_FILE}")
print(f"训练集输出: {TRAIN_PATH}")
print("-" * 30)

# ==========================================
# 2. 划分配置
# ==========================================
TEST_SIZE = 0.2  # 20% 作为测试集
RANDOM_SEED = 42 # 保证每次划分结果一致

def main():
    # --- 1. 读取数据 ---
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件")
        print(f" 路径: {INPUT_FILE}")
        print(" -> 请先在项目根目录运行: python augment_ieee33.py")
        return

    print("开始读取数据...")
    df = pd.read_csv(INPUT_FILE)
    print(f"原始数据量: {len(df)}")

    # --- 2. 数据清洗 ---
    # 扩充时如果潮流计算失败，Gamma 可能是空值，需要剔除
    df_clean = df.dropna(subset=['gamma']) 
    
    # 可选：剔除完全不合理的 Gamma (例如 < -0.5，说明系统严重崩溃)
    # df_clean = df_clean[df_clean['gamma'] > -0.5]
    
    dropped_count = len(df) - len(df_clean)
    if dropped_count > 0:
        print(f"清洗无效数据: 剔除了 {dropped_count} 条 (剩余 {len(df_clean)} 条)")

    # --- 3. 定义分层依据 ---
    # 我们希望根据 'regime' (天气类型) 进行分层
    if 'regime' not in df_clean.columns:
        print("警告：数据中没有 'regime' 列，将进行随机划分。")
        split_col = np.zeros(len(df_clean)) # 全0数组，相当于随机
    else:
        split_col = df_clean['regime']
        print("\n各天气类型样本分布:")
        print(df_clean['regime'].value_counts())

    # --- 4. 执行分层划分 ---
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    for train_idx, test_idx in splitter.split(df_clean, split_col):
        train_set = df_clean.iloc[train_idx]
        test_set = df_clean.iloc[test_idx]

    # --- 5. 验证划分结果 ---
    print("\n划分完成！")
    print(f"训练集: {len(train_set)} 条 ({(len(train_set)/len(df_clean))*100:.1f}%)")
    print(f"测试集: {len(test_set)} 条 ({(len(test_set)/len(df_clean))*100:.1f}%)")

    # 验证测试集是否包含所有天气类型
    if 'regime' in df_clean.columns:
        print("\n测试集覆盖率验证 (确保包含所有极端工况):")
        print(test_set['regime'].value_counts())

    # --- 6. 保存 ---
    train_set.to_csv(TRAIN_PATH, index=False)
    test_set.to_csv(TEST_PATH, index=False)

    print(f"\n文件已保存:")
    print(f"-> {TRAIN_PATH}")
    print(f"-> {TEST_PATH}")

if __name__ == "__main__":
    main()