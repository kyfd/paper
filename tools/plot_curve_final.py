import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==================== 【配置区域】 ====================
# 1. 你的 events 文件路径 (请修改为你最新的那个)
# 例如: /root/autodl-tmp/FishCount_Project/exp/DeepFish/12-20_xxxx/events.out.tfevents.xxxxx
EVENTS_PATH = "/root/autodl-tmp/FishCount_Project/exp/DeepFish/12-27_09-58_DeepFish_1e-05_train/events.out.tfevents.1766800717.autodl-container-acf046b3fb-903e49da"

# 2. 图片保存名称
SAVE_NAME = "loss_curve_paper.png"
# ====================================================

def smooth(scalars, weight=0.9):
    """
    平滑函数：让锯齿状的 Loss 曲线变得丝滑
    weight: 平滑系数 (0~1)，越大越平滑
    """
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_paper_curve():
    print(f"正在读取日志: {EVENTS_PATH}")
    if not os.path.exists(EVENTS_PATH):
        print("❌ 错误：文件不存在，请检查路径！")
        return

    # 加载数据 (scalars=0 表示加载所有标量数据)
    ea = EventAccumulator(EVENTS_PATH, size_guidance={'scalars': 0})
    ea.Reload()
    
    # 获取所有标签
    tags = ea.Tags()['scalars']
    print(f"包含的数据标签: {tags}")

    # === 开始画图 ===
    plt.figure(figsize=(10, 6)) # 宽10，高6 (适合论文)
    
    # 1. 绘制 Global Loss (总误差)
    if 'global' in tags:
        data = ea.Scalars('global')
        steps = [x.step for x in data]
        values = [x.value for x in data]
        
        # 画原始数据的浅色背景线 (Shadow)
        plt.plot(steps, values, color='salmon', alpha=0.3, linewidth=1)
        # 画平滑后的深色实线 (Main Line)
        plt.plot(steps, smooth(values, 0.95), color='red', label='Global Loss', linewidth=2)

    # 2. 绘制 Share Loss (共享特征误差 - 你的创新点)
    if 'share' in tags:
        data = ea.Scalars('share')
        steps = [x.step for x in data]
        values = [x.value for x in data]
        
        # Share Loss 通常比较大，除以10方便在同一张图展示趋势 (可选)
        # values = [v/10.0 for v in values] 
        
        plt.plot(steps, values, color='lightblue', alpha=0.3, linewidth=1)
        plt.plot(steps, smooth(values, 0.95), color='blue', label='Share Loss (DCFA)', linewidth=2, linestyle='--')

    # 3. 装饰图片 (让它像论文插图)
    plt.title("Training Convergence Analysis", fontsize=14, fontweight='bold')
    plt.xlabel("Training Iterations", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 自动调整边距
    plt.tight_layout()
    
    # 保存高清图 (300 DPI)
    plt.savefig(SAVE_NAME, dpi=300)
    print(f"✅ 高清曲线图已保存为: {SAVE_NAME}")
    plt.show()

if __name__ == '__main__':
    plot_paper_curve()