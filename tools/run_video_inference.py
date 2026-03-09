import sys
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

# 添加根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from model.VIC import Video_Counter
from model.kalman_filter import AdaptiveKalmanFilter
from easydict import EasyDict as edict

# ==================== 【配置区域】 ====================
# 1. 训练好的模型
MODEL_PATH = "/root/autodl-tmp/FishCount_Project/exp/DeepFish/12-30_09-57_DeepFish_1e-05_train/output/best_model.pth"
# 2. 测试图片文件夹
TEST_IMG_DIR = "/root/autodl-tmp/DeepFish/test/images"
# 3. 结果保存位置
SAVE_DIR = "/root/autodl-tmp/DeepFish/test_results"
# ======================================================

# === 辅助函数：裁边去噪 ===
def post_process_density(den_map, crop_size=16):
    """强制去除密度图边缘的数值，消除边缘效应"""
    den = den_map.clone()
    h, w = den.shape[-2], den.shape[-1]
    den[..., :crop_size, :] = 0  # 上
    den[..., h-crop_size:, :] = 0 # 下
    den[..., :, :crop_size] = 0  # 左
    den[..., :, w-crop_size:] = 0 # 右
    return den

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型
    print(">>> 正在加载 VIC 模型...")
    cfg_data = edict(); cfg_data.DEN_FACTOR = 100
    model = Video_Counter(cfg, cfg_data).to(device)
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH)
        new_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_dict)
        model.eval()
    else:
        print(f"❌ 错误：找不到权重文件 {MODEL_PATH}")
        return

    # 2. 初始化 KF
    kf = AdaptiveKalmanFilter(initial_count=0)

    # 3. 准备数据
    import glob
    img_paths = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    # 截取前 100 帧做演示 (如果想跑全部，注释掉下面这行)
    img_paths = img_paths[:100] 
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_counts = []
    kf_counts = []
    frame_indices = []

    print(f">>> 开始处理视频流 (共 {len(img_paths)} 帧)...")
    prev_img_tensor = None

    with torch.no_grad():
        for i, path in enumerate(img_paths):
            # 读取
            img_raw = cv2.imread(path)
            if img_raw is None: continue
            
            # 【修正1】分辨率必须是 512 (配合你的 SOTA 模型)
            img = cv2.resize(img_raw, (512, 512)) 
            curr_tensor = transform(img).unsqueeze(0).to(device)

            # 构造 Input Pair
            if prev_img_tensor is None:
                input_tensor = torch.cat([curr_tensor, curr_tensor], dim=0)
            else:
                input_tensor = torch.cat([prev_img_tensor, curr_tensor], dim=0)
            prev_img_tensor = curr_tensor

            # 构造 Dummy Target (为了跑通 forward)
            dummy_points = torch.tensor([[0, 0]]).float().cuda()
            dummy_point_mask = torch.tensor([True], dtype=torch.bool).cuda() # 1维 Bool
            dummy_img_mask = torch.ones((1, 512, 512)).cuda() # 注意尺寸也是 512

            dummy_dict = {
                'points': dummy_points,
                'share_mask0': dummy_point_mask, 'share_mask1': dummy_point_mask,
                'share_mask': dummy_point_mask, 'outflow_mask': dummy_point_mask, 'inflow_mask': dummy_point_mask,
                'roi_mask': dummy_img_mask, 'mask': dummy_img_mask,
                'gt_global_map': torch.zeros((1, 512, 512)).cuda(), # 占位
                'gt_shared_map': torch.zeros((1, 512, 512)).cuda(),
                'gt_out_map': torch.zeros((1, 512, 512)).cuda(), 'gt_in_map': torch.zeros((1, 512, 512)).cuda(),
                'image_path': "test_inference"
            }
            target = [dummy_dict, dummy_dict]

            # 推理
            res = model(input_tensor, target)
            pre_global_den = torch.relu(res[0])
            
            # 取当前帧 (Index 1)
            current_density = pre_global_den[1, 0]

            # 【修正2】裁边后处理 (去除左上角鬼影)
            clean_density = post_process_density(current_density.unsqueeze(0), crop_size=16)
            
            # 计算原始计数
            raw_count = clean_density.sum().item()
            
            # 简单模拟不确定性 (如果你有 BNN，这里可以用方差代替)
            uncertainty_score = 0.5 

            # 卡尔曼修正
            kf.predict()
            # 【修正3】负数保护
            final_count = max(0.0, kf.update(raw_count, uncertainty_score))

            # 记录
            raw_counts.append(raw_count)
            kf_counts.append(final_count)
            frame_indices.append(i)
            
            if i % 10 == 0:
                print(f"Frame {i}: Raw={raw_count:.2f} -> KF={final_count:.2f}")

    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, raw_counts, 'g--', alpha=0.5, label='Transformer Raw Output')
    plt.plot(frame_indices, kf_counts, 'b-', linewidth=2, label='Transformer + Kalman Filter')
    plt.title("Fish Counting: Raw vs Filtered")
    plt.xlabel("Frame Index")
    plt.ylabel("Fish Count")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(SAVE_DIR, "kf_comparison_final.png")
    plt.savefig(save_path)
    print(f"\n>>> 结果图已保存: {save_path}")

if __name__ == '__main__':
    main()