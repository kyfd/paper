import sys
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from tqdm import tqdm

# === 1. 环境路径设置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import cfg
from model.VIC import Video_Counter
from datasets.fish_dataset import MiniFishDataset

# ==================== 【配置区域】 ====================
# 1. 测试集路径
TEST_ROOT = "/root/autodl-tmp/DeepFish/test"

# 2. 模型权重路径 (使用你训练出的最好的模型)
MODEL_PATH = "/root/autodl-tmp/FishCount_Project/exp/DeepFish/12-30_09-57_DeepFish_1e-05_train/output/best_model.pth"
# 如果不知道具体路径，去左侧文件栏 exp 里面找一下复制过来

# 3. 结果保存文件夹
SAVE_DIR = "/root/autodl-tmp/DeepFish/test_results_all"
# ==========================================================

def denormalize(tensor):
    """反归一化，让图片显示正常颜色"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std) + mean
    return np.clip(img, 0, 1)

def post_process_density(den_map, crop_size=16):
    """裁边后处理：消除边缘效应"""
    den = den_map.clone()
    h, w = den.shape[-2], den.shape[-1]
    den[..., :crop_size, :] = 0
    den[..., h-crop_size:, :] = 0
    den[..., :, :crop_size] = 0
    den[..., :, w-crop_size:] = 0
    return den

def run_save_all():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    device = torch.device('cuda')

    # 1. 加载模型
    print(">>> 正在加载模型...")
    cfg_data = edict()
    cfg_data.DEN_FACTOR = 100 
    cfg_data.VAL_FRAME_INTERVALS = 1
    
    model = Video_Counter(cfg, cfg_data).to(device)
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH)
        new_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_dict)
        model.eval()
    else:
        print(f"❌ 错误：找不到权重文件 {MODEL_PATH}")
        return

    # 2. 加载数据
    print(">>> 正在加载数据...")
    dataset = MiniFishDataset(TEST_ROOT, train=False)
    # Batch Size 必须为 1，方便逐张保存
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    print(f"共 {len(dataset)} 张图片，开始全量生成...")

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc="Saving Images")):
            # 解包 11 个变量
            img1, img2, den_g1, den_s1, den_g2, den_s2, pts1, cnt1, pts2, cnt2, name_list = data
            
            # 拼装输入 [1, 2, 3, 512, 512] -> [2, 3, 512, 512]
            imgs_stacked = torch.stack([img1, img2], dim=1)
            # 【注意】这里必须改成 512，和你训练时一致
            img_input = imgs_stacked.view(-1, 3, 512, 512).to(device)

            # === 核心修复：正确构造 Target 列表 ===
            target = []
            # 因为 batch_size=1，这里只会循环一次
            for b in range(img1.shape[0]):
                # --- Frame 1 ---
                pts_t1 = pts1[b].float().to(device)
                real_num_t1 = cnt1[b].item()
                p_mask_t1 = torch.zeros(pts_t1.shape[0], dtype=torch.bool).to(device)
                if real_num_t1 > 0: p_mask_t1[:real_num_t1] = True
                
                t1_dict = {
                    'gt_global_map': den_g1[b].unsqueeze(0).cuda(),
                    'gt_shared_map': den_s1[b].unsqueeze(0).cuda(),
                    'gt_out_map': den_g1[b].unsqueeze(0).cuda(), 'gt_in_map': den_g1[b].unsqueeze(0).cuda(),
                    'points': pts_t1,
                    'share_mask0': p_mask_t1, 'share_mask1': p_mask_t1,
                    'share_mask': p_mask_t1, 'outflow_mask': p_mask_t1, 'inflow_mask': p_mask_t1,
                    'roi_mask': torch.ones_like(den_g1[b]).cuda(), 'mask': torch.ones_like(den_g1[b]).cuda(),
                    'image_path': str(name_list[b])
                }
                target.append(t1_dict)

                # --- Frame 2 ---
                pts_t2 = pts2[b].float().to(device)
                real_num_t2 = cnt2[b].item()
                p_mask_t2 = torch.zeros(pts_t2.shape[0], dtype=torch.bool).to(device)
                if real_num_t2 > 0: p_mask_t2[:real_num_t2] = True

                t2_dict = {
                    'gt_global_map': den_g2[b].unsqueeze(0).cuda(),
                    'gt_shared_map': den_s2[b].unsqueeze(0).cuda(),
                    'gt_out_map': den_g2[b].unsqueeze(0).cuda(), 'gt_in_map': den_g2[b].unsqueeze(0).cuda(),
                    'points': pts_t2,
                    'share_mask0': p_mask_t2, 'share_mask1': p_mask_t2,
                    'share_mask': p_mask_t2, 'outflow_mask': p_mask_t2, 'inflow_mask': p_mask_t2,
                    'roi_mask': torch.ones_like(den_g2[b]).cuda(), 'mask': torch.ones_like(den_g2[b]).cuda(),
                    'image_path': str(name_list[b])
                }
                target.append(t2_dict)

            # 现在 target 长度是 2，模型就不会报错了
            pre_global_den, _, _, _, _, _, _ = model(img_input, target)
            pre_global_den = torch.relu(pre_global_den)

            # === 应用裁边 ===
            # 去除边缘 16 像素的噪声
            clean_pred = post_process_density(pre_global_den, crop_size=16)

            # === 可视化 Frame 1 ===
            b_idx = 0
            fname = name_list[b_idx]
            
            # 计算数值 (不除以100，因为 VIC 内部已除)
            pred_cnt = clean_pred[0].sum().item()
            gt_cnt = cnt1[b_idx].item()
            
            # 绘图
            fig = plt.figure(figsize=(10, 4))
            plt.suptitle(f"{fname}\nGT: {gt_cnt} | Pred: {pred_cnt:.2f}", fontsize=12)

            # 1. 原图
            plt.subplot(1, 3, 1)
            plt.imshow(denormalize(img1[b_idx]))
            plt.title("Input")
            plt.axis('off')

            # 2. 真值
            plt.subplot(1, 3, 2)
            gt_map = den_g1[b_idx, 0].cpu().numpy()
            plt.imshow(gt_map, cmap='jet')
            plt.title("Ground Truth")
            plt.axis('off')

            # 3. 预测 (用裁边后的图)
            plt.subplot(1, 3, 3)
            pred_map = clean_pred[0, 0].cpu().numpy()
            plt.imshow(pred_map, cmap='jet')
            plt.title("Prediction")
            plt.axis('off')

            # 保存
            save_path = os.path.join(SAVE_DIR, f"result_{fname}")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)

    print(f"\n✅ 全部完成！图片已保存在: {SAVE_DIR}")

if __name__ == '__main__':
    run_save_all()