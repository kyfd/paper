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
from misc.utils import AverageMeter
from model.VIC import Video_Counter
from datasets.fish_dataset import MiniFishDataset

# ==================== 【配置区域】 ====================
TEST_ROOT = "/root/autodl-tmp/DeepFish/test"
# 请确认这是你最好的那个模型路径
MODEL_PATH = "/root/autodl-tmp/FishCount_Project/exp/DeepFish/12-30_09-57_DeepFish_1e-05_train/output/best_model.pth"
SAVE_DIR = "/root/autodl-tmp/DeepFish/test_results_metrics"
# ====================================================

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std) + mean
    return np.clip(img, 0, 1)

def post_process_density(den_map, crop_size=16):
    """裁边去噪"""
    den = den_map.clone()
    h, w = den.shape[-2], den.shape[-1]
    den[..., :crop_size, :] = 0
    den[..., h-crop_size:, :] = 0
    den[..., :, :crop_size] = 0
    den[..., :, w-crop_size:] = 0
    return den

# ================== 新增：GAME 计算核心函数 ==================
def compute_game(pred, gt, L=0):
    """
    计算 GAME(L) 指标
    L=0: 全图 MAE
    L=1: 切成 2x2=4 个格子计算
    L=2: 切成 4x4=16 个格子计算
    L=3: 切成 8x8=64 个格子计算
    """
    # pred, gt: [1, H, W] Tensor
    n = 2 ** L # 格子划分数 (2^L)
    H, W = pred.shape[-2], pred.shape[-1]
    
    # 计算每个格子的大小
    h_stride = H // n
    w_stride = W // n
    
    game_error = 0.0
    
    # 遍历所有格子
    for i in range(n):
        for j in range(n):
            # 切片取出当前格子
            p_grid = pred[..., i*h_stride:(i+1)*h_stride, j*w_stride:(j+1)*w_stride]
            g_grid = gt[..., i*h_stride:(i+1)*h_stride, j*w_stride:(j+1)*w_stride]
            
            # 计算该格子的计数误差
            game_error += abs(p_grid.sum().item() - g_grid.sum().item())
            
    return game_error
# =============================================================

def run_test():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    device = torch.device('cuda')

    # 1. 加载模型
    print(f">>> 正在加载模型: {MODEL_PATH}")
    cfg_data = edict(); cfg_data.DEN_FACTOR = 100; cfg_data.VAL_FRAME_INTERVALS = 1
    model = Video_Counter(cfg, cfg_data).to(device)
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH)
        new_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_dict)
        model.eval()
    else:
        print("❌ 错误：找不到权重文件")
        return

    # 2. 加载数据 (Batch=1 最稳)
    test_dataset = MiniFishDataset(TEST_ROOT, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 定义指标记录器
    mae_meter = AverageMeter()
    mse_meter = AverageMeter()
    game1_meter = AverageMeter() # GAME(1)
    game2_meter = AverageMeter() # GAME(2)
    game3_meter = AverageMeter() # GAME(3)

    print(f"开始测试 {len(test_dataset)} 张图片...")

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img1, img2, den_g1, den_s1, den_g2, den_s2, pts1, cnt1, pts2, cnt2, name_list = data
            
            imgs_stacked = torch.stack([img1, img2], dim=1)
            img_input = imgs_stacked.view(-1, 3, 512, 512).to(device)

            # Target 构造 (Batch=1 极简版)
            pts_t1 = pts1[0].float().to(device)
            den_t1 = den_g1[0].cuda()
            cnt_t1 = cnt1[0].item()
            mask_t1 = torch.zeros(pts_t1.shape[0], dtype=torch.bool).to(device)
            if cnt_t1 > 0: mask_t1[:cnt_t1] = True
            
            # 这里只需构造 T1 的 Target 即可，为了跑通
            target = [{
                'points': pts_t1,
                'share_mask0': mask_t1, 'share_mask1': mask_t1, 'share_mask': mask_t1,
                'outflow_mask': mask_t1, 'inflow_mask': mask_t1,
                'gt_global_map': den_t1, 'gt_shared_map': den_s1[0].cuda(),
                'gt_out_map': den_t1, 'gt_in_map': den_t1,
                'roi_mask': torch.ones_like(den_t1).cuda(), 'mask': torch.ones_like(den_t1).cuda(),
                'image_path': str(name_list[0])
            }] * 2 # T1, T2 复制
            
            # 推理
            pre_global_den, _, _, _, _, _, _ = model(img_input, target)
            pre_global_den = torch.relu(pre_global_den)

            # === 计算指标 (针对 T1) ===
            # 1. 拿到预测密度图和真值密度图
            # 记得裁边！
            pred_map = post_process_density(pre_global_den[0], crop_size=16)
            gt_map = den_g1[0].cuda()

            # 2. 计算 MAE / MSE
            pred_cnt = pred_map.sum().item()
            gt_cnt = gt_map.sum().item()
            
            mae_meter.update(abs(pred_cnt - gt_cnt))
            mse_meter.update((pred_cnt - gt_cnt) ** 2)

            # 3. 计算 GAME (核心新增)
            game1_meter.update(compute_game(pred_map, gt_map, L=1))
            game2_meter.update(compute_game(pred_map, gt_map, L=2))
            game3_meter.update(compute_game(pred_map, gt_map, L=3))

    print("\n" + "=" * 50)
    print(f"📊 最终测试结果 (DeepFish Benchmark):")
    print(f"--------------------------------------------------")
    print(f"Counting Metrics:")
    print(f"  MAE (GAME-0) : {mae_meter.avg:.4f}  (Baseline: 0.38 / 1.22)")
    print(f"  MSE          : {np.sqrt(mse_meter.avg):.4f}")
    print(f"--------------------------------------------------")
    print(f"Localization Metrics (GAME):")
    print(f"  GAME(1)      : {game1_meter.avg:.4f}  (Grid: 2x2)")
    print(f"  GAME(2)      : {game2_meter.avg:.4f}  (Grid: 4x4, Baseline: 1.22)")
    print(f"  GAME(3)      : {game3_meter.avg:.4f}  (Grid: 8x8)")
    print("=" * 50)

if __name__ == '__main__':
    run_test()