import os
import torch
import numpy as np
from tqdm import tqdm
from misc import tools
from config import cfg
from torch import optim
from misc.utils import *
from copy import deepcopy
import torch.nn.functional as F
from torch.nn import SyncBatchNorm
from model.VIC import Video_Counter
from misc.tools import is_main_process
from torch.utils.data import DataLoader
from datasets.fish_dataset import MiniFishDataset
from torchvision import transforms as standard_transforms
from easydict import EasyDict as edict

class Trainer():
    def __init__(self, cfg_data, pwd):
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        # ================= 1. 模型初始化 =================
        self.model = self.model_without_ddp = Video_Counter(cfg, cfg_data)
        self.model.cuda()
        self.val_frame_intervals = cfg_data.VAL_FRAME_INTERVALS

        if cfg.distributed:
            sync_model = SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                sync_model, device_ids=[cfg.gpu], find_unused_parameters=False)
            self.model_without_ddp = self.model.module

        # ================= 2. 数据加载 (适配 AutoDL 服务器) =================
        # 【注意】请根据你的实际解压位置修改，通常在 /root/autodl-tmp/ 下
        train_root = "/root/autodl-tmp/DeepFish/train"
        val_root   = "/root/autodl-tmp/DeepFish/val"

        if not os.path.exists(train_root):
            raise FileNotFoundError(f"找不到训练集: {train_root}，请检查路径是否正确！")

        print(f"=== 正在加载训练集: {train_root} ===")
        train_dataset = MiniFishDataset(train_root, train=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=2,      # 3090显卡建议开到 8 或 16
            shuffle=True,      # 训练必须打乱
            num_workers=4,     # Linux下建议开多线程 (4~8)
            drop_last=True     # 必须丢弃不成对的数据
        )
        print(f"训练集加载成功，共 {len(train_dataset)} 张图片")

        if os.path.exists(val_root):
            print(f"=== 正在加载验证集: {val_root} ===")
            val_dataset = MiniFishDataset(val_root, train=False)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=4,      # 验证集也可以开大Batch
                shuffle=False, 
                num_workers=4,
                drop_last=True
            )
        else:
            print(f"警告：未找到验证集 {val_root}，将跳过验证。")
            self.val_loader = None
        
        self.restore_transform = standard_transforms.Compose([
            standard_transforms.ToPILImage()
        ])

        # ================= 3. 优化器 =================
        param_groups = [{'params': self.model_without_ddp.parameters(), 'weight_decay': cfg.WEIGHT_DECAY}]
        self.optimizer = optim.Adam(param_groups, lr=cfg.LR_Base)
        
        self.i_tb = 0
        self.epoch = 1
        self.num_iters = cfg.MAX_EPOCH * len(self.train_loader)

        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.train_record = {'best_mae': 1e20}

        # 加载断点
        if cfg.RESUME:
            print(f"正在恢复训练: {cfg.RESUME_PATH}")
            latest_state = torch.load(cfg.RESUME_PATH)
            self.model.load_state_dict(latest_state['net'], strict=True)
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.epoch = latest_state['epoch']
            self.i_tb = latest_state['i_tb']

        # 加载预训练权重
        if cfg.PRE_TRAIN_COUNTER and os.path.exists(cfg.PRE_TRAIN_COUNTER):
            try:
                counting_pre_train = torch.load(cfg.PRE_TRAIN_COUNTER)
                model_dict = self.model.state_dict()
                new_dict = {k: v for k, v in counting_pre_train.items() if k in model_dict}
                model_dict.update(new_dict)
                self.model.load_state_dict(model_dict, strict=False)
                print("成功加载预训练权重")
            except Exception as e:
                print(f"预训练权重加载失败: {e}")
        else:
            print(f"提示：未加载预训练权重 (路径不存在或未设置): {cfg.PRE_TRAIN_COUNTER}")

        if is_main_process():
            self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd,
                                               ['exp', 'eval', 'figure', 'img', 'vis', 'output', 'visual_results'], 
                                               resume=cfg.RESUME)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH + 1):
            self.epoch = epoch
            
            # --- 训练 ---
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)
            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            
            # --- 保存模型 ---
            save_folder = os.path.join(self.exp_path, self.exp_name, 'output')
            if not os.path.exists(save_folder): os.makedirs(save_folder)
            
            save_path = os.path.join(save_folder, f'epoch_{epoch}.pth')
            if cfg.distributed:
                torch.save(self.model.module.state_dict(), save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
            print(f"【保存】模型已保存至: {save_path}")

            # --- 验证 ---
            if self.val_loader is not None and epoch % 1 == 0:
                mae = self.validate()
                if mae < self.train_record['best_mae']:
                    self.train_record['best_mae'] = mae
                    best_path = os.path.join(save_folder, 'best_model.pth')
                    torch.save(self.model.state_dict(), best_path)
                    print(f"★ 新纪录！最佳模型已保存 (MAE: {mae:.2f})")

            print('=' * 20)

    # =================================================================================
    # 核心辅助函数：构造 Target
    # =================================================================================
    def build_targets_list(self, img1, den_g1, den_s1, pts1, cnt1,
                                 img2, den_g2, den_s2, pts2, cnt2, name_list, device):
        """
        将 Batch 数据拆解并按顺序构造成列表: [Sample1_T1, Sample1_T2, Sample2_T1, Sample2_T2 ...]
        """
        target = []
        batch_size_curr = img1.shape[0]
        
        for b in range(batch_size_curr):
            # --- Frame 1 ---
            pts_t1 = pts1[b].float().to(device)
            den_t1 = den_g1[b].cuda()
            real_num_t1 = cnt1[b].item()
            
            point_mask_t1 = torch.zeros(pts_t1.shape[0], dtype=torch.bool).to(device)
            if real_num_t1 > 0: point_mask_t1[:real_num_t1] = True
            
            dict_t1 = {
                'gt_global_map': den_t1,
                'gt_shared_map': den_s1[b].cuda(),
                'gt_out_map': den_t1.clone(), 
                'gt_in_map': den_t1.clone(),
                'points': pts_t1,
                'share_mask0': point_mask_t1, 'share_mask1': point_mask_t1,
                'share_mask': point_mask_t1, 'outflow_mask': point_mask_t1, 'inflow_mask': point_mask_t1,
                'roi_mask': torch.ones_like(den_t1).to(device),
                'mask': torch.ones_like(den_t1).to(device),
                'image_path': str(name_list[b])
            }
            target.append(dict_t1)
            
            # --- Frame 2 ---
            pts_t2 = pts2[b].float().to(device)
            den_t2 = den_g2[b].cuda()
            real_num_t2 = cnt2[b].item()
            
            point_mask_t2 = torch.zeros(pts_t2.shape[0], dtype=torch.bool).to(device)
            if real_num_t2 > 0: point_mask_t2[:real_num_t2] = True
            
            dict_t2 = {
                'gt_global_map': den_t2,
                'gt_shared_map': den_s2[b].cuda(),
                'gt_out_map': den_t2.clone(),
                'gt_in_map': den_t2.clone(),
                'points': pts_t2,
                'share_mask0': point_mask_t2, 'share_mask1': point_mask_t2,
                'share_mask': point_mask_t2, 'outflow_mask': point_mask_t2, 'inflow_mask': point_mask_t2,
                'roi_mask': torch.ones_like(den_t2).to(device),
                'mask': torch.ones_like(den_t2).to(device),
                'image_path': str(name_list[b])
            }
            target.append(dict_t2)
            
        return target

    def train(self):
        self.model.train()
        lr = adjust_learning_rate(self.optimizer, cfg.LR_Base, self.num_iters, self.i_tb)
        batch_loss = {}

        for i, data in enumerate(self.train_loader, 0):
            self.i_tb += 1
            # 1. 接收数据
            img1, img2, den_g1, den_s1, den_g2, den_s2, pts1, cnt1, pts2, cnt2, name_list = data
            
            # 2. 拼装图片 [B, 2, 3, H, W] -> [2B, 3, H, W]
            imgs_stacked = torch.stack([img1, img2], dim=1)
            # 【注意】如果你在generate_h5里改了512分辨率，这里也要改成 512
            img = imgs_stacked.view(-1, 3, 512, 512).cuda() 

            # 3. 构造 Target 列表 (使用辅助函数)
            target = self.build_targets_list(img1, den_g1, den_s1, pts1, cnt1,
                                             img2, den_g2, den_s2, pts2, cnt2, name_list, img.device)

            # 4. 前向传播
            pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, loss_dict = self.model(img, target)
            
            pre_global_den = torch.relu(pre_global_den)
            pre_global_cnt = pre_global_den.sum()
            gt_global_cnt = gt_global_den.sum()

            # 5. 反向传播
            all_loss = sum(loss_dict.values())
            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()

            # 6. 日志
            loss_dict_reduced = reduce_dict(loss_dict)
            for k, v in loss_dict_reduced.items():
                if k not in batch_loss: batch_loss[k] = AverageMeter()
                batch_loss[k].update(v.item())

            if self.i_tb % cfg.PRINT_FREQ == 0:
                if is_main_process():
                    self.writer.add_scalar('lr', lr, self.i_tb)
                    for k, v in loss_dict_reduced.items():
                        self.writer.add_scalar(k, v.item(), self.i_tb)
                    
                    loss_str = ''.join([f"[{k} {v.avg:.4f}]" for k, v in batch_loss.items()])
                    print(f"[ep {self.epoch}][it {self.i_tb}]{loss_str} [gt: {gt_global_cnt.item():.1f} pre: {pre_global_cnt.item():.1f}]")

            # 7. 可视化保存
            if self.i_tb % 50 == 0:
                try:
                    save_visual_results(
                        [img, 
                         gt_global_den, pre_global_den, 
                         gt_share_den, pre_share_den, 
                         gt_in_out_den, pre_in_out_den], 
                        self.restore_transform, 
                        os.path.join(self.exp_path, self.exp_name, "training_visual"), 
                        self.i_tb, 0
                    )
                except Exception as e:
                    print(f"可视化保存失败: {e}")

    def validate(self):
        self.model.eval()
        mae_meter = AverageMeter()
        print(">>> 开始验证集测试...")
        
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                try:
                    # 1. 接收数据
                    img1, img2, den_g1, den_s1, den_g2, den_s2, pts1, cnt1, pts2, cnt2, name_list = data
                    
                    # 2. 拼装图片
                    imgs_stacked = torch.stack([img1, img2], dim=1)
                    # 【注意】这里也要改成 512
                    img = imgs_stacked.view(-1, 3, 512, 512).cuda()
                    
                    # 3. 构造 Target
                    target = self.build_targets_list(img1, den_g1, den_s1, pts1, cnt1,
                                                     img2, den_g2, den_s2, pts2, cnt2, name_list, img.device)
                    
                    # 4. 预测
                    res = self.model(img, target)
                    pre_global = torch.relu(res[0])
                    
                    # 5. 计算指标 (模型已除过DEN_FACTOR，此处无需再除)
                    pred_sum = pre_global.view(pre_global.shape[0], -1).sum(dim=1)
                    
                    gt_global = torch.cat([den_g1, den_g2], dim=0).cuda()
                    gt_sum = gt_global.view(gt_global.shape[0], -1).sum(dim=1)
                    
                    diff = torch.abs(pred_sum - gt_sum).mean()
                    mae_meter.update(diff.item())
                    
                    if i == 0:
                         print(f" [Debug] GT: {gt_sum.mean().item():.2f} | Pred: {pred_sum.mean().item():.2f}")

                except Exception as e:
                    print(f"验证集 Batch {i} 错误: {e}")
                    pass

        print(f"验证结束 | 平均 MAE: {mae_meter.avg:.2f}")
        return mae_meter.avg

if __name__ == '__main__':
    import os
    import numpy as np
    import torch
    from config import cfg
    from easydict import EasyDict as edict

    tools.init_distributed_mode(cfg)
    tools.set_randomseed(cfg.SEED + tools.get_rank())
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # 手动配置
    cfg_data = edict()
    cfg_data.DEN_FACTOR = 100  
    cfg_data.VAL_FRAME_INTERVALS = 1 

    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Trainer(cfg_data, pwd)
    cc_trainer.forward()