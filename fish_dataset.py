import torch
from torch.utils.data import Dataset
import glob
import os
import h5py
import cv2
import numpy as np
from torchvision import transforms
import random


class MiniFishDataset(Dataset):
    def __init__(self, root_path, train=True):
        self.img_dir = os.path.join(root_path, "images")
        self.h5_dir = os.path.join(root_path, "h5")
        self.train = train
        
        # === 核心参数：512 (必须与 generate_h5.py 一致) ===
        self.target_size = (512, 512) 

        self.data_list = sorted(glob.glob(os.path.join(self.h5_dir, "*.h5")))
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        self.name_to_idx = {os.path.basename(p): i for i, p in enumerate(self.img_list)}

        if len(self.data_list) == 0:
            print(f"警告：在 {self.h5_dir} 没找到数据！")

        self.transform_base = transforms.Compose([
            transforms.ToPILImage(),
            # 训练时保留颜色抖动，这是安全的增强
            transforms.ColorJitter(brightness=0.2, contrast=0.2) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        h5_path = self.data_list[item]
        name1 = os.path.basename(h5_path).replace('.h5', '.jpg')
        img1_path = os.path.join(self.img_dir, name1)

        # 找下一帧
        if name1 in self.name_to_idx:
            idx1 = self.name_to_idx[name1]
            idx2 = idx1 + 1 if idx1 + 1 < len(self.img_list) else idx1
            img2_path = self.img_list[idx2]
        else:
            img2_path = img1_path 

        # 读取图片 (512)
        def load_img(path):
            img = cv2.imread(path)
            if img is None: return np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size) 
            return img

        img1 = load_img(img1_path)
        img2 = load_img(img2_path)

        # 读取 H5 (假定已经是 512)
        with h5py.File(h5_path, 'r') as hf:
            den_g1 = np.array(hf['global_t1'])
            den_s1 = np.array(hf['shared_t1'])
            den_g2 = np.array(hf['global_t2'])
            den_s2 = np.array(hf['shared_t2'])
            pts1 = np.array(hf['points_t1'])
            cnt1 = int(np.array(hf['count_t1']))
            pts2 = np.array(hf['points_t2'])
            cnt2 = int(np.array(hf['count_t2']))

       

        # 转 Tensor
        img1 = self.transform_base(img1)
        img2 = self.transform_base(img2)

        den_g1 = torch.from_numpy(den_g1).unsqueeze(0)
        den_s1 = torch.from_numpy(den_s1).unsqueeze(0)
        den_g2 = torch.from_numpy(den_g2).unsqueeze(0)
        den_s2 = torch.from_numpy(den_s2).unsqueeze(0)

        return img1, img2, den_g1, den_s1, den_g2, den_s2, pts1, cnt1, pts2, cnt2, name1