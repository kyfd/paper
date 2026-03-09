import os
import shutil
import pandas as pd
from tqdm import tqdm

# ==================== 【配置区域】 ====================
# 1. 原始数据根目录 (指向包含 empty 和 valid 文件夹的父目录)
# 你的截图显示 images 下面有 empty/valid，所以指向 images 这一层
SOURCE_IMG_ROOT = r"/root/autodl-tmp/Localization/images" 
SOURCE_MASK_ROOT = r"/root/autodl-tmp/Localization/masks"

# 2. CSV 文件路径
CSV_PATHS = {
    'train': r"/root/autodl-tmp/Localization/train.csv",
    'val':   r"/root/autodl-tmp/Localization/val.csv",
    'test':  r"/root/autodl-tmp/Localization/test.csv"
}

# 3. 目标输出路径 (会自动创建)
TARGET_ROOT = r"/root/autodl-tmp/DeepFish"
# ====================================================

def split_dataset():
    for split_name, csv_path in CSV_PATHS.items():
        if not os.path.exists(csv_path):
            print(f"⚠️ 跳过 {split_name}，找不到CSV: {csv_path}")
            continue

        print(f"\n>>> 正在处理: {split_name} 集...")
        
        # 1. 创建目标目录 (我们将把 empty 和 valid 的图混在一起放，方便后续处理)
        save_img_dir = os.path.join(TARGET_ROOT, split_name, 'images')
        save_mask_dir = os.path.join(TARGET_ROOT, split_name, 'masks')
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_mask_dir, exist_ok=True)

        # 2. 读取 CSV
        df = pd.read_csv(csv_path)
        
        # 【关键修改】使用 'ID' 列，因为它包含了 'empty/xxx' 或 'valid/xxx' 的路径
        if 'ID' not in df.columns:
            print("❌ CSV 中缺少 'ID' 列，无法解析子文件夹路径！")
            return
            
        id_list = df['ID'].tolist()
        
        success_count = 0
        missing_count = 0

        for rel_path in tqdm(id_list):
            # rel_path 可能是 "valid/7117_xxx"
            # 1. 构造源图片路径 (ID列通常不带后缀，需要补 .jpg)
            src_img_path = os.path.join(SOURCE_IMG_ROOT, rel_path + ".jpg")
            
            # 2. 构造文件名 (为了扁平化存储，我们只保留文件名)
            file_name = os.path.basename(rel_path) # 变成 7117_xxx
            dst_img_path = os.path.join(save_img_dir, file_name + ".jpg")
            
            # 3. 复制图片
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
                success_count += 1
                
                # 4. 处理 Mask (Mask 通常都在一个文件夹里，或者是 valid 下才有)
                # 我们尝试去找对应的 Mask
                # Mask 文件名通常是 file_name + .png
                mask_name = file_name + ".png"
                src_mask_path = os.path.join(SOURCE_MASK_ROOT, mask_name)
                
                # 如果 Mask 也是分文件夹的，尝试拼接 rel_path
                if not os.path.exists(src_mask_path):
                     src_mask_path = os.path.join(SOURCE_MASK_ROOT, rel_path + ".png")
                
                # 如果找到了 Mask，就复制
                if os.path.exists(src_mask_path):
                    shutil.copy(src_mask_path, os.path.join(save_mask_dir, mask_name))
                
            else:
                missing_count += 1
                # print(f"缺失: {src_img_path}")

        print(f"✅ {split_name} 完成！成功: {success_count}, 缺失: {missing_count}")

if __name__ == '__main__':
    split_dataset()