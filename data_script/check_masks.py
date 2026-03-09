import os
import shutil
import glob
from tqdm import tqdm

# ================= 关键配置区域 =================
# 1. 你现在有图片的那个文件夹 (刚才划分好的根目录)
SPLIT_ROOT = r"D:\DeepFish_Split_Root"

# 2. 原始的 Mask 仓库 (DeepFish原数据集里的 masks 文件夹)
# 请务必确认这个路径下有一堆 .png 或 .jpg 的黑白图！
SOURCE_MASKS_DIR = r"D:\deepfish\DeepFish\Localization\masks\valid"


# ===============================================

def sync_masks():
    # --- 第1步：检查“仓库” ---
    if not os.path.exists(SOURCE_MASKS_DIR):
        print(f"❌ 错误：找不到原始 Mask 仓库: {SOURCE_MASKS_DIR}")
        return

    # 建立库存清单 (加速查找)
    print("正在清点原始 Mask 库存...")
    mask_files = os.listdir(SOURCE_MASKS_DIR)
    # 做成字典: {'文件名无后缀': '完整文件名'}
    # 例如: {'Hab1_100': 'Hab1_100.png'}
    mask_inventory = {os.path.splitext(f)[0]: f for f in mask_files}

    print(f"✅ 库存就绪，共有 {len(mask_inventory)} 个 Mask 文件。")

    # --- 第2步：开始分发 ---
    total_copied = 0
    total_no_mask = 0

    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(SPLIT_ROOT, split, 'images')
        target_mask_dir = os.path.join(SPLIT_ROOT, split, 'masks')

        # 如果 images 文件夹不存在，跳过
        if not os.path.exists(img_dir):
            print(f"⚠️ {split} 集的 images 文件夹不存在，跳过。")
            continue

        # 创建 masks 文件夹
        os.makedirs(target_mask_dir, exist_ok=True)

        # 获取该集下所有图片
        images = os.listdir(img_dir)
        print(f"\n正在为 {split} 集 ({len(images)} 张图片) 匹配 Mask...")

        split_cnt = 0

        for img_name in tqdm(images):
            # 1. 拿到图片的主名 (比如 Hab1_100)
            key = os.path.splitext(img_name)[0]

            # 2. 去库存里找有没有同名的 Mask
            if key in mask_inventory:
                mask_filename = mask_inventory[key]  # 找到了！(比如 Hab1_100.png)

                src_path = os.path.join(SOURCE_MASKS_DIR, mask_filename)

                # 为了后续代码方便，我们把 Mask 存成和图片同名 (但保留原后缀或统一png)
                # 这里建议：文件名保持一致，后缀跟随原Mask
                dst_path = os.path.join(target_mask_dir, mask_filename)

                # 3. 复制过去 (优先用硬链接，省空间)
                if not os.path.exists(dst_path):
                    try:
                        os.link(src_path, dst_path)
                    except:
                        shutil.copy(src_path, dst_path)

                split_cnt += 1
            else:
                # 没找到 Mask (正常，说明这是 No-Fish 背景图)
                total_no_mask += 1

        total_copied += split_cnt
        print(f"   -> {split}: 匹配成功 {split_cnt} 张")

    print("\n" + "=" * 30)
    print("同步完成！")
    print(f"成功匹配 Mask: {total_copied} 张")
    print(f"无 Mask 图片 : {total_no_mask} 张 (DeepFish中这是正常的背景图)")
    print(f"检查文件夹: {SPLIT_ROOT}")
    print("=" * 30)


if __name__ == '__main__':
    sync_masks()