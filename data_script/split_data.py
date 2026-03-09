import os
import glob
import random
import shutil
from tqdm import tqdm
import numpy as np

# ================= 配置区域 =================
# 1. 原始图片所在的文件夹 (所有图片堆在一起的那个地方)
SOURCE_DIR = r"D:\deepfish\DeepFish\Localization\images\valid"

# 2. 想要输出的目标文件夹
TARGET_DIR = r"D:\DeepFish_Split_Root"

# 3. 划分比例 (训练 : 验证 : 测试)
SPLIT_RATIO = [0.7, 0.1, 0.2]

# 4. 模式: 'copy' (复制文件, 占空间) 或 'link' (硬链接, 极快且不占空间, 推荐!)
# 注意: link 模式要求源文件和目标文件在同一个磁盘分区 (比如都在 D 盘)
MODE = 'link'


# ===========================================

def get_habitat_name(filename):
    """
    根据文件名提取场景名称。
    假设文件名格式为: HabitatName_SeqID_FrameID.jpg
    例如: Low_complexity_reef_1_100.jpg -> Low_complexity_reef
    """
    # DeepFish 的文件名通常比较长，我们取最后一个下划线之前的部分作为场景名
    # 或者更简单：DeepFish通常以前缀区分场景
    # 这里我们尝试一种通用的提取方式：去掉最后两部分数字
    parts = filename.split('_')
    # 如果文件名里有数字编号，通常在最后。我们把非数字的前缀作为场景名
    habitat_parts = [p for p in parts if not p.isdigit() and not p.startswith('f0')]
    # 如果提取失败，就用前两个单词组合
    if len(habitat_parts) == 0:
        return "Unknown"
    return "_".join(habitat_parts)


def split_dataset():
    # 1. 准备目录
    for split in ['train', 'val', 'test']:
        dir_path = os.path.join(TARGET_DIR, split, 'images')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 2. 读取所有图片
    print("正在读取文件列表...")
    # 支持 jpg, png, jpeg
    all_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_files.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))

    if len(all_files) == 0:
        print("错误：没找到图片，请检查路径！")
        return

    print(f"共找到 {len(all_files)} 张图片。")

    # 3. 按场景分组 (Group by Habitat)
    scene_dict = {}
    for file_path in tqdm(all_files, desc="正在按场景分组"):
        filename = os.path.basename(file_path)
        # 提取场景名作为 Key
        # 针对 DeepFish 的文件名特点，通常直到倒数第2个下划线前都是场景名
        # 比如: 7398_F1_f000040.jpg -> 7398_F1 (或者更粗粒度)
        # 为了保险，我们直接用文件名前缀分组
        scene_name = filename.rsplit('_', 1)[0]  # 简单粗暴：除了帧号，前面都算场景

        if scene_name not in scene_dict:
            scene_dict[scene_name] = []
        scene_dict[scene_name].append(file_path)

    print(f"共识别出 {len(scene_dict)} 个独立的视频片段/场景。")

    # 4. 执行划分
    stats = {'train': 0, 'val': 0, 'test': 0}

    for scene, files in tqdm(scene_dict.items(), desc="正在划分并移动"):
        # (1) 排序：保证时间连续性！绝对不能乱序！
        files.sort()

        # (2) 计算切割点
        total = len(files)
        n_train = int(total * SPLIT_RATIO[0])
        n_val = int(total * SPLIT_RATIO[1])
        # 剩下的给 test

        # (3) 切分列表
        train_files = files[:n_train]
        val_files = files[n_train: n_train + n_val]
        test_files = files[n_train + n_val:]

        # (4) 定义移动函数
        def move_files(file_list, target_split):
            for src in file_list:
                fname = os.path.basename(src)
                dst = os.path.join(TARGET_DIR, target_split, 'images', fname)

                if os.path.exists(dst): continue  # 防止重复

                if MODE == 'link':
                    try:
                        os.link(src, dst)  # 创建硬链接
                    except:
                        shutil.copy(src, dst)  # 如果硬链接失败（跨盘），回退到复制
                else:
                    shutil.copy(src, dst)
                stats[target_split] += 1

        # (5) 执行移动
        move_files(train_files, 'train')
        move_files(val_files, 'val')
        move_files(test_files, 'test')

    print("\n" + "=" * 30)
    print("划分完成！")
    print(f"训练集: {stats['train']} 张")
    print(f"验证集: {stats['val']} 张")
    print(f"测试集: {stats['test']} 张")
    print(f"文件保存在: {TARGET_DIR}")
    print("=" * 30)


if __name__ == '__main__':
    split_dataset()