# step3_real_data_mask_final.py
import os
import glob
import cv2
import h5py
import numpy as np
import scipy.spatial
from scipy.ndimage import gaussian_filter


TARGET_SIZE = 512

# 2. 数据集路径 (请每次运行前修改这里，跑三次：train, val, test)
# 当前配置: ---> 验证集 (val) <---
IMG_FOLDER = "/root/autodl-tmp/DeepFish/val/images"
MASK_FOLDER = "/root/autodl-tmp/DeepFish/val/masks"
OUTPUT_FOLDER = "/root/autodl-tmp/DeepFish/val/h5"


# ===================================================

def generate_density(img_shape, points):
    density = np.zeros(img_shape, dtype=np.float32)
    for p in points:
        x, y = int(p[0]), int(p[1])

        # 【安全保险】防止 0,0 坐标导致的边缘异常
        if x == 0 and y == 0:
            continue

        if 0 <= y < img_shape[0] and 0 <= x < img_shape[1]:
            density[y, x] = 1

    # === 修改这里：适配 512 分辨率 ===
    if img_shape[0] <= 256:
        sigma = 4
    elif img_shape[0] <= 512:
        sigma = 8   # 512分辨率用8比较合适
    else:
        sigma = 10  # 更大分辨率用10
        
    return gaussian_filter(density, sigma=sigma, mode='constant')


def get_points_from_mask(mask_path):
    mask = cv2.imread(mask_path, 0)
    if mask is None: return None
    # 提取连通域中心
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    # 第0个是背景，排除
    if num_labels > 1:
        return centroids[1:]
    return []


def pad_points(points, max_len=20):
    padded = np.zeros((max_len, 2))
    points = np.array(points)
    num = len(points)
    if num > max_len:
        padded = points[:max_len]
        num = max_len
    elif num > 0:
        padded[:num] = points
    return padded, num


def find_mask_file(mask_folder, img_name_no_ext):
    # 自动尝试多种后缀
    candidates = [
        f"{img_name_no_ext}.png",
        f"{img_name_no_ext}.jpg",
        f"{img_name_no_ext}_mask.png",
        f"{img_name_no_ext}_mask.jpg"
    ]
    for c in candidates:
        path = os.path.join(mask_folder, c)
        if os.path.exists(path): return path
    return None


def run():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    # 获取所有 jpg 图片 (不管有没有对应的 mask)
    img_paths = sorted(glob.glob(os.path.join(IMG_FOLDER, "*.jpg")))
    print(f"源目录: {IMG_FOLDER}")
    print(f"找到 {len(img_paths)} 张图片，开始全量处理 (含无鱼图)...")

    count_ok = 0
    count_no_fish = 0 # 统计一下多少张没鱼

    for i in range(len(img_paths) - 1):
        path_t1 = img_paths[i]
        path_t2 = img_paths[i + 1]
        
        # 获取文件名 (不带后缀)
        name_no_ext_1 = os.path.splitext(os.path.basename(path_t1))[0]
        name_no_ext_2 = os.path.splitext(os.path.basename(path_t2))[0]

        # 1. 尝试找 Mask
        mask_path_t1 = find_mask_file(MASK_FOLDER, name_no_ext_1)
        mask_path_t2 = find_mask_file(MASK_FOLDER, name_no_ext_2)

        # 2. 提取坐标 (关键修改点！)
        # 如果找到了 Mask，就读坐标；如果没找到，就给个空列表 []
        if mask_path_t1 is not None:
            pts1 = get_points_from_mask(mask_path_t1)
            # 如果 mask 读取失败(比如坏文件)，也当做空
            if pts1 is None: pts1 = []
        else:
            pts1 = [] # 无 Mask = 无鱼

        if mask_path_t2 is not None:
            pts2 = get_points_from_mask(mask_path_t2)
            if pts2 is None: pts2 = []
        else:
            pts2 = []
            
        # 统计一下
        if len(pts1) == 0: count_no_fish += 1

        # 3. 读原图算缩放
        img = cv2.imread(path_t1)
        if img is None: continue
        h, w = img.shape[:2]
        sx, sy = TARGET_SIZE / w, TARGET_SIZE / h

        # 4. 缩放坐标 (如果有坐标才缩放)
        if len(pts1) > 0: pts1 = pts1 * [sx, sy]
        if len(pts2) > 0: pts2 = pts2 * [sx, sy]

        # 5. 生成密度图 
        # (如果 pts 为空，generate_density 会自动生成全 0 的黑图，非常完美)
        den_g1 = generate_density((TARGET_SIZE, TARGET_SIZE), pts1)
        den_g2 = generate_density((TARGET_SIZE, TARGET_SIZE), pts2)
        den_s1, den_s2 = den_g1, den_g2

        # 6. 填充坐标
        p1_pad, n1 = pad_points(pts1)
        p2_pad, n2 = pad_points(pts2)

        # 7. 保存 H5
        save_name = os.path.basename(path_t1).replace('.jpg', '.h5')
        save_path = os.path.join(OUTPUT_FOLDER, save_name)

        with h5py.File(save_path, 'w') as hf:
            hf['global_t1'] = den_g1
            hf['global_t2'] = den_g2
            hf['shared_t1'] = den_s1
            hf['shared_t2'] = den_s2
            hf['points_t1'] = p1_pad
            hf['count_t1'] = n1
            hf['points_t2'] = p2_pad
            hf['count_t2'] = n2

        count_ok += 1
        if count_ok % 100 == 0: print(f"已处理 {count_ok} 张...")

    print(f"处理完成！共生成 {count_ok} 个文件。")
    print(f"其中无鱼图片 (Negative Samples): {count_no_fish} 张")


if __name__ == '__main__':
    run()