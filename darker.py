import cv2
import os
import numpy as np
from tqdm import tqdm


def darker(data_dir, ratio=0.5, phase="train"):
    """
    生成低光照图像。

    Args:
        data_dir (str): 数据集根目录。
        ratio (float): 亮度降低的比例，0到1之间。
        phase (str): 阶段，可选值为 "train" 或 "test"。
    """

    high_dir = os.path.join(data_dir, "our485", "high") if phase == "train" else os.path.join(
        data_dir, "eval15", "high")
    low_dir = os.path.join(data_dir, "our485", "low") if phase == "train" else os.path.join(
        data_dir, "eval15", "low")

    # 确保 low 目录存在
    os.makedirs(low_dir, exist_ok=True)

    # 获取所有高光照图像的文件名
    image_files = [f for f in os.listdir(high_dir) if f.endswith(
        ('.png', '.jpg', '.jpeg', '.bmp'))]

    print("开始处理数据集...")
    pbar = tqdm(image_files)
    index = 0
    for image_file in pbar:
        # 读取高光照图像
        high_img_path = os.path.join(high_dir, image_file)
        high_img = cv2.imread(high_img_path)

        if high_img is None:
            print(f"无法读取图像: {high_img_path}")
            continue

        # 转换为 HSV 色彩空间
        hsv_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2HSV)

        # 降低 V (亮度) 通道的值
        hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2].astype(
            np.float32) * ratio, 0, 255).astype(np.float32)

        # 转换回 BGR 色彩空间
        dark_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        # 构建低光照图像的保存路径
        low_img_path = os.path.join(
            low_dir, f"{image_file}")

        # 保存低光照图像
        cv2.imwrite(low_img_path, dark_img)

        index += 1

    print("处理完成！请手动检查数据集。")


if __name__ == '__main__':
    # 1. 设置数据集目录
    data_dir = "../datasets/kitti_LOL"  # 替换成你的数据集路径
    ratio = 0.1  # 亮度降低的比例

    # 2. 生成低光照图像
    darker(data_dir, ratio=ratio, phase="train")
    darker(data_dir, ratio=ratio, phase="test")
