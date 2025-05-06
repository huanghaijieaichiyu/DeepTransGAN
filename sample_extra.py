# 从nuscenes数据集中提取不同地点的图片

import os
import json
import numpy as np
import PIL.Image as Image
import glob
from tqdm import tqdm
from darker import Darker
import cv2  # 添加 cv2 导入


def extract_images(nuscenes_data_path, output_path):

    image_path_front = glob.glob(os.path.join(
        nuscenes_data_path, 'samples', 'CAM_FRONT', '*.jpg'))

    image_path_front_left = glob.glob(os.path.join(
        nuscenes_data_path, 'samples', 'CAM_FRONT_LEFT', '*.jpg'))

    image_path_front_right = glob.glob(os.path.join(
        nuscenes_data_path, 'samples', 'CAM_FRONT_RIGHT', '*.jpg'))

    image_path_back = glob.glob(os.path.join(
        nuscenes_data_path, 'samples', 'CAM_BACK', '*.jpg'))

    image_path_back_left = glob.glob(os.path.join(
        nuscenes_data_path, 'samples', 'CAM_BACK_LEFT', '*.jpg'))

    image_path_back_right = glob.glob(os.path.join(
        nuscenes_data_path, 'samples', 'CAM_BACK_RIGHT', '*.jpg'))

    image_path = [image_path_front, image_path_front_left, image_path_front_right,
                  image_path_back, image_path_back_left, image_path_back_right]
    image_path.sort()

    # 每50个图片提取1张
    for i in tqdm(range(len(image_path))):
        for j in range(len(image_path[i])):
            if j % 50 == 0:
                image = Image.open(image_path[i][j])
                image.save(os.path.join(output_path, f'{i}_{j}.jpg'))


def darken_images(input_path, output_path):
    image_path_list = glob.glob(os.path.join(input_path, '*.jpg'))
    image_path_list.sort()
    # 实例化 Darker - data_dir 对于 adjust_image 不是必需的
    darker = Darker()
    # 定义参数
    base_ratio = 0.05
    headlight_mask_params = {
        'center_y_factor': 0.95,
        'beam_width_factor': 0.7,
        'falloff_sharpness': 3.0,
        'max_intensity': 0.98
    }
    effect_params = {
        'saturation_factor': 0.5,
        'color_shift_factor': 0.12,
        'noise_sigma': 8.0,
        'headlight_boost': 0.9,
        'saturation_boost': 0.5,
        'color_shift_dampen': 0.7
    }

    os.makedirs(output_path, exist_ok=True)  # 确保输出目录存在

    print(f"Applying darkening effect to {len(image_path_list)} images...")
    for image_path in tqdm(image_path_list):
        try:
            # 使用 cv2 加载图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image: {image_path}")
                continue

            h, w = image.shape[:2]
            # 获取蒙版
            mask = darker.get_mask(h, w, **headlight_mask_params)

            # 应用暗化调整，单独传递效果参数
            dark_image = Darker.adjust_image(  # 调用静态方法
                image,
                mask,
                base_ratio,
                saturation_factor=effect_params['saturation_factor'],
                color_shift_factor=effect_params['color_shift_factor'],
                noise_sigma=effect_params['noise_sigma'],
                headlight_boost=effect_params['headlight_boost'],
                saturation_boost=effect_params['saturation_boost'],
                color_shift_dampen=effect_params['color_shift_dampen']
            )

            # 构建输出路径
            file_name = os.path.basename(image_path)
            output_file_path = os.path.join(output_path, file_name)

            # 使用 cv2 保存暗化图像
            cv2.imwrite(output_file_path, dark_image)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")


if __name__ == '__main__':
    # 定义清晰的路径
    raw_nuscenes_data_path = '/mnt/f/datasets/nuscenes'
    # 用于提取图像的中间路径
    extracted_images_path = '/mnt/f/datasets/nuscenes_extracted_sample'
    # 暗化图像的最终路径
    darkened_images_path = '/mnt/f/datasets/nuscenes_lol_darkened'

    print(
        f"Step 1: Extracting images from {raw_nuscenes_data_path} to {extracted_images_path}...")
    os.makedirs(extracted_images_path, exist_ok=True)  # 确保目录存在
    extract_images(raw_nuscenes_data_path, extracted_images_path)
    print("Image extraction complete.")

    print(
        f"Step 2: Darkening images from {extracted_images_path} to {darkened_images_path}...")
    darken_images(extracted_images_path, darkened_images_path)
    print("Image darkening complete.")
