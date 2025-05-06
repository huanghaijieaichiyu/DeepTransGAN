# 讲nuscenes的图片转换为视频

import os
import cv2
import glob
from tqdm import tqdm


def video_writer(image_path, video_path):
    # 支持的图片扩展名列表
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = []

    # 查找所有支持格式的图片文件
    print(f"正在目录 {image_path} 中查找支持的图片格式: {supported_extensions}...")
    for ext in supported_extensions:
        # 使用 os.path.join 确保路径分隔符正确
        # 使用 f-string 格式化通配符
        pattern = os.path.join(image_path, f'*{ext}')
        found_files = glob.glob(pattern)
        if found_files:
            print(f"找到 {len(found_files)} 个 '{ext}' 文件。")
            image_paths.extend(found_files)

    if not image_paths:  # 检查是否找到了任何图片
        print(f"警告：在目录 {image_path} 中未找到任何支持的图片文件 ({supported_extensions})。")
        return

    # 按文件名排序，确保帧顺序正确
    # 注意：这依赖于文件名的自然排序 (例如 'frame_001', 'frame_002', ...)
    image_paths.sort()
    print(f"共找到 {len(image_paths)} 个图片文件，已排序。")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    try:  # 添加错误处理
        # 读取第一帧以获取尺寸
        first_image_path = image_paths[0]
        image = cv2.imread(first_image_path)
        if image is None:
            print(f"错误：无法读取第一帧图像 {first_image_path}。")
            return

        # 获取图像尺寸
        height, width, _ = image.shape  # 获取高宽

        # 创建视频写入器
        video = cv2.VideoWriter(video_path, fourcc, 30,
                                (width, height))  # 使用获取的宽高
        if not video.isOpened():
            print(f"错误：无法在 {video_path} 创建视频写入器。")
            return

        # 写入视频
        print(f"开始将 {len(image_paths)} 帧图像写入视频 {video_path}...")
        for image_path in tqdm(image_paths, desc="合成视频"):
            img = cv2.imread(image_path)  # 重命名变量避免覆盖
            if img is None:
                print(f"警告：跳过无法读取的图像 {image_path}。")
                continue
            # 确保帧尺寸一致，如果不一致则调整大小 (可选，但建议)
            if img.shape[0] != height or img.shape[1] != width:
                print(
                    f"警告: 图像 {os.path.basename(image_path)} ({img.shape[1]}x{img.shape[0]}) 与第一帧 ({width}x{height}) 尺寸不匹配，将调整大小。")
                img = cv2.resize(img, (width, height))
            video.write(img)  # 写入读取的（可能调整过大小的）图像

        video.release()
        print(f"视频已成功保存到 {video_path}")
    except Exception as e:
        print(f"写入视频时发生错误: {e}")
        # 如果视频写入器已打开，尝试释放
        if 'video' in locals() and video.isOpened():
            video.release()


if __name__ == '__main__':
    image_path = '/mnt/f/datasets/test_video'
    video_path = '/mnt/f/datasets/test_video.mp4'
    print(f'image_path: {image_path}')
    print(f'video_path: {video_path}')
    video_writer(image_path, video_path)
