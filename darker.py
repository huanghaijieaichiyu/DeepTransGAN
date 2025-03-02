import cv2
import os
import numpy as np
from tqdm import tqdm
import random


class Darker:
    def __init__(self, data_dir=None, ratio=0.5, phase="train"):
        """
        Args:
            data_dir (str或None): 数据集根目录，仅对处理图像有效。如果为 None，则无法调用process_images方法。
            ratio (float): 亮度降低的比例，0到1之间。
            phase (str): 阶段，可选值为 "train" 或 "test"。
        """
        self.ratio = ratio
        self.phase = phase
        self.data_dir = data_dir
        if data_dir:  # 针对图像处理
            self.high_dir = os.path.join(
                data_dir, "our485", "high") if phase == "train" else os.path.join(data_dir, "eval15", "high")
            self.low_dir = os.path.join(
                data_dir, "our485", "low") if phase == "train" else os.path.join(data_dir, "eval15", "low")
            os.makedirs(self.low_dir, exist_ok=True)

    @staticmethod
    def adjust_image(img, ratio):
        """
        对传入的图像执行暗化处理
        """
        seed = random.uniform(0, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * ratio, 0, 255)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(
            np.float32) * seed * ratio, 0, 255)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0].astype(
            np.float32) * 0.5 * seed * ratio, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def process_images(self):
        """
        批量生成低光照图像
        """
        image_files = [f for f in os.listdir(self.high_dir) if f.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.bmp'))]
        print("开始处理图像数据集...")
        for image_file in tqdm(image_files):
            high_img_path = os.path.join(self.high_dir, image_file)
            high_img = cv2.imread(high_img_path)
            if high_img is None:
                print(f"无法读取图像: {high_img_path}")
                continue
            dark_img = self.adjust_image(high_img, self.ratio)
            low_img_path = os.path.join(self.low_dir, image_file)
            cv2.imwrite(low_img_path, dark_img)
        print("图像处理完成！请手动检查数据集。")

    def process_video(self, video_path, output_path="dark_video.mp4"):
        """
        生成低光照视频
        Args:
            video_path (str): 源视频路径
            output_path (str): 输出视频路径
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法打开视频文件！")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("开始处理视频...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            dark_frame = self.adjust_image(frame, self.ratio)
            out.write(dark_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("视频处理完成！")


if __name__ == '__main__':
    # 处理图像示例
    data_dir = "../datasets/kitti_LOL"  # 替换成你的数据集路径
    ratio = 0.05  # 亮度降低的比例

    # 处理 train 和 test 两个阶段的图像数据集
    darker_train = Darker(data_dir, ratio=ratio, phase="train")
    darker_train.process_images()

    darker_test = Darker(data_dir, ratio=ratio, phase="test")
    darker_test.process_images()

    # 若需处理视频，传入视频路径
    # video_path = "examples/upc_dark.mp4"
    # darker_video = Darker(ratio=ratio)
    # darker_video.process_video(video_path)
