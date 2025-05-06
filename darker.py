import cv2
import os
import numpy as np
from tqdm import tqdm
import random
from typing import Optional, Union, Tuple
from pathlib import Path

# --- Helper Functions ---


def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 10) -> np.ndarray:
    """Adds Gaussian noise to an image."""
    if image is None:
        return None
    img_float = image.astype(np.float32)
    row, col, ch = img_float.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = img_float + gauss
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


def create_headlight_mask(
    height: int,
    width: int,
    center_y_factor: float = 0.9,  # 车灯垂直中心位置（0=顶, 1=底）
    center_x_factor: float = 0.5,  # 车灯水平中心位置（0=左, 1=右）
    beam_width_factor: float = 0.6,  # 底部光束宽度比例
    falloff_sharpness: float = 2.0,  # 光线衰减锐度 (越高衰减越快)
    max_intensity: float = 1.0     # 光束中心最大强度
) -> np.ndarray:
    """Creates a mask simulating headlight illumination."""
    Y, X = np.ogrid[:height, :width]

    # 计算像素到光束中心的归一化距离
    # 垂直距离，底部为0，向上增加
    dist_y = (center_y_factor * height - Y) / (center_y_factor * height)
    dist_y = np.maximum(0, dist_y)  # 只考虑光束前方的区域

    # 水平距离，中心为0，向两侧增加
    center_x = center_x_factor * width
    dist_x = np.abs(X - center_x) / (beam_width_factor * width / 2)  # 归一化到光束半宽

    # 组合距离，考虑光束形状 (例如，越远越窄)
    # 这里简化处理，使用简单的衰减组合
    # 垂直衰减
    falloff_y = np.exp(-falloff_sharpness * dist_y)
    # 水平衰减 (在光束范围内衰减较慢)
    falloff_x = np.exp(-falloff_sharpness * 0.5 * np.maximum(0,
                       dist_x - (1-dist_y)*0.3)**2)  # 简单的水平衰减，随y距离变化

    # 合并蒙版，限制最大强度
    mask = max_intensity * falloff_y * falloff_x
    mask = np.clip(mask, 0, 1)  # 确保值在0-1

    # (可选) 添加模糊使过渡更平滑
    mask = cv2.GaussianBlur(
        mask, (int(width*0.05) | 1, int(height*0.05) | 1), 0)  # 核大小需奇数

    return mask.astype(np.float32)


class Darker:
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        ratio: float = 0.1,  # 基础暗化比例 (远处最暗程度)
        phase: str = "train"
    ):
        """Initialize the Darker class for image/video darkening.

        Args:
            data_dir: Root directory of the dataset. Required for image processing.
                If None, process_images method cannot be called.
            ratio: Brightness reduction ratio between 0 and 1.
            phase: Processing phase, either "train" or "test".

        Raises:
            ValueError: If ratio is not between 0 and 1 or phase is invalid.
        """
        if not 0 <= ratio <= 1:
            raise ValueError("Ratio must be between 0 and 1")
        if phase not in ["train", "test"]:
            raise ValueError('Phase must be either "train" or "test"')

        self.ratio = ratio
        self.phase = phase
        self.data_dir = Path(data_dir) if data_dir else None
        self.mask_cache = {}  # 缓存蒙版以提高效率

        if self.data_dir:
            base_dir = "our485" if phase == "train" else "eval15"
            self.high_dir = self.data_dir / base_dir / "high"
            self.low_dir = self.data_dir / base_dir / "low"
            os.makedirs(self.low_dir, exist_ok=True)

            if not self.high_dir.exists():
                raise FileNotFoundError(
                    f"High-quality images directory not found: {self.high_dir}"
                )

    def get_mask(self, height: int, width: int, **mask_params) -> np.ndarray:
        """获取或创建并缓存光照蒙版"""
        key = (height, width)  # 使用尺寸作为缓存键
        if key not in self.mask_cache:
            # 从mask_params中提取参数，如果未提供则使用默认值
            mask_config = {
                'center_y_factor': mask_params.get('center_y_factor', 0.9),
                'center_x_factor': mask_params.get('center_x_factor', 0.5),
                'beam_width_factor': mask_params.get('beam_width_factor', 0.6),
                'falloff_sharpness': mask_params.get('falloff_sharpness', 2.5),
                # 最大强度略小于1，保留一点环境暗度
                'max_intensity': mask_params.get('max_intensity', 0.95)
            }
            self.mask_cache[key] = create_headlight_mask(
                height, width, **mask_config)
        return self.mask_cache[key]

    @staticmethod
    def adjust_image(
        img: np.ndarray,
        mask: np.ndarray,  # 需要传入蒙版
        base_ratio: float,  # 基础暗化比例
        saturation_factor: float = 0.6,  # 基础饱和度因子
        color_shift_factor: float = 0.1,  # 基础颜色偏移因子
        noise_sigma: float = 5.0,        # 噪声标准差
        headlight_boost: float = 0.8,    # 车灯区域提亮强度 (0-1)
        saturation_boost: float = 0.3,   # 车灯区域饱和度提升比例
        color_shift_dampen: float = 0.5  # 车灯区域颜色偏移减弱比例
    ) -> np.ndarray:
        """Apply darkening effect simulating headlights."""
        if img is None:
            raise ValueError("Input image cannot be None")
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError("Image and mask dimensions must match")

        # 确保 mask 维度匹配图像 (H, W, 1) 以便广播
        mask_3d = mask[:, :, np.newaxis]

        # 1. Adjust Brightness (V) and Saturation (S) in HSV
        seed = random.uniform(0.8, 1.2)  # 细微亮度随机性
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 基础暗化
        v_channel = hsv[:, :, 2]
        base_dark_v = v_channel * base_ratio * seed

        # 根据蒙版提亮车灯区域
        # 提亮量 = (原始亮度 - 基础暗化亮度) * 蒙版 * 提亮因子
        # boost_amount = (v_channel - base_dark_v) * mask * headlight_boost
        # 或者更简单：最终亮度 = 基础暗化 * (1 - mask * boost) + 原始亮度 * mask * boost
        # 或者，混合方式：
        final_v = base_dark_v * (1 - mask_3d[:, :, 0] * headlight_boost) + v_channel * (
            mask_3d[:, :, 0] * headlight_boost * 0.5)  # 混合一点原始亮度
        # 另一种更简单的亮度调整: 亮度系数 = base_ratio + (1 - base_ratio) * mask * boost
        # brightness_multiplier = base_ratio + (1 - base_ratio) * mask_3d[:,:,0] * headlight_boost
        # final_v = v_channel * brightness_multiplier

        hsv[:, :, 2] = np.clip(final_v, 0, 255)

        # 调整饱和度 S，车灯区域饱和度降低较少
        s_channel = hsv[:, :, 1]
        sat_multiplier = saturation_factor + \
            (1 - saturation_factor) * mask_3d[:, :, 0] * saturation_boost
        hsv[:, :, 1] = np.clip(s_channel * sat_multiplier, 0, 255)

        adjusted_hsv = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 2. Simulate Color Shift (Blue shift) in BGR, less shift in bright areas
        bgr_float = adjusted_hsv.astype(np.float32)
        avg_intensity = np.mean(final_v)  # 使用调整后的亮度作为参考
        # 基础偏移量
        base_shift_amount = color_shift_factor * avg_intensity * 0.5
        # 根据蒙版减弱偏移 (蒙版亮的地方偏移小)
        shift_multiplier = np.clip(
            1.0 - mask_3d[:, :, 0] * color_shift_dampen, 0, 1)
        dynamic_shift = base_shift_amount * shift_multiplier

        bgr_float[:, :, 0] += dynamic_shift  # Blue channel
        bgr_float[:, :, 2] -= dynamic_shift * 0.5  # Red channel
        bgr_shifted = np.clip(bgr_float, 0, 255).astype(np.uint8)

        # 3. Add Noise (保持均匀或稍微基于蒙版调整)
        if noise_sigma > 0:
            # sigma_adjusted = noise_sigma * (1 + (1 - mask_3d[:,:,0]) * 0.1) # 暗区噪声稍大
            noisy_img = add_gaussian_noise(
                bgr_shifted, sigma=noise_sigma)  # 简化：均匀噪声
        else:
            noisy_img = bgr_shifted

        return noisy_img

    def process_images(
        self,
        # 添加蒙版和效果控制参数
        mask_params: dict = {},
        saturation_factor: float = 0.6,
        color_shift_factor: float = 0.1,
        noise_sigma: float = 7.0,  # 稍微增加默认噪声
        headlight_boost: float = 0.85,  # 增强车灯亮度
        saturation_boost: float = 0.4,
        color_shift_dampen: float = 0.6
    ) -> None:
        """Batch process images to generate low-light versions.

        Raises:
            RuntimeError: If data_dir was not provided during initialization.
            FileNotFoundError: If no valid images found in high_dir.
        """
        if not self.data_dir:
            raise RuntimeError(
                "Data directory not provided during initialization")

        image_files = [
            f for f in os.listdir(self.high_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        if not image_files:
            raise FileNotFoundError(
                f"No valid images found in {self.high_dir}")

        print(
            f"Processing {len(image_files)} images with headlight simulation...")
        first_image = True
        mask = None

        for image_file in tqdm(image_files):
            high_img_path = self.high_dir / image_file
            high_img = cv2.imread(str(high_img_path))

            if high_img is None:
                print(f"Warning: Could not read image: {high_img_path}")
                continue

            try:
                # 获取或创建蒙版 (只在第一次或尺寸变化时创建)
                h, w = high_img.shape[:2]
                current_mask = self.get_mask(h, w, **mask_params)

                # 调用更新后的 adjust_image
                dark_img = self.adjust_image(
                    high_img,
                    current_mask,
                    self.ratio,  # 现在是 base_ratio
                    saturation_factor,
                    color_shift_factor,
                    noise_sigma,
                    headlight_boost,
                    saturation_boost,
                    color_shift_dampen
                )
                low_img_path = self.low_dir / image_file
                cv2.imwrite(str(low_img_path), dark_img)
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

        print("Image processing completed!")

    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path] = "dark_video.mp4",
        # 添加蒙版和效果控制参数
        mask_params: dict = {},
        saturation_factor: float = 0.6,
        color_shift_factor: float = 0.1,
        noise_sigma: float = 7.0,
        headlight_boost: float = 0.85,
        saturation_boost: float = 0.4,
        color_shift_dampen: float = 0.6
    ) -> None:
        """Generate a low-light version of the input video.

        Args:
            video_path: Path to the source video file.
            output_path: Path for the output darkened video.

        Raises:
            FileNotFoundError: If the input video file doesn't exist.
            RuntimeError: If video processing fails.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 获取蒙版
            mask = self.get_mask(height, width, **mask_params)

            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height))

            print(
                f"Processing video with headlight simulation ({total_frames} frames)...")
            with tqdm(total=total_frames) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 应用效果
                    dark_frame = self.adjust_image(
                        frame,
                        mask,  # 使用缓存的蒙版
                        self.ratio,
                        saturation_factor,
                        color_shift_factor,
                        noise_sigma,
                        headlight_boost,
                        saturation_boost,
                        color_shift_dampen
                    )
                    out.write(dark_frame)
                    pbar.update(1)
        finally:
            cap.release()
            if 'out' in locals() and out.isOpened():  # 确保 out 被定义且打开
                out.release()
            # cv2.destroyAllWindows()

        print(f"Video processing completed! Output saved to: {output_path}")


if __name__ == '__main__':
    # Example usage for image processing
    data_dir = "/mnt/f/datasets/nuscenes_lol"  # 替换为你的数据集路径
    base_ratio = 0.05  # 基础暗度（远处最暗）可以更低

    # 定义蒙版参数 (可以调整)
    headlight_mask_params = {
        'center_y_factor': 0.95,  # 光源更靠近底部
        'beam_width_factor': 0.7,  # 光束稍宽
        'falloff_sharpness': 3.0,  # 衰减更快
        'max_intensity': 0.98   # 中心最大亮度
    }

    # 定义效果参数
    effect_params = {
        'saturation_factor': 0.5,
        'color_shift_factor': 0.12,
        'noise_sigma': 8.0,
        'headlight_boost': 0.9,  # 更强的提亮
        'saturation_boost': 0.5,
        'color_shift_dampen': 0.7  # 亮区颜色偏移抑制更强
    }

    # Process both train and test phase image datasets
    for phase in ["train", "test"]:
        try:
            print(f"\nProcessing {phase} phase...")
            darker = Darker(data_dir, ratio=base_ratio, phase=phase)
            darker.process_images(
                mask_params=headlight_mask_params, **effect_params)
        except FileNotFoundError as e:
            print(f"Skipping {phase} phase: {str(e)}")
        except Exception as e:
            print(f"Error processing {phase} phase: {str(e)}")

    # Example usage for video processing
    # video_path = "examples/input.mp4" # 确保此文件存在
    # output_video_path = "examples/output_headlight_sim.mp4"
    # print(f"\nProcessing video {video_path}...")
    # try:
    #     darker_video = Darker(ratio=base_ratio) # data_dir 可以为 None
    #     darker_video.process_video(
    #         video_path,
    #         output_video_path,
    #         mask_params=headlight_mask_params,
    #         **effect_params
    #     )
    # except FileNotFoundError as e:
    #     print(f"Video processing skipped: {str(e)}")
    # except Exception as e:
    #     print(f"Error processing video: {str(e)}")
