# diffusion_predictor.py
import argparse
import os
import cv2  # 用于视频处理
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset  # 导入 Dataset 用于类型提示
from torchvision import transforms
from tqdm import tqdm
import time
import traceback  # 导入 traceback 用于打印错误

# Diffusers 相关导入
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput

from datasets.data_set import LowLightDataset  # 重命名导入以区分
from utils.video_writer import video_writer  # 导入视频合成函数


def save_output_path(base_path, model_type='predict'):
    """创建一个带有时间戳的唯一输出目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # 使用进程ID确保并发运行时路径唯一
    path = os.path.join(
        base_path, f"diffusion_{model_type}_{timestamp}_{os.getpid()}")
    os.makedirs(path, exist_ok=True)
    # 在主输出目录下创建 'predictions' 子目录
    os.makedirs(os.path.join(path, 'predictions'), exist_ok=True)
    return path


class BaseDiffusionPredictor:
    """扩散模型预测器的基类"""

    def __init__(self, args):
        self.args = args
        # 设置运行设备
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        print(f"使用的设备: {self.device}")

        # 加载预训练的 UNet 模型
        # 假设模型使用 save_pretrained 保存
        try:
            # Linter 对 from_pretrained().to() 的类型推断可能有问题，但用法是正确的
            self.model: UNet2DModel = UNet2DModel.from_pretrained(
                args.model_path).to(self.device)  # type: ignore
            self.model.eval()  # 设置为评估模式
            print(f"成功从 {args.model_path} 加载 UNet 模型")
        except Exception as e:
            print(f"加载 UNet 模型时出错: {e}")
            raise  # 无法继续，抛出异常

        # 初始化噪声调度器
        # 使用与训练时兼容的设置 (例如 beta_schedule)
        self.scheduler: DDPMScheduler = DDPMScheduler(  # 添加类型提示
            num_train_timesteps=1000,  # 如果训练时不同，需要修改
            beta_schedule="squaredcos_cap_v2"  # 确保与训练设置匹配
        )
        # 设置推断步数
        self.scheduler.set_timesteps(args.num_inference_steps)
        print(f"初始化 DDPMScheduler，推断步数: {args.num_inference_steps}")

        # 定义图像转换：调整大小并将像素值归一化到 [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),  # 转换到 [0, 1]
            transforms.Normalize([0.5], [0.5]),  # 归一化到 [-1, 1]
        ])
        # 定义将 Tensor 转换回 PIL Image 的操作 (用于保存)
        self.to_pil = transforms.ToPILImage()

        # 设置并创建输出路径
        self.output_path = save_output_path(
            args.output_dir, model_type='predict')
        print(f"预测结果将保存到: {self.output_path}")

    def _sample(self, low_light_condition_batch: torch.Tensor) -> torch.Tensor:  # 添加类型提示
        """
        执行条件扩散采样（逆过程）的核心函数。
        Args:
            low_light_condition_batch (torch.Tensor): 低光照条件图像批次，形状 [B, C, H, W]，值域 [-1, 1]
        Returns:
            torch.Tensor: 生成的增强图像批次，形状 [B, C, H, W]，值域 [0, 1]
        """
        batch_size = low_light_condition_batch.shape[0]

        # 1. 从标准正态分布开始生成初始噪声 (latents)
        # 输出通道数通常为 3 (RGB)
        latents = torch.randn(
            (batch_size, 3, self.args.resolution, self.args.resolution),
            device=self.device,
            dtype=self.model.dtype  # 确保数据类型匹配模型
        )

        # 根据调度器的要求缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma

        # 确保条件图像在正确的设备和数据类型上
        low_light_condition_batch = low_light_condition_batch.to(
            self.device, dtype=self.model.dtype)

        # 2. 迭代执行去噪步骤
        # 从 T 迭代到 0
        for t in tqdm(self.scheduler.timesteps, desc="采样步骤", leave=False, disable=self.args.disable_tqdm):
            # 关闭梯度计算以节省内存和计算
            with torch.no_grad():
                # 准备模型输入: 将 latents 和 条件图像 在通道维度拼接
                # UNet 期望的输入形状: [B, in_channels, H, W]
                # in_channels = latent_channels (3) + condition_channels (3) = 6
                # Linter 可能无法完美推断 torch.cat 的类型，但用法是正确的
                model_input = torch.cat(
                    [latents, low_light_condition_batch], dim=1)  # type: ignore

                # 时间步需要为 LongTensor，并扩展到批次大小
                timestep_tensor = torch.tensor(
                    [t] * batch_size, device=self.device).long()

                # 预测噪声残差 (unet 的输出)
                noise_pred = self.model(model_input, timestep_tensor).sample

                # 使用调度器计算上一步的噪声样本 (x_{t-1})
                # scheduler.step 需要 int 类型的时间步
                current_timestep = int(
                    t.item() if isinstance(t, torch.Tensor) else t)
                # Linter 可能无法推断 step 输出的确切类型，但用法是正确的
                step_output = self.scheduler.step(
                    noise_pred, current_timestep, latents)  # type: ignore

                # 检查调度器输出类型并更新 latents (DDPMSchedulerOutput 是 dataclass)
                if isinstance(step_output, DDPMSchedulerOutput):
                    latents = step_output.prev_sample
                elif isinstance(step_output, torch.Tensor):  # 处理直接返回 Tensor 的情况
                    latents = step_output
                else:  # 处理其他可能的返回类型，例如元组 (旧版本?)
                    print(
                        f"警告: 调度器步骤返回了非预期的类型: {type(step_output)}，尝试获取第一个元素。")
                    try:
                        latents = step_output[0]  # 假设样本是第一个元素
                    except (TypeError, IndexError):
                        print("错误：无法从调度器输出中提取样本。")
                        # 可以选择在此处中断或使用之前的 latents
                        pass  # 或者 raise Erorr("...")

        # 3. 后处理: 将生成的图像从 [-1, 1] 缩放回 [0, 1] 并裁剪
        # latents 现在是去噪后的图像
        # Linter 对 / 和 + 的类型推断可能有问题，但对 Tensor 操作是正确的
        enhanced_images_0_1 = (latents / 2 + 0.5).clamp(0, 1)  # type: ignore
        return enhanced_images_0_1

    def predict_images(self):
        """预测图像数据集的方法（子类实现）"""
        raise NotImplementedError("子类必须实现 predict_images 方法")

    def predict_video(self):
        """预测单个视频的方法（子类实现）"""
        raise NotImplementedError("子类必须实现 predict_video 方法")


class ImageDiffusionPredictor(BaseDiffusionPredictor):
    """用于图像数据集预测的扩散模型预测器"""

    def __init__(self, args):
        super().__init__(args)
        # 特定于图像预测的数据集和数据加载器设置
        try:
            # 使用评估专用的转换 (不含随机增强)
            eval_transform = transforms.Compose([
                transforms.Resize((args.resolution, args.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # 归一化到 [-1, 1]
            ])
            # 加载数据集 - 确保指定了正确的 'phase' (例如 'test')
            # 假设 LowLightDataset 接受 image_dir, transform, phase 参数
            # Linter 警告是由于占位符类的存在，实际运行时类型应匹配
            self.dataset: Dataset = LowLightDataset(  # type: ignore
                image_dir=args.data_dir, transform=eval_transform, phase="test"  # 或者适合预测的 phase
            )
            if len(self.dataset) == 0:
                raise ValueError(f"在指定 phase 的 {args.data_dir} 中未找到图像。")

            # 创建数据加载器
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,  # 预测时不需要打乱
                num_workers=args.num_workers,
                pin_memory=True  # 如果 GPU 可用，加速数据传输
            )
            print(f"成功加载图像数据集，包含 {len(self.dataset)} 张图像。")
        except Exception as e:
            print(f"从 {args.data_dir} 加载图像数据集时出错: {e}")
            raise

    def predict_images(self):
        """执行图像数据集的预测"""
        print("开始图像预测...")
        img_index = 0  # 用于生成输出文件名
        # 定义保存预测结果的目录
        output_dir = os.path.join(self.output_path, 'predictions')

        # 使用 tqdm 显示进度条
        pbar = tqdm(self.dataloader, desc="图像预测进度")
        for batch in pbar:
            # 从数据加载器获取批次数据
            # 假设数据加载器产生 (low_light, high_light) 或仅 low_light
            # 我们只需要 low_light 作为条件
            if isinstance(batch, (list, tuple)):
                low_light_batch = batch[0]  # 取第一个元素作为条件图像
            else:
                low_light_batch = batch  # 假设批次本身就是条件图像

            # 确保 low_light_batch 是 Tensor
            if not isinstance(low_light_batch, torch.Tensor):
                print(f"警告：数据加载器返回了非 Tensor 类型：{type(low_light_batch)}，跳过此批次。")
                continue

            # 执行扩散采样过程
            enhanced_batch = self._sample(
                low_light_batch)  # 返回值域为 [0, 1] 的 Tensor

            # 逐个保存批次中的图像
            for i in range(enhanced_batch.shape[0]):
                enhanced_image_tensor = enhanced_batch[i].cpu()  # 移动到 CPU
                # 将 Tensor ([0, 1]) 转换为 PIL Image
                enhanced_pil = self.to_pil(enhanced_image_tensor)

                # 构建输出文件名
                # 注意：如果需要使用原始文件名，需要修改数据集类以返回文件名
                # 这里使用简单的索引作为文件名
                img_save_path = os.path.join(
                    output_dir, f"enhanced_{img_index:05d}.png")
                enhanced_pil.save(img_save_path)
                img_index += 1

            # 更新进度条后缀信息
            pbar.set_postfix({"已保存": f"{img_index} 张图像"})

        print(f"图像预测完成。共保存 {img_index} 张图像到 {output_dir}")

    def predict_video(self):
        """图像预测器不支持视频预测"""
        print("错误: ImageDiffusionPredictor 不支持 predict_video 方法。请使用 'video' 模式。")


class VideoDiffusionPredictor(BaseDiffusionPredictor):
    """用于单个视频预测的扩散模型预测器"""

    def __init__(self, args):
        super().__init__(args)
        # 检查视频文件是否存在
        if not os.path.isfile(args.video_path):
            raise FileNotFoundError(f"视频文件未找到: {args.video_path}")
        self.video_path = args.video_path

        # 定义用于处理单帧视频的转换 (输入是 numpy 数组)
        self.frame_transform = transforms.Compose([
            # 如果输入是 numpy 数组，需要先转为 PIL Image
            transforms.ToPILImage(),
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),  # 转为 [0, 1]
            transforms.Normalize([0.5], [0.5]),  # 归一化到 [-1, 1]
        ])

    def predict_images(self):
        """视频预测器不支持图像数据集预测"""
        print("错误: VideoDiffusionPredictor 不支持 predict_images 方法。请使用 'image' 模式。")

    def predict_video(self):
        """执行单个视频的预测"""
        print(f"开始视频预测: {self.video_path}")
        # 定义保存单帧图像的目录
        frames_output_dir = os.path.join(self.output_path, 'predictions')

        # 1. 打开视频文件并获取属性
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"错误: 无法打开视频文件 {self.video_path}")
                return

            # 获取视频帧率、尺寸和总帧数
            fps = cap.get(cv2.CAP_PROP_FPS)
            # 输出视频的尺寸将是模型的 resolution
            out_width = self.args.resolution
            out_height = self.args.resolution
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                print("警告：无法读取视频总帧数，进度条可能不准确。")
            print(
                f"视频属性: {fps:.2f} FPS, 总帧数: {total_frames if total_frames > 0 else '未知'}")
            print(f"输出分辨率: {out_width}x{out_height}")

        except Exception as e:
            print(f"读取视频属性时出错: {e}")
            return

        # 2. 逐帧处理视频
        frame_count = 0
        # 如果总帧数已知，创建进度条
        pbar = tqdm(total=total_frames if total_frames >
                    0 else None, desc="视频处理进度")
        frames_processed = False  # 标记是否处理了至少一帧

        try:
            while True:
                # 读取一帧
                ret, frame_bgr = cap.read()
                if not ret:
                    break  # 视频结束或读取错误

                # a. 预处理帧: BGR -> RGB -> 应用转换 -> 增加批次维度
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                low_light_tensor = self.frame_transform(
                    frame_rgb).unsqueeze(0)  # [1, 3, H, W]

                # b. 执行扩散采样
                enhanced_tensor_0_1 = self._sample(
                    low_light_tensor)  # 返回 [1, 3, H, W], 值域 [0, 1]

                # c. 后处理帧: Tensor [0, 1] -> Numpy [0, 255] (HWC BGR)
                enhanced_frame_np = enhanced_tensor_0_1.squeeze(
                    0).cpu().numpy()  # 移除批次维度, 移到 CPU
                enhanced_frame_np = np.transpose(
                    enhanced_frame_np, (1, 2, 0))  # C, H, W -> H, W, C
                enhanced_frame_uint8 = (
                    enhanced_frame_np * 255).clip(0, 255).astype(np.uint8)  # 转为 uint8
                enhanced_frame_bgr = cv2.cvtColor(
                    enhanced_frame_uint8, cv2.COLOR_RGB2BGR)  # 转回 BGR 以便 OpenCV 处理

                # d. 保存处理后的帧 (每一帧都保存)
                img_save_path = os.path.join(
                    frames_output_dir, f"frame_{frame_count:06d}.png")
                # 确保帧尺寸符合预期 (理论上应该匹配，因为 transform 已处理)
                if enhanced_frame_bgr.shape[1] != out_width or enhanced_frame_bgr.shape[0] != out_height:
                    enhanced_frame_bgr = cv2.resize(
                        enhanced_frame_bgr, (out_width, out_height))
                cv2.imwrite(img_save_path, enhanced_frame_bgr)

                # 可选：实时显示帧 (可能降低处理速度)
                if self.args.display_video:
                    # 调整原始帧大小以便并排显示
                    display_orig = cv2.resize(
                        frame_bgr, (out_width, out_height))
                    # 水平拼接原始帧和增强帧
                    combined_display = cv2.hconcat(
                        [display_orig, enhanced_frame_bgr])
                    cv2.imshow("原始 vs 增强 (Diffusion)", combined_display)
                    # 按 'q' 键退出显示
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("接收到退出信号。")
                        break

                frame_count += 1
                frames_processed = True  # 标记已处理帧
                if total_frames > 0:  # 仅当总帧数已知时更新进度条
                    pbar.update(1)
                else:  # 如果总帧数未知，仅更新描述
                    pbar.set_description(f"视频处理进度 (帧 {frame_count})")

        except Exception as e:
            print(f"\n在处理第 {frame_count} 帧时发生错误: {e}")
            traceback.print_exc()  # 打印详细错误信息
        finally:
            # 3. 清理资源
            pbar.close()
            cap.release()  # 释放视频捕获对象
            cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
            print(f"视频帧处理完成。共处理并保存 {frame_count} 帧到 {frames_output_dir}。")

        # 4. 调用 video_writer 合成视频 (仅当成功处理了帧)
        if frames_processed:
            video_output_path = os.path.join(
                self.output_path, "enhanced_video.mp4")
            print(f"\n开始使用 {frames_output_dir} 中的帧合成视频...")
            try:
                # 调用导入的函数
                video_writer(frames_output_dir, video_output_path)
            except Exception as e:
                print(f"调用 video_writer 合成视频时发生错误: {e}")
        else:
            print("未处理任何帧，跳过视频合成。")


def parse_predict_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="扩散模型预测脚本")

    # 通用参数
    parser.add_argument("--mode", type=str, required=True, choices=['image', 'video'],
                        help="预测模式: 'image' 处理图像数据集, 'video' 处理单个视频文件。")
    parser.add_argument("--model_path", type=str, required=True,
                        help="指向包含 'diffusion_pytorch_model.bin' 和 'config.json' 的已训练 UNet 模型目录的路径。")
    parser.add_argument("--output_dir", type=str, default="diffusion_predictions",
                        help="保存预测结果的根目录。")
    parser.add_argument("--device", type=str, default="cuda", choices=['cuda', 'cpu'],
                        help="运行预测的设备 ('cuda' 或 'cpu')。")
    parser.add_argument("--resolution", type=int, default=256,
                        help="模型训练时使用的图像分辨率。")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="DDPM/DDIM 采样步数。")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="用于图像数据加载器的工作线程数 (仅 image 模式)。")
    parser.add_argument("--disable_tqdm", action="store_true",
                        help="禁用采样过程中的 tqdm 进度条。")

    # 图像模式特定参数
    parser.add_argument("--data_dir", type=str, default="../datasets/kitti_LOL",
                        help="图像数据集的根目录路径 (仅 image 模式)。")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="图像预测时的批次大小 (仅 image 模式)。")

    # 视频模式特定参数
    parser.add_argument("--video_path", type=str,
                        help="输入视频文件的路径 (仅 video 模式)。")
    parser.add_argument("--display_video", action="store_true",
                        help="在处理过程中实时显示原始帧和增强帧 (可能较慢, 仅 video 模式)。")

    args = parser.parse_args()

    # 参数校验
    if args.mode == 'video' and not args.video_path:
        parser.error("--video_path 参数在 video 模式下是必需的。")
    if args.mode == 'image' and not args.data_dir:
        parser.error("--data_dir 参数在 image 模式下是必需的。")
    # 检查 model_path 是否存在且是目录
    if not os.path.isdir(args.model_path):
        parser.error(
            f"--model_path '{args.model_path}' 不是一个有效的目录。请提供包含模型文件的目录路径。")
    # 检查模型权重文件 (.bin 或 .safetensors) 和配置文件是否存在
    model_bin_path = os.path.join(
        args.model_path, 'diffusion_pytorch_model.bin')
    model_safetensors_path = os.path.join(
        args.model_path, 'diffusion_pytorch_model.safetensors')
    config_path = os.path.join(args.model_path, 'config.json')

    if not os.path.exists(model_bin_path) and not os.path.exists(model_safetensors_path):
        parser.error(
            f"在 '{args.model_path}' 目录下未找到 'diffusion_pytorch_model.bin' 或 'diffusion_pytorch_model.safetensors'。")
    if not os.path.exists(config_path):
        parser.error(f"在 '{args.model_path}' 目录下未找到 'config.json'。")

    return args


if __name__ == "__main__":
    args = parse_predict_args()

    # 根据模式选择并运行相应的预测器
    if args.mode == 'image':
        # 检查 LowLightDataset 是否有效加载且不是占位符
        is_placeholder = 'LowLightDataset' in globals() and \
                         issubclass(globals()['LowLightDataset'], Dataset) and \
                         len(globals()['LowLightDataset']()) == 0  # 检查占位符的特征

        if 'LowLightDataset' in globals() and not is_placeholder:
            try:
                predictor = ImageDiffusionPredictor(args)
                predictor.predict_images()
            except Exception as e:
                print(f"执行图像预测时发生错误: {e}")
                traceback.print_exc()
        else:
            print("错误：LowLightDataset 未正确加载或为占位符，无法执行图像预测。请检查 'datasets/data_set.py' 文件。")

    elif args.mode == 'video':
        try:
            predictor = VideoDiffusionPredictor(args)
            predictor.predict_video()
        except Exception as e:
            print(f"执行视频预测时发生错误: {e}")
            traceback.print_exc()
    else:
        # 这部分理论上不会执行，因为 argparse 的 choices 参数会限制模式
        print(f"错误: 不支持的模式 '{args.mode}'")
