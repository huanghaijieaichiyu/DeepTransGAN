import argparse
import os
import math
import logging
from pathlib import Path
import random
from PIL import Image
import time  # 引入 time 模块

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader

import diffusers
import xformers  # 导入 xformers
# Import ConfigMixin for type hinting
from diffusers.configuration_utils import ConfigMixin

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# 引入 DDPMSchedulerOutput 用于类型提示
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from torchvision import transforms
from torcheval.metrics.functional import peak_signal_noise_ratio

from utils.misic import ssim, save_path
from datasets.data_set import LowLightDataset  # 假设数据集类可用
import lpips

# 检查 diffusers 版本
check_min_version("0.10.0")  # 示例版本，根据需要调整

logger = get_logger(__name__, log_level="INFO")


# === 定义条件编码器 ===
class ConditioningEncoder(nn.Module):
    def __init__(self, in_channels=3, base_c=64, levels=3, out_dim=256):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        # 根据 levels 动态调整 base_c，确保最后一层通道数合理
        # 例如，如果 levels=3, base_c=64, 则通道为 64, 128, 256
        # 如果 levels=4, base_c=32, 则通道为 32, 64, 128, 256

        actual_base_c = base_c
        if levels > 0:
            # 目标是让最后一层卷积前的通道数接近 out_dim 或一个较大的固定值
            # 例如，如果希望最后一层卷积前的通道数为 256，而 levels=4
            # 256 / (2**(4-1)) = 256 / 8 = 32. So actual_base_c = 32.
            # 确保 actual_base_c 至少为 16 或 32
            # target_last_conv_channels = 256 # 可调整
            # if levels > 0:
            #     actual_base_c = max(16, target_last_conv_channels // (2**(levels -1)))

            for i in range(levels):
                output_channels_level = actual_base_c * (2**i)
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(current_channels, output_channels_level,
                                  kernel_size=3, padding=1),
                        nn.SiLU(),
                        nn.Conv2d(output_channels_level,
                                  output_channels_level, kernel_size=3, padding=1),
                        nn.SiLU(),
                        nn.Conv2d(output_channels_level, output_channels_level,
                                  kernel_size=3, padding=1, stride=2),  # 下采样
                        nn.SiLU()
                    )
                )
                current_channels = output_channels_level

        # Final projection to out_dim (cross_attention_dim)
        # current_channels 是最后一个下采样模块的输出通道数
        # Kernel 1x1 to change dim
        self.final_conv = nn.Conv2d(current_channels, out_dim, kernel_size=1)
        logger.info(
            f"ConditioningEncoder initialized: in_channels={in_channels}, base_c={actual_base_c}, levels={levels}, final_conv_in={current_channels}, out_dim={out_dim}")

    def forward(self, condition_image):
        x = condition_image
        for layer in self.layers:
            x = layer(x)
        # x is now B, C_last, H_final, W_final
        x = self.final_conv(x)  # B, out_dim, H_final, W_final

        # Reshape for cross-attention: B, H_final * W_final, out_dim
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height * width).permute(0, 2, 1)
        return x
# =======================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a conditional diffusion model training script for low-light enhancement.")
    parser.add_argument(
        "--data", type=str, default="../datasets/kitti_LOL", help="数据集根目录"
    )
    parser.add_argument(
        "--output_dir", type=str, default="run_diffusion_cross_attn", help="所有输出 (模型, 日志等) 的根目录"
    )
    parser.add_argument("--overwrite_output_dir",
                        action="store_true", help="是否覆盖输出目录")
    parser.add_argument("--cache_dir", type=str, default=None, help="缓存目录")
    parser.add_argument("--seed", type=int,
                        default=random.randint(0, 1000000), help="随机种子")
    parser.add_argument(
        "--resolution", type=int, default=256, help="输入图像分辨率"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="训练和评估的批次大小"
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps", type=int, default=None, help="如果设置，将覆盖 num_train_epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="是否启用梯度检查点"
    )
    parser.add_argument(
        # 调整了默认值
        "--learning_rate", type=float, default=2e-4, help="优化器初始学习率 (调整,可能需要更低)"
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help="学习率调度器类型"
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="学习率预热步数"
    )
    parser.add_argument(
        "--b1", type=float, default=0.9, help="AdamW 优化器的 beta1 参数"
    )
    parser.add_argument(
        "--b2", type=float, default=0.999, help="AdamW 优化器的 beta2 参数"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="AdamW 优化器的权重衰减"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-07, help="AdamW 优化器的 epsilon 参数"
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="最大梯度范数（用于梯度裁剪）"
    )
    parser.add_argument(
        "--mixed_precision", type=str, default='fp16', choices=["no", "fp16", "bf16"],
        help="是否使用混合精度训练。选择 'fp16' 或 'bf16' (需要 PyTorch >= 1.10)，或 'no' 关闭。"
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="是否启用 xformers 内存高效注意力"
    )
    parser.add_argument(
        "--checkpointing_steps", type=int, default=5000, help="每 N 步保存一次检查点"
    )
    parser.add_argument(
        "--checkpoints_total_limit", type=int, default=5, help="限制检查点总数。删除旧的检查点。"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="从哪个检查点恢复训练 ('latest' 或特定路径)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Dataloader 使用的工作线程数"
    )

    # === UNet 参数 ===
    parser.add_argument(
        "--unet_in_channels", type=int, default=3, help="UNet 输入通道数 (通常为3，对应噪声图像)"
    )
    parser.add_argument(
        "--unet_out_channels", type=int, default=3, help="UNet 输出通道数 (通常为3，对应预测噪声)"
    )
    parser.add_argument(
        "--unet_layers_per_block", type=int, default=2, help="UNet 中每个块的 ResNet 层数"
    )
    parser.add_argument(
        "--unet_block_channels", nargs='+', type=int, default=[64, 64, 128, 128, 256, 256],
        help="UNet 各层级的通道数 (根据 UNet2DConditionModel 的典型配置调整)"  # 调整了默认值
    )
    # 为 UNet2DConditionModel 更新默认的 block types 以包含 CrossAttn
    parser.add_argument(
        "--unet_down_block_types", nargs='+', type=str,
        default=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                 "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],  # 示例调整
        help="UNet 下采样块类型 (使用 CrossAttn 变体)"
    )
    parser.add_argument(
        "--unet_up_block_types", nargs='+', type=str,
        default=["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                 "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],  # 示例调整
        help="UNet 上采样块类型 (使用 CrossAttn 变体)"
    )
    parser.add_argument(
        "--unet_mid_block_type", type=str, default="UNetMidBlock2DCrossAttn",  # 确保中间块也支持交叉注意力
        help="UNet 中间块类型"
    )
    parser.add_argument(
        "--cross_attention_dim", type=int, default=128, help="交叉注意力维度 (ConditioningEncoder的输出维度)"
    )

    # === Conditioning Encoder 参数 ===
    parser.add_argument(
        "--cond_enc_base_c", type=int, default=64, help="ConditioningEncoder 基础通道数"
    )
    parser.add_argument(
        "--cond_enc_levels", type=int, default=3, help="ConditioningEncoder 层级数 (下采样次数)"
    )
    # cross_attention_dim 已经定义在 UNet 参数部分，它也是 cond_enc 的输出维度

    # === 推断和验证参数 ===
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="采样/推断步数"
    )
    parser.add_argument(
        "--num_validation_images", type=int, default=4, help="验证时生成的图像数量"
    )
    parser.add_argument(
        "--validation_epochs", type=int, default=5, help="每 N 个 epoch 运行一次验证 (将基于步数触发)"
    )
    # === 添加 LPIPS 损失权重 ===
    parser.add_argument(
        "--lambda_lpips", type=float, default=0.5, help="LPIPS 损失的权重"
    )
    # =========================
    # === 添加 Accelerate 日志报告目标 ===
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ===========================

    # === 添加轻量化 UNet 参数 ===
    parser.add_argument(
        "--lightweight_unet", action="store_true", help="使用轻量化的 UNet 配置进行快速测试"
    )
    # ===========================

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and args.mixed_precision == "fp16":
        logger.warning(
            "FP16 is not recommended for multi-GPU training. Setting mixed_precision to 'no'.")
        args.mixed_precision = "no"

    # === 轻量化配置覆盖 ===
    if args.lightweight_unet:
        args.unet_layers_per_block = 1
        args.unet_block_channels = [32, 64, 128]  # 进一步减少通道数
        args.unet_down_block_types = [
            "DownBlock2D",  # 第一个下采样块使用普通版本
            "CrossAttnDownBlock2D",  # 中间使用交叉注意力
            "DownBlock2D",
        ]
        args.unet_up_block_types = [
            "UpBlock2D",
            "CrossAttnUpBlock2D",  # 中间使用交叉注意力
            "UpBlock2D",  # 最后一个上采样块使用普通版本
        ]
        args.cond_enc_base_c = 16  # 进一步减小条件编码器基础通道
        args.cond_enc_levels = 2  # 保持层级，但通道数减少
        args.cross_attention_dim = 64  # 大幅减少交叉注意力维度
    # ======================

    return args


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if args.lightweight_unet:
        logger.info("使用轻量化 UNet 和 ConditioningEncoder 配置...")

    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"设置随机种子为: {args.seed}")

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.overwrite_output_dir:
                logger.info(f"Overwriting output directory {args.output_dir}")
        # save_path(args.output_dir) # os.makedirs 已经处理

    # 初始化条件编码器
    logger.info("Initializing ConditioningEncoder model...")
    condition_encoder = ConditioningEncoder(
        in_channels=3,  # 低光照图像是3通道
        base_c=args.cond_enc_base_c,
        levels=args.cond_enc_levels,
        out_dim=args.cross_attention_dim
    )

    # 初始化 UNet2DConditionModel
    logger.info("Initializing UNet2DConditionModel...")
    logger.info(f"  In channels: {args.unet_in_channels}")
    logger.info(f"  Out channels: {args.unet_out_channels}")
    logger.info(f"  Layers per block: {args.unet_layers_per_block}")
    logger.info(f"  Block channels: {args.unet_block_channels}")
    logger.info(f"  Down blocks: {args.unet_down_block_types}")
    logger.info(f"  Up blocks: {args.unet_up_block_types}")
    logger.info(f"  Mid block type: {args.unet_mid_block_type}")
    logger.info(f"  Cross Attention Dim: {args.cross_attention_dim}")

    model = UNet2DConditionModel(
        sample_size=args.resolution,
        in_channels=args.unet_in_channels,  # 应该是3 (噪声图像)
        out_channels=args.unet_out_channels,  # 应该是3 (预测噪声)
        layers_per_block=args.unet_layers_per_block,
        block_out_channels=tuple(args.unet_block_channels),  # 需要是 tuple
        down_block_types=tuple(args.unet_down_block_types),  # 需要是 tuple
        up_block_types=tuple(args.unet_up_block_types),   # 需要是 tuple
        mid_block_type=args.unet_mid_block_type,
        cross_attention_dim=args.cross_attention_dim,
        # attention_head_dim 可以尝试设置，或者让模型使用默认值
        # attention_head_dim = args.cross_attention_dim // 4 # 示例
    )

    if args.enable_xformers_memory_efficient_attention:
        try:
            model.enable_xformers_memory_efficient_attention()
            # condition_encoder 不需要这个，因为它主要是卷积
            logger.info("启用 xformers 内存高效注意力 for UNet")
        except Exception as e:
            logger.warning(f"无法启用 xformers: {e}. 继续而不使用 xformers.")

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        # condition_encoder 并没有标准的 enable_gradient_checkpointing 方法
        # 如果需要，需在其 forward 方法内手动使用 torch.utils.checkpoint.checkpoint
        # logger.info("Gradient checkpointing enabled for ConditioningEncoder (manual impl. needed if effective)")

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    # 初始化优化器 (包含 condition_encoder 的参数)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) +
        list(condition_encoder.parameters()),  # <--- 合并参数
        lr=args.learning_rate,
        betas=(args.b1, args.b2),
        weight_decay=args.weight_decay,
        eps=args.epsilon,
    )

    preprocess = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # 输出 [-1, 1]
        ]
    )
    eval_preprocess = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # 输出 [-1, 1]
        ]
    )

    try:
        train_dataset = LowLightDataset(
            image_dir=args.data, transform=preprocess, phase="train")
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        eval_dataset = LowLightDataset(
            image_dir=args.data, transform=eval_preprocess, phase="test")
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
        )
    except Exception as e:
        logger.error(
            f"加载数据集失败，请检查路径 '{args.data}' 和 LowLightDataset 实现: {e}")
        return

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps if args.max_train_steps else len(
            train_dataloader) * args.num_train_epochs * args.gradient_accumulation_steps,
    )

    # 使用 Accelerate 准备 (包含 condition_encoder)
    model, condition_encoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, condition_encoder, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        run_name = Path(
            args.output_dir).name if args.output_dir else "diffusion_cross_attn_run"
        config_to_log = vars(args).copy()
        for key in ["unet_block_channels", "unet_down_block_types", "unet_up_block_types"]:
            if key in config_to_log and isinstance(config_to_log[key], list):
                config_to_log[key] = ",".join(map(str, config_to_log[key]))
        accelerator.init_trackers(run_name, config=config_to_log)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps
    global_step = 0
    first_epoch = 0

    if args.resume:
        if args.resume != "latest":
            path = os.path.basename(args.resume)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume}' does not exist. Starting a new training run."
            )
            args.resume = None
        else:
            checkpoint_path = os.path.join(args.output_dir, path)
            accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
            try:
                accelerator.load_state(checkpoint_path)
                global_step = int(path.split("-")[1])
                accelerator.print(
                    f"Successfully loaded state from {checkpoint_path}. Resuming global step {global_step}.")
            except Exception as e:
                accelerator.print(
                    f"Failed to load state from {checkpoint_path}: {e}. Starting from scratch.")
                global_step = 0
            first_epoch = global_step // num_update_steps_per_epoch

    if accelerator.is_main_process:
        logger.info("Initializing LPIPS model...")
    try:
        lpips_model = lpips.LPIPS(net='alex').to(accelerator.device)
        lpips_model.eval()
        logger.info("LPIPS model initialized successfully.")
    except Exception as e:
        logger.error(
            f"Failed to initialize LPIPS model: {e}. LPIPS loss will not be used.")
        lpips_model = None

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num eval examples = {len(eval_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Starting epoch = {first_epoch + 1}")
    logger.info(f"  Starting global step = {global_step}")

    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        condition_encoder.train()  # <--- 设置 condition_encoder 为训练模式
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # low_light_images 作为条件, clean_images 作为目标
            low_light_images, clean_images = batch

            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config["num_train_timesteps"], (bsz,), device=clean_images.device
            ).int()  # 保持 .int()

            noisy_images = noise_scheduler.add_noise(
                clean_images, noise, timesteps
            )

            # <--- 同时 accumulate 两个模型
            with accelerator.accumulate(model, condition_encoder):
                # 1. 通过 ConditionEncoder 获取条件嵌入
                # low_light_images 已经是 [-1, 1]
                encoder_hidden_states = condition_encoder(low_light_images)
                # encoder_hidden_states: [B, SeqLen, CrossAttnDim]

                # 2. U-Net 预测噪声 (输入为 noisy_images, 条件为 encoder_hidden_states)
                # model_input 现在只是 noisy_images
                noise_pred = model(
                    noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                # 3. 计算 MSE 损失
                loss_mse = F.mse_loss(noise_pred.float(), noise.float())

                # 4. 计算 LPIPS 损失
                loss_lpips = torch.tensor(0.0).to(accelerator.device)
                if lpips_model is not None and args.lambda_lpips > 0:
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
                        timesteps.device)  # 类型和设备已匹配
                    sqrt_alpha_prod = alphas_cumprod[timesteps].sqrt(
                    ).view(-1, 1, 1, 1)
                    sqrt_one_minus_alpha_prod = (
                        1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)

                    pred_x0 = (noisy_images - sqrt_one_minus_alpha_prod *
                               noise_pred) / sqrt_alpha_prod
                    pred_x0_clamp = torch.clamp(pred_x0, -1.0, 1.0)
                    clean_images_clamp = torch.clamp(clean_images, -1.0, 1.0)
                    loss_lpips = lpips_model(
                        pred_x0_clamp.float(), clean_images_clamp.float()).mean()

                # 5. 合并损失
                loss = loss_mse + args.lambda_lpips * loss_lpips

                gathered_loss = accelerator.gather(
                    loss.repeat(args.batch_size))
                if gathered_loss is not None:
                    if isinstance(gathered_loss, torch.Tensor):
                        avg_loss = gathered_loss.mean()
                        train_loss += avg_loss.item() / args.gradient_accumulation_steps
                    else:
                        logger.warning(
                            f"accelerator.gather returned unexpected type: {type(gathered_loss)}")
                        avg_loss = torch.tensor(0.0)
                else:
                    avg_loss = torch.tensor(0.0)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # 合并模型参数以进行统一的梯度裁剪
                    all_parameters = list(
                        model.parameters()) + list(condition_encoder.parameters())
                    # 过滤掉不需要梯度的参数 (可选，但良好实践)
                    parameters_to_clip = [
                        p for p in all_parameters if p.grad is not None]
                    if parameters_to_clip:  # 确保有参数需要裁剪
                        accelerator.clip_grad_norm_(
                            parameters_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    current_lr = lr_scheduler.get_last_lr(
                    )[0] if lr_scheduler else args.learning_rate
                    logs = {
                        "loss": train_loss,
                        "loss_mse": loss_mse.item() / args.gradient_accumulation_steps if accelerator.is_main_process else 0.0,
                        "loss_lpips": loss_lpips.item() / args.gradient_accumulation_steps if accelerator.is_main_process and lpips_model is not None and args.lambda_lpips > 0 else 0.0,
                        "lr": current_lr,
                        "epoch": epoch + 1
                    }
                    postfix_dict = {
                        "loss": f"{train_loss:.4f}",
                        "mse": f"{logs['loss_mse']:.4f}",
                        "lpips": f"{logs['loss_lpips']:.4f}",
                        "lr": f"{current_lr:.6f}",
                        "epoch": epoch + 1
                    }
                    progress_bar.set_postfix(**postfix_dict)
                    accelerator.log(logs, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if args.validation_epochs > 0 and num_update_steps_per_epoch > 0:
                        validation_steps = args.validation_epochs * num_update_steps_per_epoch
                        if global_step % validation_steps == 0 or global_step == args.max_train_steps:
                            logger.info(
                                f"Running validation at step {global_step} (Epoch {epoch+1})...")

                            # 获取原始模型
                            unet_eval = accelerator.unwrap_model(model)
                            condition_encoder_eval = accelerator.unwrap_model(
                                condition_encoder)  # <--- Unwrap cond_enc
                            unet_eval.eval()
                            condition_encoder_eval.eval()  # <--- 设置 cond_enc 为评估模式

                            scheduler_config: dict = noise_scheduler.config
                            sampling_scheduler = DDPMScheduler(
                                **scheduler_config)
                            sampling_scheduler.set_timesteps(
                                args.num_inference_steps)

                            val_psnr_list = []
                            val_ssim_list = []
                            generated_images_pil = []
                            val_progress_bar = tqdm(
                                total=len(eval_dataloader), desc="Validation", leave=False, position=1, disable=not accelerator.is_local_main_process)

                            for val_step, val_batch in enumerate(eval_dataloader):
                                low_light_images_val, clean_images_val = val_batch
                                batch_size_val = low_light_images_val.shape[0]

                                # 获取条件嵌入
                                with torch.no_grad():
                                    encoder_hidden_states_val = condition_encoder_eval(
                                        low_light_images_val)

                                latents = torch.randn_like(
                                    clean_images_val, device=accelerator.device)
                                latents = latents * sampling_scheduler.init_noise_sigma

                                timesteps_to_iterate = sampling_scheduler.timesteps
                                for t_idx, t in enumerate(tqdm(timesteps_to_iterate, leave=False, desc="Sampling", position=2, disable=not accelerator.is_local_main_process)):
                                    with torch.no_grad():
                                        # model_input_val 现在只是 latents
                                        timestep_tensor = torch.tensor(
                                            [t] * batch_size_val, device=accelerator.device).long()
                                        noise_pred_val = unet_eval(
                                            # 输入 latents (即 noisy_sample)
                                            latents,
                                            timestep_tensor,
                                            encoder_hidden_states=encoder_hidden_states_val  # <--- 传递条件嵌入
                                        ).sample

                                        current_timestep_val = int(
                                            t.item() if isinstance(t, torch.Tensor) else t)
                                        step_output = sampling_scheduler.step(
                                            noise_pred_val, current_timestep_val, latents)

                                        if isinstance(step_output, DDPMSchedulerOutput):
                                            latents = step_output.prev_sample
                                        else:
                                            logger.warning(
                                                f"Unexpected type from scheduler step: {type(step_output)}. Using noise_pred as fallback.")
                                            latents = noise_pred_val

                                enhanced_images = latents  # [-1, 1]
                                enhanced_images_0_1 = (
                                    enhanced_images / 2 + 0.5).clamp(0, 1)
                                clean_images_0_1_val = (
                                    clean_images_val / 2 + 0.5).clamp(0, 1)

                                try:
                                    current_psnr = peak_signal_noise_ratio(
                                        enhanced_images_0_1.cpu(), clean_images_0_1_val.cpu()
                                    ).item()
                                    current_ssim = ssim(
                                        enhanced_images_0_1.cpu(), clean_images_0_1_val.cpu()
                                    ).item()
                                    val_psnr_list.append(current_psnr)
                                    val_ssim_list.append(current_ssim)
                                    val_progress_bar.set_postfix(
                                        PSNR=f"{current_psnr:.2f}", SSIM=f"{current_ssim:.4f}")
                                except Exception as e:
                                    logger.error(f"计算指标时出错: {e}")
                                    val_psnr_list.append(float('nan'))
                                    val_ssim_list.append(float('nan'))

                                if len(generated_images_pil) < args.num_validation_images:
                                    num_to_save = min(
                                        batch_size_val, args.num_validation_images - len(generated_images_pil))
                                    for i in range(num_to_save):
                                        enhanced_pil = transforms.ToPILImage()(
                                            enhanced_images_0_1[i].cpu())
                                        # low_light_images_val 已经是 [-1, 1]
                                        low_light_pil = transforms.ToPILImage()(
                                            (low_light_images_val[i].cpu() / 2 + 0.5).clamp(0, 1))
                                        clean_pil = transforms.ToPILImage()(
                                            clean_images_0_1_val[i].cpu())

                                        total_width = low_light_pil.width * 3
                                        max_height = low_light_pil.height
                                        combined_image = Image.new(
                                            'RGB', (total_width, max_height))
                                        combined_image.paste(
                                            low_light_pil, (0, 0))
                                        combined_image.paste(
                                            enhanced_pil, (low_light_pil.width, 0))
                                        combined_image.paste(
                                            clean_pil, (low_light_pil.width * 2, 0))
                                        generated_images_pil.append(
                                            combined_image)
                                val_progress_bar.update(1)
                            val_progress_bar.close()

                            valid_psnr = [
                                p for p in val_psnr_list if not math.isnan(p)]
                            valid_ssim = [
                                s for s in val_ssim_list if not math.isnan(s)]
                            avg_psnr = sum(valid_psnr) / \
                                len(valid_psnr) if valid_psnr else 0.0
                            avg_ssim = sum(valid_ssim) / \
                                len(valid_ssim) if valid_ssim else 0.0
                            logger.info(
                                f"Step {global_step} Validation Results: Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}")

                            metrics_to_log = {
                                "val_psnr": avg_psnr, "val_ssim": avg_ssim}
                            accelerator.log(metrics_to_log, step=global_step)

                            if generated_images_pil:
                                tracker_key = "validation_enhanced_samples"
                                try:
                                    accelerator.log(
                                        {tracker_key: generated_images_pil}, step=global_step)
                                    logger.info(
                                        f"验证样本图像已记录到 tracker ({args.report_to})")
                                except Exception as e:
                                    logger.warning(f"无法将验证图像记录到 tracker: {e}")

                                sample_dir = os.path.join(
                                    args.output_dir, "validation_samples")
                                os.makedirs(sample_dir, exist_ok=True)
                                for idx, img in enumerate(generated_images_pil):
                                    save_filename = os.path.join(
                                        sample_dir, f"epoch-{epoch+1}_step-{global_step}_sample-{idx}.png")
                                    try:
                                        img.save(save_filename)
                                    except Exception as save_err:
                                        logger.error(
                                            f"保存验证样本图像失败 {save_filename}: {save_err}")
                                logger.info(f"验证样本图像已保存到本地目录 {sample_dir}")

                            del unet_eval, condition_encoder_eval  # 清理
                            torch.cuda.empty_cache()
                            model.train()  # 切回训练模式
                            condition_encoder.train()  # 切回训练模式

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_dir = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}")
                    # Accelerator 会保存所有 prepare过的模型和优化器状态
                    accelerator.save_state(save_dir)
                    logger.info(f"已保存检查点到 {save_dir}")

                    if args.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith(
                                "checkpoint") and os.path.isdir(os.path.join(args.output_dir, d))],
                            key=lambda x: int(x.split('-')[1])
                        )
                        if len(ckpts) > args.checkpoints_total_limit:
                            num_to_remove = len(ckpts) - \
                                args.checkpoints_total_limit
                            for old_ckpt in ckpts[:num_to_remove]:
                                old_ckpt_path = os.path.join(
                                    args.output_dir, old_ckpt)
                                if os.path.isdir(old_ckpt_path):
                                    import shutil
                                    try:
                                        shutil.rmtree(old_ckpt_path)
                                        logger.info(
                                            f"已删除旧检查点: {old_ckpt_path}")
                                    except OSError as e:
                                        logger.error(
                                            f"删除检查点失败 {old_ckpt_path}: {e}")

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
        if global_step >= args.max_train_steps:
            logger.info("达到最大训练步数，停止训练。")
            break

    accelerator.end_training()
    logger.info("训练完成")

    if accelerator.is_main_process:
        # 保存最终模型 (U-Net 和 ConditioningEncoder)
        # Unwrap 模型
        unet_final = accelerator.unwrap_model(model)
        condition_encoder_final = accelerator.unwrap_model(condition_encoder)

        # 保存 U-Net
        save_path_unet_final = os.path.join(args.output_dir, "unet_final")
        unet_final.save_pretrained(save_path_unet_final)
        logger.info(f"最终 U-Net 模型已保存到 {save_path_unet_final}")

        # 保存 ConditioningEncoder (使用 torch.save)
        save_path_cond_enc_final = os.path.join(
            args.output_dir, "condition_encoder_final.pth")
        torch.save(condition_encoder_final.state_dict(),
                   save_path_cond_enc_final)
        logger.info(
            f"最终 ConditioningEncoder 模型已保存到 {save_path_cond_enc_final}")


if __name__ == "__main__":
    main()
