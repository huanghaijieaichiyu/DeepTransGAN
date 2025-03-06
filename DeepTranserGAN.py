'''
code by 黄小海  2025/2/19

这是一个基于PyTorch的深度学习项目，用于低光照图像增强。
数据集结构：
- datasets
    - kitti_LOL
        - eval15
            - high
            - low
        - our485
            - high
            - low
- DeepTranserGAN



'''
import os
import time
import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import StepLR
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torcheval.metrics.functional import peak_signal_noise_ratio
from torchvision import transforms
from tqdm import tqdm
import torch.amp as amp  # 更新导入以使用新的autocast API

from datasets.data_set import LowLightDataset
from models.base_mode import Generator, Discriminator, LightSWTformer
from utils.loss import BCEBlurWithLogitsLoss
from utils.misic import set_random_seed, get_opt, get_loss, ssim, model_structure, save_path


# 添加 SpectralNorm 实现，用于稳定判别器训练
def spectral_norm(module, name='weight', power_iterations=1):
    """
    对模块应用谱归一化
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        original_method = module.forward

        u = torch.randn(module.weight.size(0), 1, requires_grad=False)
        u = u.to(module.weight.device)

        def _l2normalize(v):
            return v / (torch.norm(v) + 1e-12)

        def spectral_norm_forward(*args, **kwargs):
            weight = module.weight
            weight_mat = weight.view(weight.size(0), -1)

            for _ in range(power_iterations):
                v = _l2normalize(torch.matmul(weight_mat.t(), u))
                u.data = _l2normalize(torch.matmul(weight_mat, v))

            sigma = torch.dot(u.view(-1), torch.matmul(weight_mat, v).view(-1))
            weight = weight / sigma

            return original_method(*args, **kwargs)

        module.forward = spectral_norm_forward

    return module

# 添加 SpectralNormConv2d 类，用于替换判别器中的卷积层


class SpectralNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=bias)
        self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)


class BaseTrainer:
    def __init__(self, args, generator, discriminator=None, critic=None):
        self.n_pretain = 2  # 减少判别器预训练次数，避免过度训练
        self.args = args
        self.generator = generator
        self.discriminator = discriminator
        self.critic = critic
        self.device = torch.device('cpu')
        if args.device == 'cuda':
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.generator = self.generator.to(self.device)

        # 初始化生成器权重
        self._initialize_weights(self.generator)

        if self.discriminator is not None:
            self.discriminator = self.discriminator.to(self.device)
            # 初始化判别器权重
            self._initialize_weights(self.discriminator)
            # 应用谱归一化到判别器
            self._apply_spectral_norm(self.discriminator)
        if self.critic is not None:
            self.critic = self.critic.to(self.device)
            # 初始化评论器权重
            self._initialize_weights(self.critic)
            # 应用谱归一化到评论器
            self._apply_spectral_norm(self.critic)

        # 增强数据增强
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomRotation(10),  # 随机旋转±10度
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 亮度和对比度变化
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 小幅平移
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到 [-1, 1]
        ])

        self.train_data = LowLightDataset(
            image_dir=args.data, transform=self.transform, phase="train")
        self.train_loader = DataLoader(self.train_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                       shuffle=True, drop_last=True, pin_memory=True)  # 启用pin_memory加速数据传输

        # 测试数据不需要增强，但需要标准化
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到 [-1, 1]
        ])

        self.test_data = LowLightDataset(
            image_dir=args.data, transform=self.test_transform, phase="test")
        self.test_loader = DataLoader(self.test_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                      shuffle=False, drop_last=False, pin_memory=True)

        # 优化器初始化 - 使用不同的学习率
        self.g_optimizer, self.d_optimizer = self._get_optimizers(args)

        # 损失函数组合
        self.g_loss = get_loss(args.loss).to(
            self.device) if args.loss else nn.MSELoss().to(self.device)
        self.stable_loss = nn.SmoothL1Loss().to(self.device)

        # 路径设置
        self.path = save_path(
            args.save_path) if args.resume == '' else args.resume
        self.log = tensorboard.writer.SummaryWriter(log_dir=self.path, filename_suffix=str(args.epochs),
                                                    flush_secs=180)

        # 模型保存策略
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.patience = args.patience if hasattr(args, 'patience') else 10
        self.patience_counter = 0
        self.eval_interval = 5  # 每5个epoch评估一次

        # 学习率调度 - 使用余弦退火
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.g_optimizer, T_0=10, T_mult=2, eta_min=args.lr * 0.001)
        if self.d_optimizer is not None:
            self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.d_optimizer, T_0=10, T_mult=2, eta_min=args.lr * 0.001)
        else:
            self.scheduler_d = None

        # 梯度累积步数
        self.grad_accum_steps = args.grad_accum_steps if hasattr(
            args, 'grad_accum_steps') else 1

        # 标签平滑参数
        self.label_smoothing = 0.1

        # 梯度惩罚权重
        self.lambda_gp = 10

        # 添加噪声到判别器输入
        self.noise_factor = 0.05

        # 两时间尺度更新规则 (TTUR)
        self.d_updates_per_g = 2  # 每训练一次生成器，训练判别器的次数

        # NaN检测和处理
        self.nan_detected = False
        self.nan_count = 0
        self.max_nan_count = 5  # 连续NaN次数阈值

        # 感知损失权重
        self.perceptual_weight = 0.5  # 增加感知损失权重

        # 确保路径存在
        if self.args.resume == '':
            os.makedirs(os.path.join(self.path, 'generator'), exist_ok=True)
            if self.discriminator is not None:
                os.makedirs(os.path.join(
                    self.path, 'discriminator'), exist_ok=True)
            if self.critic is not None:
                os.makedirs(os.path.join(self.path, 'critic'), exist_ok=True)

        self.train_log = self.path + '/log.txt'
        self.args_dict = args.__dict__
        self.epoch = 0
        self.Ssim = [0.]
        self.PSN = [0.]

        # 记录训练配置
        self._log_training_config()

        # 导入 rich 库用于美化显示
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
            self.rich_available = True
            self.console = Console()
        except ImportError:
            self.rich_available = False
            print("提示: 安装 rich 库可以获得更美观的训练显示效果 (pip install rich)")

        # 简化损失函数配置
        self.pixel_loss_weight = 10.0  # 像素损失权重

        # 添加梯度缩放因子，防止梯度爆炸
        self.gradient_scale = 0.1

        # 添加学习率调整因子
        self.lr_decay_factor = 0.5
        self.lr_decay_patience = 5
        self.lr_decay_counter = 0

        # 添加梯度裁剪值
        self.grad_clip_discriminator = 0.5
        self.grad_clip_generator = 1.0

    def _log_training_config(self):
        """记录训练配置到日志文件"""
        with open(self.train_log, "a") as f:
            f.write("=== Training Configuration ===\n")
            for key, value in self.args_dict.items():
                f.write(f"{key}: {value}\n")
            f.write("============================\n\n")

    def load_checkpoint(self):
        if self.args.resume != '':
            # 加载生成器
            g_path_checkpoint = os.path.join(
                self.args.resume, 'generator/last.pt')
            if os.path.exists(g_path_checkpoint):
                g_checkpoint = torch.load(
                    g_path_checkpoint, map_location=self.device)
                self.generator.load_state_dict(g_checkpoint['net'])
                self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
                self.epoch = g_checkpoint['epoch']
                if 'best_psnr' in g_checkpoint:
                    self.best_psnr = g_checkpoint['best_psnr']
                if 'best_ssim' in g_checkpoint:
                    self.best_ssim = g_checkpoint['best_ssim']
                if 'scheduler' in g_checkpoint:
                    self.scheduler_g.load_state_dict(g_checkpoint['scheduler'])
            else:
                raise FileNotFoundError(
                    f"Generator checkpoint {g_path_checkpoint} not found.")

            # 加载判别器或评论器
            d_path_checkpoint = os.path.join(
                self.args.resume, 'discriminator/last.pt') if self.discriminator else os.path.join(self.args.resume, 'critic/last.pt')
            if os.path.exists(d_path_checkpoint):
                d_checkpoint = torch.load(
                    d_path_checkpoint, map_location=self.device)
                if self.discriminator:
                    self.discriminator.load_state_dict(d_checkpoint['net'])
                elif self.critic:
                    self.critic.load_state_dict(d_checkpoint['net'])
                if self.d_optimizer is not None:
                    self.d_optimizer.load_state_dict(d_checkpoint['optimizer'])
                    if 'scheduler' in d_checkpoint and self.scheduler_d is not None:
                        self.scheduler_d.load_state_dict(
                            d_checkpoint['scheduler'])
            else:
                raise FileNotFoundError(
                    f"Discriminator/Critic checkpoint {d_path_checkpoint} not found.")

            print(f'Continuing training from epoch: {self.epoch + 1}')
            self.path = self.args.resume
            self.args.resume = ''

    def save_checkpoint(self, is_best=False):
        """保存检查点，可选择是否为最佳模型"""
        save_path = os.path.join(
            self.path, 'generator', 'best.pt' if is_best else 'last.pt')

        # 保存生成器
        g_checkpoint = {
            'net': self.generator.state_dict(),
            'optimizer': self.g_optimizer.state_dict(),
            'epoch': self.epoch,
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'scheduler': self.scheduler_g.state_dict()
        }
        torch.save(g_checkpoint, save_path)

        # 保存判别器或评论器
        d_save_path = os.path.join(
            self.path,
            'discriminator' if self.discriminator else 'critic',
            'best.pt' if is_best else 'last.pt'
        )

        d_net_state = None
        if self.discriminator is not None:
            d_net_state = self.discriminator.state_dict()
        elif self.critic is not None:
            d_net_state = self.critic.state_dict()

        if d_net_state is not None and self.d_optimizer is not None:
            d_checkpoint = {
                'net': d_net_state,
                'optimizer': self.d_optimizer.state_dict(),
                'epoch': self.epoch,
            }
            if self.scheduler_d is not None:
                d_checkpoint['scheduler'] = self.scheduler_d.state_dict()

            torch.save(d_checkpoint, d_save_path)

    def log_message(self, message):
        print(message)
        with open(self.train_log, "a") as f:
            f.write(f"{message}\n")

    def write_log(self, epoch, gen_loss, dis_loss, d_x, d_g_z1, d_g_z2):
        train_log_txt_formatter = (
            '{time_str} \t [Epoch] \t {epoch:03d} \t [gLoss] \t {gloss_str} \t [dLoss] \t {dloss_str} \t {Dx_str} \t ['
            'Dgz0] \t {Dgz0_str} \t [Dgz1] \t {Dgz1_str}\n')
        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"), epoch=epoch + 1,
                                                  gloss_str=" ".join(
                                                      ["{:4f}".format(np.mean(gen_loss))]),
                                                  dloss_str=" ".join(
                                                      ["{:4f}".format(np.mean(dis_loss))]),
                                                  Dx_str=" ".join(
                                                      ["{:4f}".format(d_x)]),
                                                  Dgz0_str=" ".join(
                                                      ["{:4f}".format(d_g_z1)]),
                                                  Dgz1_str=" ".join(["{:4f}".format(d_g_z2)]))
        with open(self.train_log, "a") as f:
            f.write(to_write)

    def visualize_results(self, epoch, gen_loss, dis_loss, high_images, low_images, fake):
        self.log.add_scalar('generation loss', np.mean(gen_loss), epoch + 1)
        self.log.add_scalar('discrimination loss',
                            np.mean(dis_loss), epoch + 1)
        self.log.add_scalar('learning rate', self.g_optimizer.state_dict()[
                            'param_groups'][0]['lr'], epoch + 1)
        self.log.add_images('real', high_images, epoch + 1)
        self.log.add_images('input', low_images, epoch + 1)
        self.log.add_images('fake', fake, epoch + 1)

    def evaluate_model(self):
        with torch.no_grad():
            self.log_message(
                f"Evaluating the generator model at epoch {self.epoch + 1}")
            self.generator.eval()
            if self.discriminator:
                self.discriminator.eval()
            elif self.critic:
                self.critic.eval()

            Ssim = []
            PSN = []

            for i, (low_images, high_images) in enumerate(self.test_loader):
                low_images = low_images.to(self.device)
                high_images = high_images.to(self.device)

                # 生成增强图像
                fake_eval = self.generator(low_images)

                # 计算SSIM和PSNR
                ssim_value = ssim(fake_eval, high_images).item()
                psnr_value = peak_signal_noise_ratio(
                    fake_eval, high_images).item()

                Ssim.append(ssim_value)
                PSN.append(psnr_value)

            # 计算平均指标
            avg_ssim = np.mean(Ssim)
            avg_psnr = np.mean(PSN)

            # 记录到TensorBoard
            self.log.add_scalar('SSIM', avg_ssim, self.epoch + 1)
            self.log.add_scalar('PSNR', avg_psnr, self.epoch + 1)

            self.log_message(
                f"Model SSIM: {avg_ssim:.4f}  PSNR: {avg_psnr:.4f}")

            # 检查是否为最佳模型
            is_best = False
            if avg_psnr > self.best_psnr:
                self.log_message(
                    f"New best PSNR: {avg_psnr:.4f} (previous: {self.best_psnr:.4f})")
                self.best_psnr = avg_psnr
                is_best = True
                self.patience_counter = 0
            elif avg_ssim > self.best_ssim:
                self.log_message(
                    f"New best SSIM: {avg_ssim:.4f} (previous: {self.best_ssim:.4f})")
                self.best_ssim = avg_ssim
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                self.log_message(
                    f"No improvement. Patience: {self.patience_counter}/{self.patience}")

            # 如果是最佳模型，保存检查点
            if is_best:
                self.save_checkpoint(is_best=True)

            # 检查早停
            if self.patience_counter >= self.patience:
                self.log_message("Early stopping triggered.")
                return True  # 停止训练

            return False

    def create_rich_progress(self):
        """创建美化的进度条"""
        if not self.rich_available:
            return None

        from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=True
        )

    def print_epoch_summary(self, epoch, gen_loss, dis_loss, metrics=None):
        """打印每个epoch结束时的训练摘要"""
        metrics = metrics or {}

        # 计算平均损失
        avg_gen_loss = sum(gen_loss) / len(gen_loss) if gen_loss else 0
        avg_dis_loss = sum(dis_loss) / len(dis_loss) if dis_loss else 0

        if self.rich_available:
            from rich.table import Table

            # 创建表格
            table = Table(
                title=f"📊 训练摘要 - Epoch {epoch + 1}/{self.args.epochs}", expand=True)

            # 添加列
            table.add_column("类别", style="cyan")
            table.add_column("指标", style="magenta")
            table.add_column("数值", style="green")

            # 添加损失数据
            table.add_row("损失", "生成器平均损失", f"{avg_gen_loss:.6f}")
            table.add_row("损失", "判别器平均损失", f"{avg_dis_loss:.6f}")

            # 添加学习率数据
            if hasattr(self, 'g_optimizer') and self.g_optimizer is not None:
                g_lr = self.g_optimizer.param_groups[0]['lr']
                table.add_row("学习率", "生成器", f"{g_lr:.6f}")
            if hasattr(self, 'd_optimizer') and self.d_optimizer is not None:
                d_lr = self.d_optimizer.param_groups[0]['lr']
                table.add_row("学习率", "判别器/评论器", f"{d_lr:.6f}")

            # 添加其他指标
            for key, value in metrics.items():
                table.add_row("指标", key, f"{value}" if isinstance(
                    value, str) else f"{value:.4f}")

            # 添加内存使用情况
            if torch.cuda.is_available():
                table.add_row(
                    "GPU内存", "已分配", f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                table.add_row(
                    "GPU内存", "缓存", f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB")
                table.add_row(
                    "GPU内存", "最大分配", f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

            # 打印表格
            self.console.print()
            self.console.print(table)
            self.console.print()
        else:
            # 创建分隔线
            separator = "=" * 80

            # 打印摘要
            print(f"\n{separator}")
            print(f"📊 训练摘要 - Epoch {epoch + 1}/{self.args.epochs}")
            print(f"{separator}")

            # 损失信息
            print(f"📉 损失统计:")
            print(f"   生成器平均损失: {avg_gen_loss:.6f}")
            print(f"   判别器平均损失: {avg_dis_loss:.6f}")

            # 学习率信息
            print(f"🔍 学习率:")
            if hasattr(self, 'g_optimizer') and self.g_optimizer is not None:
                g_lr = self.g_optimizer.param_groups[0]['lr']
                print(f"   生成器: {g_lr:.6f}")
            if hasattr(self, 'd_optimizer') and self.d_optimizer is not None:
                d_lr = self.d_optimizer.param_groups[0]['lr']
                print(f"   判别器/评论器: {d_lr:.6f}")

            # 其他指标
            if metrics:
                print(f"📈 其他指标:")
                for key, value in metrics.items():
                    print(f"   {key}: {value}")

            # 内存使用情况
            if torch.cuda.is_available():
                print(f"💾 GPU 内存:")
                print(
                    f"   已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(
                    f"   缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
                print(
                    f"   最大分配: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

            print(f"{separator}\n")

    def train_epoch(self):
        raise NotImplementedError

    def train(self):
        set_random_seed(
            self.args.seed, deterministic=self.args.deterministic, benchmark=self.args.benchmark)
        self.load_checkpoint()

        stop_training = False
        while self.epoch < self.args.epochs and not stop_training:
            # 训练一个epoch
            self.train_epoch()

            # 更新学习率
            self.scheduler_g.step()
            if self.scheduler_d is not None:
                self.scheduler_d.step()

            # 定期评估模型
            if (self.epoch + 1) % self.eval_interval == 0:
                stop_training = self.evaluate_model()

            # 保存最新检查点
            self.save_checkpoint()

            self.epoch += 1

        self.log.close()
        self.log_message(f"Training completed after {self.epoch} epochs.")

    def _apply_spectral_norm(self, model):
        """对模型中的卷积层应用谱归一化"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                spectral_norm(module)

    def _get_optimizers(self, args):
        """获取优化器，为生成器和判别器使用不同的学习率"""
        # 生成器使用较小的学习率
        g_lr = args.lr * 0.5
        # 判别器使用较大的学习率 (TTUR)
        d_lr = args.lr * 2.0

        # 为生成器使用 Adam 优化器，较小的 beta 值
        g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=g_lr,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay if hasattr(
                args, 'weight_decay') else 1e-5
        )

        # 为判别器使用 RMSprop 优化器，更稳定
        d_optimizer = None
        if self.discriminator is not None:
            d_optimizer = torch.optim.RMSprop(
                self.discriminator.parameters(),
                lr=d_lr,
                weight_decay=args.weight_decay if hasattr(
                    args, 'weight_decay') else 1e-5
            )
        elif self.critic is not None:
            d_optimizer = torch.optim.RMSprop(
                self.critic.parameters(),
                lr=d_lr,
                weight_decay=args.weight_decay if hasattr(
                    args, 'weight_decay') else 1e-5
            )

        return g_optimizer, d_optimizer

    def add_noise_to_input(self, tensor, noise_factor=None):
        """向输入添加噪声，提高稳定性"""
        if noise_factor is None:
            noise_factor = self.noise_factor

        if noise_factor > 0:
            noise = torch.randn_like(tensor) * noise_factor
            return tensor + noise
        return tensor

    def _initialize_weights(self, model):
        """初始化模型权重，使用Kaiming初始化"""
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _check_nan_values(self, loss_value, model_name):
        """检查NaN值并处理"""
        if torch.isnan(loss_value).any() or torch.isinf(loss_value).any():
            self.nan_detected = True
            self.nan_count += 1
            self.log_message(f"警告: 检测到NaN/Inf值在{model_name}损失中，尝试恢复训练...")

            # 如果连续多次检测到NaN，降低学习率
            if self.nan_count >= self.max_nan_count:
                self.log_message(f"连续{self.max_nan_count}次检测到NaN，降低学习率...")
                for param_group in self.g_optimizer.param_groups:
                    param_group['lr'] *= 0.5
                if self.d_optimizer is not None:
                    for param_group in self.d_optimizer.param_groups:
                        param_group['lr'] *= 0.5
                self.nan_count = 0

            return True
        else:
            self.nan_detected = False
            self.nan_count = 0
            return False

    def _check_gradients(self, model, model_name):
        """检查梯度是否包含NaN值"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    self.log_message(
                        f"警告: 检测到NaN/Inf梯度在{model_name}的{name}参数中")
                    param.grad.data.zero_()  # 将NaN梯度置零
                    return True
        return False


class StandardGANTrainer(BaseTrainer):
    def __init__(self, args, generator, discriminator):
        super().__init__(args, generator, discriminator=discriminator)
        if discriminator is None:
            raise ValueError("Discriminator model is not initialized.")
        # 使用 BCEWithLogitsLoss 以兼容 autocast
        self.d_loss = nn.BCEWithLogitsLoss().to(self.device)

        # 添加 L1 损失用于像素级监督
        self.l1_loss = nn.L1Loss().to(self.device)

        # 添加梯度缩放因子，防止梯度爆炸
        self.gradient_scale = 0.1

        # 添加学习率调整因子
        self.lr_decay_factor = 0.5
        self.lr_decay_patience = 5
        self.lr_decay_counter = 0

        # 添加梯度裁剪值
        self.grad_clip_discriminator = 0.5
        self.grad_clip_generator = 1.0

    def compute_generator_loss(self, fake_outputs, fake_images, real_images):
        """计算生成器损失 - 简化版本"""
        # 对抗损失 - 希望判别器将生成的图像识别为真实图像
        adv_loss = self.d_loss(fake_outputs, torch.ones_like(fake_outputs))

        # 像素级损失 - 生成的图像应该与真实图像相似
        pixel_loss = self.l1_loss(fake_images, real_images)

        # 总损失 = 对抗损失 + 加权像素损失
        total_loss = adv_loss + pixel_loss * self.pixel_loss_weight

        return total_loss

    def train_epoch(self):
        if self.discriminator is None:
            self.log_message(
                "Discriminator is not initialized. Cannot train GAN.")
            return

        self.discriminator.train()
        self.generator.train()
        source_g = [0.]
        d_g_z2 = 0.
        gen_loss = []
        dis_loss = []

        # 使用 rich 进度条（如果可用）
        progress = None
        task_id = None
        if hasattr(self, 'rich_available') and self.rich_available:
            progress = self.create_rich_progress()
            if progress is not None:
                task_id = progress.add_task(
                    f"[cyan]Epoch {self.epoch + 1}/{self.args.epochs}", total=len(self.train_loader))
                progress.start()
        else:
            # 美化进度条
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                        bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                        colour='green', ncols=100)

        # 梯度累积相关变量
        d_loss_accumulator = 0
        g_loss_accumulator = 0
        batch_count = 0

        for i, (low_images, high_images) in enumerate(self.train_loader):
            batch_count += 1
            low_images = low_images.to(self.device)
            high_images = high_images.to(self.device)

            # 检查输入数据是否包含NaN
            if torch.isnan(low_images).any() or torch.isnan(high_images).any():
                self.log_message("警告: 输入数据包含NaN值，跳过此批次")
                continue

            # 使用混合精度训练 - 使用正确的autocast
            with autocast(enabled=self.args.autocast):
                # 训练判别器
                for j in range(self.d_updates_per_g):  # 使用 TTUR
                    if self.d_optimizer is not None:
                        self.d_optimizer.zero_grad(
                            set_to_none=True)  # 更高效的梯度清零
                    else:
                        continue  # 如果没有判别器优化器，跳过判别器训练

                    # 生成假图像
                    with torch.no_grad():
                        fake = self.generator(low_images)
                    # 判别器对假图像的判断
                    fake_inputs = self.discriminator(fake)
                    # 判别器对真图像的判断
                    real_inputs = self.discriminator(high_images)

                    # 检查判别器输出是否包含NaN
                    if torch.isnan(fake_inputs).any() or torch.isnan(real_inputs).any():
                        self.log_message("警告: 判别器输出包含NaN值，跳过此批次")
                        break
                    # 创建标签 - 使用标签平滑
                    real_label = torch.ones_like(fake_inputs, requires_grad=False) * (1 - self.label_smoothing) + \
                        torch.rand_like(fake_inputs) * self.label_smoothing
                    fake_label = torch.zeros_like(fake_inputs, requires_grad=False) + \
                        torch.rand_like(fake_inputs) * self.label_smoothing

                    # 计算判别器损失
                    d_real_output = self.d_loss(real_inputs, real_label)
                    d_x = real_inputs.mean().item()

                    d_fake_output = self.d_loss(fake_inputs, fake_label)
                    d_g_z1 = fake_inputs.mean().item()

                    # 总判别器损失 - 简化版本
                    d_output = (d_real_output + d_fake_output) / 2.0

                    # 添加梯度惩罚 - 使用改进的梯度惩罚函数
                    if self.lambda_gp > 0:
                        gp = compute_gradient_penalty(
                            self.discriminator, high_images, fake, self.lambda_gp)
                        d_output = d_output + gp

                    # 检查损失值是否为NaN
                    if self._check_nan_values(d_output, "判别器"):
                        break

                    d_loss_accumulator += d_output.item()

                # 使用scaler进行反向传播和优化器步进
                d_output.backward()
                self.d_optimizer.step()

                # 检查梯度是否包含NaN
                if self.discriminator is not None:
                    if self._check_gradients(self.discriminator, "判别器"):
                        # 如果检测到NaN梯度，降低学习率
                        self.lr_decay_counter += 1
                        if self.lr_decay_counter >= self.lr_decay_patience and self.d_optimizer is not None:
                            for param_group in self.d_optimizer.param_groups:
                                param_group['lr'] *= self.lr_decay_factor
                            self.log_message(
                                f"降低判别器学习率至 {self.d_optimizer.param_groups[0]['lr']}")
                            self.lr_decay_counter = 0
                        break

                # 训练生成器
                self.g_optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

                # 重新生成假图像（因为需要梯度）
                fake = self.generator(low_images)

                # 判别器对新生成的假图像的判断
                fake_inputs = self.discriminator(fake)

                # 计算生成器损失 - 使用简化的损失计算函数
                g_output = self.compute_generator_loss(
                    fake_inputs, fake, high_images)

                g_loss_accumulator += g_output.item()

                d_g_z2 = fake_inputs.mean().item()

                # 使用scaler进行反向传播
                g_output.backward()
                self.g_optimizer.step()

                # 检查梯度是否包含NaN
                if self._check_gradients(self.generator, "生成器"):
                    # 如果检测到NaN梯度，降低学习率
                    self.lr_decay_counter += 1
                    if self.lr_decay_counter >= self.lr_decay_patience and self.g_optimizer is not None:
                        for param_group in self.g_optimizer.param_groups:
                            param_group['lr'] *= self.lr_decay_factor
                        self.log_message(
                            f"降低生成器学习率至 {self.g_optimizer.param_groups[0]['lr']}")
                        self.log_message(
                            f"降低生成器学习率至 {self.g_optimizer.param_groups[0]['lr']}")
                        self.lr_decay_counter = 0
                    continue

                # 梯度累积
                if batch_count % self.grad_accum_steps == 0 or i == len(self.train_loader) - 1:
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), self.grad_clip_generator)
                    # 记录损失
                    gen_loss.append(g_loss_accumulator / self.grad_accum_steps)
                    dis_loss.append(d_loss_accumulator / self.grad_accum_steps)

                    # 重置累积器
                    g_loss_accumulator = 0
                    d_loss_accumulator = 0
                    batch_count = 0

                source_g.append(d_g_z2)

                # 更新进度条描述
                if hasattr(self, 'rich_available') and self.rich_available and progress is not None and task_id is not None:
                    epoch_info = f"Epoch: [{self.epoch + 1}/{self.args.epochs}]"
                    batch_info = f"Batch: [{i + 1}/{len(self.train_loader)}]"
                    loss_info = f"D: {d_output.item():.4f} | G: {g_output.item():.4f}"
                    metrics = f"D(x): {d_x:.3f} | D(G(z)): {d_g_z2:.3f}"

                    # 安全获取学习率
                    lr_g = self.g_optimizer.param_groups[0]['lr'] if hasattr(
                        self, 'g_optimizer') and self.g_optimizer is not None else 0
                    lr_d = self.d_optimizer.param_groups[0]['lr'] if hasattr(
                        self, 'd_optimizer') and self.d_optimizer is not None else 0
                    lr_info = f"lr_G: {lr_g:.6f} | lr_D: {lr_d:.6f}"

                    progress.update(
                        task_id, advance=1, description=f"[cyan]{epoch_info} | {batch_info} | {loss_info} | {metrics} | {lr_info}")
                else:
                    epoch_info = f"Epoch: [{self.epoch + 1}/{self.args.epochs}]"
                    batch_info = f"Batch: [{i + 1}/{len(self.train_loader)}]"
                    loss_info = f"D: {d_output.item():.4f} | G: {g_output.item():.4f}"
                    metrics = f"D(x): {d_x:.3f} | D(G(z)): {d_g_z2:.3f}"

                    # 添加学习率信息
                    lr_g = self.g_optimizer.param_groups[0]['lr'] if hasattr(
                        self, 'g_optimizer') and self.g_optimizer is not None else 0
                    lr_d = self.d_optimizer.param_groups[0]['lr'] if hasattr(
                        self, 'd_optimizer') and self.d_optimizer is not None else 0
                    lr_info = f"lr_G: {lr_g:.6f} | lr_D: {lr_d:.6f}"

                    if 'pbar' in locals():
                        pbar.set_description(
                            f"🔄 {epoch_info} | {batch_info} | 📉 {loss_info} | 📊 {metrics} | 🔍 {lr_info}"
                        )

                # 每个epoch结束时保存检查点，而不是每个batch
                if i == len(self.train_loader) - 1:
                    self.save_checkpoint()

        # 关闭进度条
        if hasattr(self, 'rich_available') and self.rich_available and progress is not None:
            progress.stop()
        elif 'pbar' in locals():
            pbar.close()

        # 记录训练日志
        self.write_log(self.epoch, gen_loss, dis_loss, d_x, d_g_z1, d_g_z2)

        # 打印训练摘要
        metrics = {
            "D(x)": d_x,
            "D(G(z))": d_g_z2,
            "PSNR": self.evaluate_model() if self.epoch % 5 == 0 else "未计算"
        }
        self.print_epoch_summary(self.epoch, gen_loss, dis_loss, metrics)

        # 可视化结果
        self.visualize_results(self.epoch, gen_loss,
                               dis_loss, high_images, low_images, fake)


class WGAN_GPTrainer(BaseTrainer):
    def __init__(self, args, generator, critic):
        super().__init__(args, generator, critic=critic)
        self.lambda_gp = 10
        if self.critic is None:
            raise ValueError("Critic model is not initialized.")

        # 使用自定义优化器
        self.c_optimizer, self.g_optimizer = self._get_wgan_optimizers(args)

        # 添加 L1 损失用于像素级监督
        self.l1_loss = nn.L1Loss().to(self.device)

    def _get_wgan_optimizers(self, args):
        """获取 WGAN 专用优化器"""
        # 生成器使用较小的学习率
        g_lr = args.lr * 0.5
        # 评论器使用较大的学习率 (TTUR)
        c_lr = args.lr * 2.0

        # 为生成器使用 Adam 优化器，较小的 beta 值
        g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=g_lr,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay if hasattr(
                args, 'weight_decay') else 1e-5
        )

        # 为评论器使用 RMSprop 优化器，更稳定
        c_optimizer = None
        if self.critic is not None:
            c_optimizer = torch.optim.RMSprop(
                self.critic.parameters(),
                lr=c_lr,
                weight_decay=args.weight_decay if hasattr(
                    args, 'weight_decay') else 1e-5
            )

        return c_optimizer, g_optimizer

    def compute_generator_loss(self, critic_fake, fake_images, real_images):
        """计算生成器损失 - 简化版本"""
        # 对抗损失 - 希望评论器给生成的图像高分
        adv_loss = -torch.mean(critic_fake)

        # 像素级损失 - 生成的图像应该与真实图像相似
        pixel_loss = self.l1_loss(fake_images, real_images)

        # 总损失 = 对抗损失 + 加权像素损失 + 加权感知损失
        total_loss = adv_loss + pixel_loss * self.pixel_loss_weight
        return total_loss

    def train_epoch(self):
        if self.critic is None:
            self.log_message(
                "Critic is not initialized. Cannot train WGAN-GP.")
            return

        self.critic.train()
        self.generator.train()
        source_g = [0.]
        g_z = 0.
        gen_loss = []
        critic_loss = []

        # 使用 rich 进度条（如果可用）
        progress = None
        task_id = None
        if hasattr(self, 'rich_available') and self.rich_available:
            progress = self.create_rich_progress()
            if progress is not None:
                task_id = progress.add_task(
                    f"[cyan]Epoch {self.epoch + 1}/{self.args.epochs}", total=len(self.train_loader))
                progress.start()
        else:
            # 美化进度条
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                        bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                        colour='blue', ncols=100)

        # 梯度累积相关变量
        c_loss_accumulator = 0
        g_loss_accumulator = 0
        batch_count = 0

        for i, (low_images, high_images) in enumerate(self.train_loader):
            batch_count += 1
            low_images = low_images.to(self.device)
            high_images = high_images.to(self.device)

            # 检查输入数据是否包含NaN
            if torch.isnan(low_images).any() or torch.isnan(high_images).any():
                self.log_message("警告: 输入数据包含NaN值，跳过此批次")
                continue

            # 使用混合精度训练 - 使用正确的autocast
            with autocast(enabled=self.args.autocast):
                # 训练评论器（Critic）
                for j in range(self.d_updates_per_g):  # 使用 TTUR
                    if self.c_optimizer is not None:
                        self.c_optimizer.zero_grad(set_to_none=True)
                    else:
                        continue  # 如果没有评论器优化器，跳过评论器训练

                    # 生成假图像
                    with torch.no_grad():
                        fake_images = self.generator(low_images)

                    # 检查生成的图像是否包含NaN
                    if torch.isnan(fake_images).any():
                        self.log_message("警告: 生成器输出包含NaN值，跳过此批次")
                        break

                    # 添加噪声到输入
                    real_with_noise = self.add_noise_to_input(high_images)
                    fake_with_noise = self.add_noise_to_input(
                        fake_images.detach())

                    # 评论器对真假图像的评价
                    critic_real = self.critic(real_with_noise)
                    critic_fake = self.critic(fake_with_noise)

                    # 检查评论器输出是否包含NaN
                    if torch.isnan(critic_real).any() or torch.isnan(critic_fake).any():
                        self.log_message("警告: 评论器输出包含NaN值，跳过此批次")
                        break

                    # 计算梯度惩罚
                    gradient_penalty = compute_gradient_penalty(self.critic,
                                                                real_with_noise, fake_with_noise)

                    # 检查梯度惩罚是否为NaN
                    if torch.isnan(gradient_penalty).any():
                        self.log_message("警告: 梯度惩罚计算结果为NaN，跳过此批次")
                        break

                    # 计算Wasserstein距离和总损失
                    wasserstein_distance = torch.mean(
                        critic_real) - torch.mean(critic_fake)
                    loss_critic = -wasserstein_distance + self.lambda_gp * gradient_penalty

                    # 检查损失值是否为NaN
                    if self._check_nan_values(loss_critic, "评论器"):
                        break

                    # 累积损失
                    c_loss_accumulator += loss_critic.item()

                    # 使用scaler进行反向传播和优化器步进
                    loss_critic.backward()
                    self.c_optimizer.step()

                    # 检查梯度是否包含NaN
                    if self.critic is not None:
                        if self._check_gradients(self.critic, "评论器"):
                            break

                    # 梯度累积
                    if (j == self.d_updates_per_g - 1) and (batch_count % self.grad_accum_steps == 0 or i == len(self.train_loader) - 1):
                        # 梯度裁剪
                        if self.critic is not None and self.c_optimizer is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.critic.parameters(), 1.0)
                            self.c_optimizer.step()

                # 如果检测到NaN，跳过生成器训练
                if self.nan_detected:
                    continue

                # 训练生成器
                self.g_optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

                # 生成新的假图像
                fake_images = self.generator(low_images)

                # 检查生成的图像是否包含NaN
                if torch.isnan(fake_images).any():
                    self.log_message("警告: 生成器输出包含NaN值，跳过此批次")
                    continue

                # 评论器对新生成的假图像的评价
                critic_fake = self.critic(fake_images)

                # 检查评论器输出是否包含NaN
                if torch.isnan(critic_fake).any():
                    self.log_message("警告: 评论器输出包含NaN值，跳过此批次")
                    continue

                # 计算生成器损失 - 使用简化的损失计算函数
                loss_generator = self.compute_generator_loss(
                    critic_fake, fake_images, high_images)

                # 检查损失值是否为NaN
                if self._check_nan_values(loss_generator, "生成器"):
                    continue

                g_loss_accumulator += loss_generator.item()

                # 记录评论器对假图像的平均评分
                g_z = critic_fake.mean().item()

                # 使用scaler进行反向传播
                loss_generator.backward()
                self.g_optimizer.step()

                # 检查梯度是否包含NaN
                if self._check_gradients(self.generator, "生成器"):
                    break

                # 梯度累积
                if batch_count % self.grad_accum_steps == 0 or i == len(self.train_loader) - 1:
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), 5.0)
                    self.g_optimizer.step()

                    # 记录损失
                    gen_loss.append(g_loss_accumulator / self.grad_accum_steps)
                    critic_loss.append(c_loss_accumulator /
                                       self.grad_accum_steps)

                    # 重置累积器
                    g_loss_accumulator = 0
                    c_loss_accumulator = 0
                    batch_count = 0

                source_g.append(g_z)

                # 更新进度条描述
                if hasattr(self, 'rich_available') and self.rich_available and progress is not None and task_id is not None:
                    epoch_info = f"Epoch: [{self.epoch + 1}/{self.args.epochs}]"
                    batch_info = f"Batch: [{i + 1}/{len(self.train_loader)}]"
                    loss_info = f"C: {loss_critic.item():.4f} | G: {loss_generator.item():.4f}"
                    metrics = f"G(z): {g_z:.3f}"

                    # 安全获取学习率
                    lr_g = self.g_optimizer.param_groups[0]['lr'] if hasattr(
                        self, 'g_optimizer') and self.g_optimizer is not None else 0
                    lr_c = self.c_optimizer.param_groups[0]['lr'] if hasattr(
                        self, 'c_optimizer') and self.c_optimizer is not None else 0
                    lr_info = f"lr_G: {lr_g:.6f} | lr_C: {lr_c:.6f}"

                    progress.update(
                        task_id, advance=1, description=f"[cyan]{epoch_info} | {batch_info} | {loss_info} | {metrics} | {lr_info}")
                else:
                    epoch_info = f"Epoch: [{self.epoch + 1}/{self.args.epochs}]"
                    batch_info = f"Batch: [{i + 1}/{len(self.train_loader)}]"
                    loss_info = f"C: {loss_critic.item():.4f} | G: {loss_generator.item():.4f}"
                    metrics = f"G(z): {g_z:.3f}"

                    # 添加学习率信息
                    lr_g = self.g_optimizer.param_groups[0]['lr'] if hasattr(
                        self, 'g_optimizer') and self.g_optimizer is not None else 0
                    lr_c = self.c_optimizer.param_groups[0]['lr'] if hasattr(
                        self, 'c_optimizer') and self.c_optimizer is not None else 0
                    lr_info = f"lr_G: {lr_g:.6f} | lr_C: {lr_c:.6f}"

                    if 'pbar' in locals():
                        pbar.set_description(
                            f"🔄 {epoch_info} | {batch_info} | 📉 {loss_info} | 📊 {metrics} | 🔍 {lr_info}"
                        )

                # 每个epoch结束时保存检查点，而不是每个batch
                if i == len(self.train_loader) - 1:
                    self.save_checkpoint()

        # 关闭进度条
        if hasattr(self, 'rich_available') and self.rich_available and progress is not None:
            progress.stop()
        elif 'pbar' in locals():
            pbar.close()

        # 记录训练日志
        self.write_log(self.epoch, gen_loss, critic_loss, 0, 0, g_z)

        # 打印训练摘要
        metrics = {
            "G(z)": g_z,
            "PSNR": self.evaluate_model() if self.epoch % 5 == 0 else "未计算"
        }
        self.print_epoch_summary(self.epoch, gen_loss, critic_loss, metrics)

        # 可视化结果
        self.visualize_results(
            self.epoch, gen_loss, critic_loss, high_images, low_images, fake_images)


def train(args):
    generator = Generator()
    discriminator = Discriminator()
    model_structure(generator, (3, 256, 256))
    model_structure(discriminator, (3, 256, 256))
    trainer = StandardGANTrainer(args, generator, discriminator)
    trainer.train()


def train_WGAN(args):
    generator = Generator()
    critic = Discriminator()
    model_structure(generator, (3, 256, 256))
    model_structure(critic, (3, 256, 256))
    trainer = WGAN_GPTrainer(args, generator, critic)
    trainer.train()


class BasePredictor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')
        if args.device == 'cuda':
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = args.data
        self.model = args.model
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.save_path = args.save_path
        self.generator = Generator(1, 1)
        model_structure(
            self.generator, (3, 256, 256))
        checkpoint = torch.load(self.model)
        self.generator.load_state_dict(checkpoint['net'])
        self.generator.to(self.device)
        self.generator.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def predict_images(self):
        raise NotImplementedError

    def predict_video(self):
        raise NotImplementedError


class ImagePredictor(BasePredictor):
    def __init__(self, args):
        super().__init__(args)
        self.test_data = LowLightDataset(
            image_dir=self.data, transform=self.transform, phase="test")
        self.test_loader = DataLoader(self.test_data,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      drop_last=True)

    def predict_images(self):
        # 防止同名覆盖
        path = save_path(self.save_path, model='predict')
        img_pil = transforms.ToPILImage()
        pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader), bar_format='{l_bar}{bar:10}| {n_fmt}/{'
                    'total_fmt} {elapsed}')
        torch.no_grad()
        i = 0
        if not os.path.exists(os.path.join(path, 'predictions')):
            os.makedirs(os.path.join(path, 'predictions'))
        for i, (low_images, high_images) in pbar:
            lamb = 255.  # 取绝对值最大值，避免负数超出索引
            low_images = low_images.to(self.device) / lamb
            high_images = high_images.to(self.device) / lamb

            fake = self.generator(low_images)
            for j in range(self.batch_size):
                fake_img = np.array(
                    img_pil(fake[j]), dtype=np.float32)

                if i > 10 and i % 10 == 0:  # 图片太多，十轮保存一次
                    img_save_path = os.path.join(
                        path, 'predictions', str(i) + '.jpg')
                    cv2.imwrite(img_save_path, fake_img)
                i = i + 1
            pbar.set_description('Processed %d images' % i)
        pbar.close()

    def predict_video(self):
        raise NotImplementedError  # 该类不支持视频预测


class VideoPredictor(BasePredictor):
    def __init__(self, args):
        super().__init__(args)
        self.video = args.video
        self.save_video = args.save_video

    def predict_images(self):
        raise NotImplementedError  # 该类不支持图片预测

    def predict_video(self):
        print('running on the device: ', self.device)

        try:
            # 使用 OpenCV 打开视频，避免 imageio 的迭代问题
            cap = cv2.VideoCapture(self.data)
            if not cap.isOpened():
                print(f"Error: Could not open video {self.data}")
                return

            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(
                f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        except Exception as e:
            print(f"Error opening video: {e}")
            return  # 退出函数

        # 设置视频写入器
        writer = None
        if self.save_video:
            output_path = os.path.join(self.save_path, 'fake.mp4')
            try:
                # 使用OpenCV的VideoWriter
                # 使用整数形式的fourcc代码，避免使用VideoWriter_fourcc
                fourcc = int(cv2.VideoWriter.fourcc(
                    'M', 'P', '4', 'V'))  # MP4V编码
                writer = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

                if not writer.isOpened():
                    print("Error: Could not create video writer")
                    writer = None
            except Exception as e:
                print(f"Error creating video writer: {e}")
                writer = None

        # 创建保存目录
        if not os.path.exists(os.path.join(self.save_path, 'predictions')):
            os.makedirs(os.path.join(self.save_path, 'predictions'))

        # 使用上下文管理器，确保 no_grad 状态正确管理
        with torch.no_grad():
            frame_count = 0
            pbar = tqdm(total=total_frames, desc="Processing video frames")

            try:
                while True:
                    # 读取一帧
                    ret, frame = cap.read()
                    if not ret:
                        break  # 视频结束

                    # 视频帧处理
                    frame_resized = cv2.resize(frame, (640, 480))
                    frame_pil = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    frame_tensor = torch.tensor(np.array(
                        frame_pil, np.float32) / 255., dtype=torch.float32).to(self.device)
                    frame_tensor = frame_tensor.permute(
                        2, 0, 1).unsqueeze(0)  # [1, 3, 480, 640]

                    # 生成增强图像
                    fake = self.generator(frame_tensor)
                    fake_np = fake.squeeze(0).permute(
                        1, 2, 0).cpu().detach().numpy()

                    # 转换为显示格式
                    fake_display = (np.clip(fake_np, 0, 1)
                                    * 255).astype(np.uint8)
                    fake_display_bgr = cv2.cvtColor(
                        fake_display, cv2.COLOR_RGB2BGR)

                    # 显示帧
                    cv2.imshow('Enhanced', fake_display_bgr)
                    cv2.imshow('Original', frame_resized)

                    # 保存视频
                    if self.save_video and writer is not None:
                        writer.write(fake_display_bgr)

                    # 每10帧保存一张图片
                    if frame_count % 10 == 0:
                        img_save_path = os.path.join(
                            self.save_path, 'predictions', f'frame_{frame_count:04d}.jpg')
                        cv2.imwrite(img_save_path, fake_display_bgr)

                    # 检查按键
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC键
                        break

                    frame_count += 1
                    pbar.update(1)

            except Exception as e:
                print(f"Error processing video: {e}")

            finally:
                # 释放资源
                pbar.close()
                cap.release()
                if writer is not None:
                    writer.release()
                cv2.destroyAllWindows()
                print(f"Processed {frame_count} frames")


def predict(args):
    """预测函数"""
    if args.mode == 'image':
        predictor = ImagePredictor(args)
        predictor.predict_images()
    elif args.mode == 'video':
        predictor = VideoPredictor(args)
        predictor.predict_video()
    else:
        print("不支持的预测模式")

    print("预测完成")


def compute_gradient_penalty(critic, real_samples, fake_samples, lambda_gp=10.0):
    """计算梯度惩罚 - 改进版本，增加数值稳定性"""
    # 确保输入没有NaN
    if torch.isnan(real_samples).any() or torch.isnan(fake_samples).any():
        # 如果输入包含NaN，返回零惩罚
        return torch.tensor(0.0, device=real_samples.device, requires_grad=True)

    batch_size = real_samples.size(0)

    # 生成随机插值系数
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_samples.device)

    # 创建插值样本
    interpolates = real_samples + alpha * (fake_samples - real_samples)
    interpolates.requires_grad_(True)

    # 计算评论器对插值样本的输出
    critic_interpolates = critic(interpolates)

    # 创建梯度输出
    grad_outputs = torch.ones_like(
        critic_interpolates, device=real_samples.device)

    # 计算梯度
    try:
        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
    except RuntimeError:
        # 如果梯度计算失败，返回零惩罚
        print("警告: 梯度计算失败，返回零惩罚")
        return torch.tensor(0.0, device=real_samples.device, requires_grad=True)

    # 检查梯度是否包含NaN
    if torch.isnan(gradients).any():
        print("警告: 梯度包含NaN，返回零惩罚")
        return torch.tensor(0.0, device=real_samples.device, requires_grad=True)

    # 展平梯度并计算范数
    gradients = gradients.view(batch_size, -1)

    # 添加小的epsilon值以防止除零错误
    epsilon = 1e-10
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + epsilon)

    # 计算惩罚 - 使用clamp防止极端值
    gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)
    gradient_penalty = torch.clamp(gradient_penalty, 0.0, 1000.0)  # 限制惩罚范围

    return gradient_penalty
