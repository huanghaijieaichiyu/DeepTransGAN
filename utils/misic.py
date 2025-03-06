import os
import random
from copy import copy
from math import exp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from timm.optim import Lion, RMSpropTF
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torchvision import transforms
from rich.console import Console
from rich.table import Table

from datasets.data_set import LowLightDataset
from utils.loss import BCEBlurWithLogitsLoss, FocalLoss


def set_random_seed(seed: int = 10, deterministic: bool = False, benchmark: bool = False) -> int:
    """
    Sets the random seed for Python's `random`, NumPy, and PyTorch (CPU and CUDA).
    Optionally enables deterministic behavior and benchmarking in CUDA.

    Args:
        seed (int): The random seed to set. Defaults to 10.
        deterministic (bool): Whether to enforce deterministic behavior in CUDA.
                              Reduces performance but ensures reproducibility. Defaults to False.
        benchmark (bool): Whether to enable CUDA benchmarking.
                          Improves performance by finding the fastest algorithms, 
                          but can introduce variability. Defaults to False.

    Returns:
        int: The random seed that was set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA seed setting
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        if deterministic:
            torch.backends.cudnn.deterministic = True
        else:
            # Explicitly disable for non-deterministic mode
            torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark

    return seed


def get_opt(args, generator, discriminator):
    """
    Initializes and returns the optimizers for the generator and discriminator based on the specified optimizer type.

    Args:
        args: An object containing the configuration parameters, including optimizer type, learning rate, and beta values.
        generator (torch.nn.Module): The generator model.
        discriminator (torch.nn.Module): The discriminator model.

    Returns:
        tuple: A tuple containing the discriminator optimizer and the generator optimizer.
    """
    optimizer_type = args.optimizer.lower()  # 转换为小写，忽略大小写

    if optimizer_type == 'adamw':
        g_optimizer = AdamW(
            params=generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        d_optimizer = AdamW(
            params=discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    elif optimizer_type == 'adam':
        g_optimizer = Adam(
            params=generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        d_optimizer = Adam(
            params=discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    elif optimizer_type == 'sgd':
        g_optimizer = SGD(
            params=generator.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        d_optimizer = SGD(
            params=discriminator.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    elif optimizer_type == 'lion':
        # 确保 Lion 已导入
        g_optimizer = Lion(params=generator.parameters(),
                           lr=args.lr, betas=(args.b1, args.b2))
        d_optimizer = Lion(params=discriminator.parameters(),
                           lr=args.lr, betas=(args.b1, args.b2))
    elif optimizer_type == 'rmp':
        # 确保 RMSpropTF 已导入
        g_optimizer = RMSpropTF(params=generator.parameters(), lr=args.lr)
        d_optimizer = RMSpropTF(params=discriminator.parameters(), lr=args.lr)
    else:
        raise ValueError(f'No such optimizer: {args.optimizer}')

    return d_optimizer, g_optimizer


def get_loss(loss_name: str) -> nn.Module:
    """
    Returns a loss function based on the specified loss name.

    Args:
        loss_name (str): The name of the loss function to retrieve.

    Returns:
        nn.Module: The loss function.

    Raises:
        NotImplementedError: If the specified loss function is not supported.
    """
    loss_name = loss_name.lower()  # 转换为小写，忽略大小写

    if loss_name == 'bceblurwithlogitsloss':
        loss = BCEBlurWithLogitsLoss()
    elif loss_name == 'mse':
        loss = nn.MSELoss()
    elif loss_name == 'focalloss':
        # 确保 FocalLoss 已导入
        loss = FocalLoss(nn.BCEWithLogitsLoss())
    elif loss_name == 'bce':
        loss = nn.BCEWithLogitsLoss()
    else:
        print(f'No such loss function: {loss_name}')
        raise NotImplementedError

    return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def save_path(path: str, model: str = 'train') -> str:
    """
    Generates a unique file path for saving models, incrementing a counter
    if the path already exists. The directory is not created by this function.

    Args:
        path (str): The base path for saving the model.
        model (str): The base name of the model. Defaults to 'train'.

    Returns:
        str: A unique file path that does not currently exist.
    """
    file_path = os.path.join(path, model)
    i = 1
    while os.path.exists(file_path):
        file_path = os.path.join(path, f'{model}({i})')
        i += 1

    return file_path


def model_structure(model, img_size: tuple[int, int, int]) -> tuple[float, float]:
    """
    打印PyTorch模型的详细结构，包括层名称、形状、参数数量和总模型大小。
    使用ptflops计算FLOPs，使用rich库美化输出。

    Args:
        model (nn.Module): 要分析的PyTorch模型。
        img_size (tuple[int, int, int]): 输入图像大小 (C, H, W)。

    Returns:
        tuple[float, float]: 包含参数数量(百万)和GFLOPs的元组。
    """
    try:
        from ptflops import get_model_complexity_info
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print("请安装必要的库: pip install ptflops rich")
        return 0, 0

    console = Console()

    # 创建表格
    table = Table(title=f"模型结构分析: {model.__class__.__name__}")
    table.add_column("层名称", style="cyan")
    table.add_column("参数形状", style="green")
    table.add_column("参数数量", justify="right", style="yellow")

    # 计算每层参数
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            # 简化名称显示
            name = name.replace("module.", "")
            shape_str = str(list(param.shape))
            table.add_row(name, shape_str, f"{param_count:,}")

    # 使用ptflops计算FLOPs
    macs, params = get_model_complexity_info(
        model,
        (img_size[0], img_size[1], img_size[2]),
        as_strings=False,
        print_per_layer_stat=False
    )

    # 确保macs是数值类型
    macs_value = float(macs) if macs is not None else 0.0

    # 添加总结行
    table.add_section()
    table.add_row(
        "[bold]总计[/bold]",
        "",
        f"[bold]{total_params:,}[/bold]"
    )

    # 打印表格
    console.print(table)

    # 打印总结信息
    console.print(
        f"[bold]模型大小:[/bold] {total_params * 4 / (1024**2):.2f} MB (假设FP32精度)")
    console.print(f"[bold]计算量:[/bold] {macs_value * 2 * 1e-9:.2f} GFLOPs")

    return total_params * 1e-6, macs_value * 2 * 1e-9
