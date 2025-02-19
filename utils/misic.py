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
    Prints a detailed structure of the PyTorch model, including layer names, shapes,
    parameter counts, and total model size. Similar layers are grouped for better readability.

    Args:
        model (nn.Module): The PyTorch model to analyze.
        img_size (tuple[int, int, int]): Input image size (C, H, W).

    Returns:
        tuple[float, float]: A tuple containing the number of parameters (in millions)
                             and the number of GFLOPs.
    """

    blank = ' '
    print('-' * 142)
    print(f'|{"Layer Name":^70}|{"Weight Shape":^45}|{"#Params":^15}|')
    print('-' * 142)

    num_para = 0
    type_size = 4  # Assuming float32
    macs, _ = get_model_complexity_info(model, img_size, as_strings=False, print_per_layer_stat=False,
                                        verbose=False)

    layer_summary = {}
    for name, param in model.named_parameters():
        shape = str(param.shape)
        each_para = param.numel()  # 更简洁的写法
        num_para += each_para
        if shape not in layer_summary:
            layer_summary[shape] = {"names": [], "count": 0, "params": 0}
        layer_summary[shape]["names"].append(name)
        layer_summary[shape]["count"] += 1
        layer_summary[shape]["params"] += each_para

    for shape, data in layer_summary.items():
        names = ", ".join(data["names"])
        if len(names) > 67:
            names = names[:64] + "..."  # 截断过长的层名称
        num_str = str(data["params"])
        print(f'|{names:<70}|{shape:<45}|{num_str:>15}|')

    print('-' * 142)
    print(f'Total Parameters: {num_para}')
    print(
        f'Model Size ({model._get_name()}): {num_para * type_size / 1e6:.2f} MB')
    # type: ignore
    print(f'GFLOPs ({model._get_name()}): {2 * macs * 1e-9:.2f} G')
    print('-' * 142)

    return num_para * 1e-6, 2 * macs * 1e-9  # type: ignore
