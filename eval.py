'''
code by 黄小海 2025.3.3


'''

import argparse
from ast import Is
from cv2 import transform
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from scipy import linalg
import numpy as np
import lpips
from torchmetrics.image.kid import KernelInceptionDistance
from utils.misic import ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from datasets.data_set import LowLightDataset
from tqdm import tqdm

from models.base_mode import Generator


class Evaluator:
    def __init__(self, model, device='cuda'):
        """初始化 Evaluator 类，加载预训练模型并设置设备"""
        self.model = model
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        # 加载预训练的 Inception V3 模型
        self.inception_model = self.model.to(self.device)
        self.inception_model.eval()
        self.inception_model.fc = nn.Identity()  # 移除分类层
        # 用于调整图像到 Inception V3 输入尺寸 (256, 256)
        self.up = nn.Upsample(size=(256, 256), mode='bilinear',
                              align_corners=False).to(self.device)
        # 加载 LPIPS 模型
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)

    def inception_score(self, generated_images, batch_size=16, splits=10):
        """计算 Inception Score"""
        preds = self._get_inception_features(generated_images, batch_size)
        # Softmax 操作
        preds = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
        # 计算 KL 散度
        kl_div = preds * (np.log(preds) - np.log(np.mean(preds, axis=0)))
        kl_div = np.sum(kl_div, axis=1)
        # 分割计算得分
        split_scores = []
        for k in range(splits):
            part = kl_div[k * (len(generated_images) // splits): (k + 1) * (len(generated_images) // splits)]
            split_scores.append(np.exp(np.mean(part)))
        return np.mean(split_scores), np.std(split_scores)

    def fid(self, real_images, generated_images, batch_size=16):
        """计算 Fréchet Inception Distance (FID)"""
        real_features = self._get_inception_features(real_images, batch_size)
        gen_features = self._get_inception_features(
            generated_images, batch_size)
        # 计算均值和协方差
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(
            real_features, rowvar=False)
        mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(
            gen_features, rowvar=False)
        # 计算 FID
        diff = mu_real - mu_gen
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
        return fid

    def lpips(self, real_images, generated_images, batch_size=16):
        """计算 Learned Perceptual Image Patch Similarity (LPIPS)"""
        assert len(real_images) == len(generated_images), "两组图像数量必须相同"
        distances = []
        for i in range(0, len(real_images), batch_size):
            real_batch = real_images[i:i + batch_size].to(self.device)
            gen_batch = generated_images[i:i + batch_size].to(self.device)
            with torch.no_grad():
                dist = self.lpips_model(real_batch, gen_batch)
            distances.append(dist.cpu().numpy())
        return np.mean(np.concatenate(distances))

    def kid(self, real_images, generated_images, subset_size=50):
        """计算 Kernel Inception Distance (KID)"""
        kid_metric = KernelInceptionDistance(
            subset_size=subset_size).to(self.device)
        kid_metric.update(real_images, real=True)
        kid_metric.update(generated_images, real=False)
        return kid_metric.compute()

    def ssim(self, real_images, generated_images):
        """计算 Structural Similarity Index (SSIM)"""
        assert len(real_images) == len(generated_images), "两组图像数量必须相同"
        scores = []

        score = ssim(real_images, generated_images)
        scores.append(score)
        return np.mean(scores)

    def psnr(self, real_images, generated_images):
        """计算 Peak Signal-to-Noise Ratio (PSNR)"""
        assert len(real_images) == len(generated_images), "两组图像数量必须相同"
        scores = []
        for real, gen in zip(real_images, generated_images):
            real_np = real.cpu().numpy().transpose(1, 2, 0)
            gen_np = gen.cpu().numpy().transpose(1, 2, 0)
            score = psnr(real_np, gen_np,
                         data_range=gen_np.max() - gen_np.min())
            scores.append(score)
        return np.mean(scores)

    def _get_inception_features(self, images, batch_size=16):
        """获取 Inception V3 特征"""
        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(self.device)
            batch = self.up(batch)  # 调整尺寸到 (299, 299)
            with torch.no_grad():
                feat = self.inception_model(batch)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)


def main(args):
    model = Generator(1, 1).to('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model.load_state_dict(torch.load(args.model)['net'])
    except:
        print('loading model from {} has something wrong!'.format(args.model))
    evaluator = Evaluator(model,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
    # 假设 real_images 和 generated_images 是形状为 (N, C, H, W) 的张量列表

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size[0], args.img_size[1])),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_data = LowLightDataset(
        image_dir=args.data, transform=transform, phase="test")
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, drop_last=False)
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), bar_format='{l_bar}{bar:10}| {n_fmt}/{'
                'total_fmt} {elapsed}')

    Is = []
    FId = []
    LPIPs = []
    ssim = []
    psnr = []
    torch.no_grad()
    model.eval()
    for i, (low_images, high_images) in pbar:
        real_images = high_images
        generated_images = low_images
    # 创建 Evaluator 实例

        # 计算各种评估指标
        is_mean, is_std = evaluator.inception_score(generated_images)
        # fid_score = evaluator.fid(real_images, generated_images)
        lpips_score = evaluator.lpips(real_images, generated_images)
        ssim_score = evaluator.ssim(real_images, generated_images)
        psnr_score = evaluator.psnr(real_images, generated_images)
        Is.append(is_mean)
        # FId.append(fid_score)
        LPIPs.append(lpips_score)
        ssim.append(ssim_score)
        psnr.append(psnr_score)
    print(f"Inception Score: {np.mean(Is):.4f} ± {np.mean(is_std):.4f}")
    # print(f"FID: {np.mean(fid_score):.4f}")
    print(f"LPIPS: {np.mean(lpips_score):.4f}")
    # kid_score = evaluator.kid(real_images, generated_images)  # 需要 torchmetrics 支持
    # print(f"KID: {kid_score}")
    print(f"SSIM: {np.mean(ssim_score):.4f}")
    print(f"PSNR: {np.mean(psnr_score):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 命令行选项、参数和子命令解析器
    parser.add_argument("--data", type=str,
                        default='../datasets/kitti_LOL', help="path to dataset")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="size of the batches")  # batch大小
    parser.add_argument("--img_size", type=tuple,
                        default=(256, 256), help="size of the image")

    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of data loading workers, if in windows, must be 0"
                        )
    parser.add_argument("--model", type=str, default="runs/train(1)/generator/last.pt",
                        help="train or test model")
    arges = parser.parse_args()

    main(arges)
