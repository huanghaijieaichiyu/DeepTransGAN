import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.Repvit import RepViTBlock
from models.common import EMA, SPPELAN, C2f, C2fCIB, Concat, Disconv, Gencov, PSA, Conv, CBAM, ADown, DilateBlock


class BaseNetwork(nn.Module):
    """
    Abstract base class for Generator, Discriminator, and Critic.
    Provides common functionality like parameter initialization.
    """

    def __init__(self):
        super().__init__()
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Generator(BaseNetwork):
    def __init__(self, depth=1, weight=1):
        super().__init__()
        self.depth = depth
        self.weight = weight

        # 定义层
        self.conv1 = Gencov(3, math.ceil(8 * self.depth))
        self.conv2 = self._make_sequential(
            math.ceil(8 * self.depth), math.ceil(16 * self.depth), self.weight)
        self.conv3 = self._make_sequential(
            math.ceil(16 * self.depth), math.ceil(32 * self.depth), self.weight)
        self.conv4 = self._make_sequential(
            math.ceil(32 * self.depth), math.ceil(64 * self.depth), self.weight)
        self.conv5 = nn.Sequential(
            SPPELAN(math.ceil(64 * self.depth), math.ceil(64 *
                    self.depth), math.ceil(32 * self.depth)),
            PSA(math.ceil(64 * self.depth), math.ceil(64 * self.depth)),
            RepViTBlock(math.ceil(64 * self.depth),
                        math.ceil(64 * self.depth), 1, 1, 0, 0)
        )
        self.conv6 = nn.Sequential(
            RepViTBlock(math.ceil(64 * self.depth), math.ceil(64 *
                        self.depth), math.ceil(3 * self.weight), 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv7 = nn.Sequential(
            RepViTBlock(math.ceil(96 * self.depth), math.ceil(96 *
                        self.depth), math.ceil(self.weight), 1, 0, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv8 = nn.Sequential(
            RepViTBlock(math.ceil(112 * self.depth),
                        math.ceil(112 * self.depth), 1, 1, 0, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.conv9 = Gencov(math.ceil(112 * self.depth), 3,
                            math.ceil(self.weight), act=False, bn=False)

        self.tanh = nn.Sigmoid()  # 使用 Sigmoid 替代 Tanh，以输出 [0, 1]
        self.concat = Concat()

    def _make_sequential(self, in_channels, out_channels, weight):
        """
        Creates a sequential block with RepViTBlocks.
        """
        return nn.Sequential(
            RepViTBlock(in_channels, out_channels, math.ceil(weight), 2),
            RepViTBlock(out_channels, out_channels, 1, 1, 0, 0)
        )

    def forward(self, x):
        # 检查输入尺寸
        assert x.shape[2:] == torch.Size(
            [256, 256]), "输入尺寸必须为 [B, 3, 256, 256]"

        # 编码器
        x1 = self.conv1(x)  # [B, 8*d, 256, 256]
        x2 = self.conv2(x1)  # [B, 16*d, 128, 128]
        x3 = self.conv3(x2)  # [B, 32*d, 64, 64]
        x4 = self.conv4(x3)  # [B, 64*d, 32, 32]
        x5 = self.conv5(x4)  # [B, 64*d, 32, 32]

        # 解码器
        x6 = self.conv6(x5)  # [B, 64*d, 64, 64]
        # [B, 96*d, 128, 128] (64*d + 32*d)
        x7 = self.conv7(self.concat([x6, x3]))
        # [B, 112*d, 256, 256] (16*d + 96*d)
        x8 = self.conv8(self.concat([x2, x7]))
        x9 = self.tanh(self.conv9(x8))  # [B, 3, 256, 256]

        return x9


class AdvancedGenerator(BaseNetwork):
    """
    An advanced generator network with dilated attention and multi-scale feature fusion.
    Input: [B, 3, 256, 256]
    Output: [B, 3, 256, 256]
    """

    def __init__(self, depth=1):
        super().__init__()
        self.depth = depth
        base_ch = 32  # Base number of channels

        # Initial feature extraction
        self.conv1 = nn.Sequential(
            Gencov(3, base_ch, k=7),
            # Use standard Conv from common.py
        )

        # Encoder blocks with dilated attention
        self.enc2 = self._make_encoder_block(base_ch, base_ch * 2)
        self.enc3 = self._make_encoder_block(base_ch * 2, base_ch * 4)
        self.enc4 = self._make_encoder_block(base_ch * 4, base_ch * 8)

        # Multi-scale feature processing
        self.msf = nn.ModuleList([
            SPPELAN(base_ch * 8, base_ch * 8, base_ch * 4)
        ])

        # PSA attention for feature enhancement
        self.psa = PSA(base_ch * 8, base_ch * 8)

        # Decoder blocks with advanced upsampling
        self.dec3 = self._make_decoder_block(
            base_ch * 16, base_ch * 4)  # Doubled input channels for skip
        self.dec2 = self._make_decoder_block(
            base_ch * 8, base_ch * 2)   # Doubled input channels for skip
        self.dec1 = self._make_decoder_block(base_ch * 4, base_ch)

        # Final refinement
        self.refine = nn.Sequential(
            Gencov(base_ch * 2, base_ch, k=3),
            Gencov(base_ch, 3, k=7, bn=False, act=False),
            nn.Sigmoid()
        )

    def _make_encoder_block(self, in_ch, out_ch):
        """Enhanced encoder block with dilated convolutions and attention"""
        return nn.Sequential(
            Gencov(in_ch, out_ch * 2, k=3, s=2),
            C2f(out_ch * 2, out_ch)  # Channel reduction
        )

    def _make_decoder_block(self, in_ch, out_ch):
        """Enhanced decoder block with attention and residual connection"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            C2fCIB(in_ch, out_ch)
        )

    def _init_weights(self, m):
        """Initialize weights for Conv2d, Linear, and BatchNorm2d layers"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input validation
        assert x.shape[2:] == torch.Size(
            [256, 256]), "Input size must be [B, 3, 256, 256]"

        # Initial feature extraction
        x1 = self.conv1(x)      # [B, 32, 256, 256]

        # Encoder path with progressive feature learning
        x2 = self.enc2(x1)     # [B, 64, 128, 128]
        x3 = self.enc3(x2)     # [B, 128, 64, 64]
        x4 = self.enc4(x3)     # [B, 256, 32, 32]

        # Multi-scale feature processing
        feat = x4
        for msf_block in self.msf:
            feat = msf_block(feat)

        # Feature enhancement with PSA
        feat = self.psa(feat)

        # Decoder path with skip connections and feature fusion
        d3 = self.dec3(torch.cat([feat, x4], 1))   # [B, 128, 64, 64]
        d2 = self.dec2(torch.cat([d3, x3], 1))     # [B, 64, 128, 128]
        d1 = self.dec1(torch.cat([d2, x2], 1))     # [B, 32, 256, 256]

        # Final refinement with original features
        out = self.refine(torch.cat([d1, x1], 1))  # [B, 3, 256, 256]

        return out


class Discriminator(BaseNetwork):
    """
    Discriminator model with no activation function
    """

    def __init__(self, depth=1, weight=1):
        super().__init__()
        self.depth = depth
        self.weight = weight

        # 简化后的判别器层
        # 调整 kernel_size 和 stride
        self.conv1 = Disconv(3, 8 * self.depth, 4, 2, 1)
        self.conv2 = self._make_sequential(
            8 * self.depth, 16 * self.depth)  # 简化 _make_sequential
        self.conv3 = self._make_sequential(16 * self.depth, 32 * self.depth)
        self.conv4 = nn.Sequential(  # 进一步简化 conv4 和 conv5 部分
            Disconv(32 * self.depth, 64 * self.depth, 4, 2, 1),
            Disconv(64 * self.depth, 64 * self.depth, 4, 2, 1),
        )

        # 最终判别层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * self.depth, 1)
        self.sigmoid = nn.Identity()  # 使用 Identity，配合 BCEWithLogitsLoss

    def _make_sequential(self, in_channels, out_channels):  # 简化 _make_sequential
        """
        创建更基础的卷积序列模块。
        """
        return nn.Sequential(
            # 调整 kernel_size 和 stride
            Disconv(in_channels, out_channels, 4, 2, 1),
            Disconv(out_channels, out_channels, 1, 1, 1),
        )

    def forward(self, x):
        # 判别器前向传播
        x = self.conv1(x)   # [B, 8*d, 128, 128]
        x = self.conv2(x)   # [B, 16*d, 64, 64]
        x = self.conv3(x)   # [B, 32*d, 32, 32]
        x = self.conv4(x)   # [B, 64*d, 8, 8]  (简化后下采样更快)
        x = self.avgpool(x)  # [B, 64*d, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)      # [B, 1]
        x = self.sigmoid(x)
        return x
