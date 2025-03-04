import math
import torch.nn as nn
import torch
import torch.nn.functional as F

from models.Repvit import RepViTBlock
from models.common import SPPELAN, Concat, Disconv, Gencov, PSA


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

        # Define layers
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

        self.tanh = nn.Sigmoid()
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
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # Decoder
        x6 = self.conv6(x5)
        x7 = self.conv7(self.concat([x6, x3]))
        x8 = self.conv8(self.concat([x2, x7]))
        x9 = self.tanh(self.conv9(x8))

        return x9.view(-1, 3, x.shape[2], x.shape[3])


class Discriminator(BaseNetwork):
    """
    Discriminator model with no activation function
    """

    def __init__(self, batch_size=8, img_size=256):
        super().__init__()
        self.batch_size = batch_size
        self.ratio = img_size / 256.  # Store ratio as an instance variable

        # Define layers
        self.conv_in = nn.Sequential(Disconv(3, 8), RepViTBlock(8, 16, 3, 2))
        self.conv1 = self._make_sequential(16, 32, 3)
        self.conv_out = Disconv(4, 1, bn=False, act=False)
        self.act = nn.Identity()  # BCEWithLogitLOSS 在输入前已经sigmoid，无需再限制输出

    def _make_sequential(self, in_channels, out_channels, weight):
        """
        Creates a sequential block for the discriminator.
        """
        return nn.Sequential(
            RepViTBlock(in_channels, out_channels, weight, 2),
            Disconv(out_channels, out_channels * 2),  # Increased out_channels
            RepViTBlock(out_channels * 2, in_channels, weight, 2),
            Disconv(in_channels, in_channels // 2),  # Reduced out_channels
            RepViTBlock(in_channels // 2, in_channels // 4, weight, 2),
            Disconv(in_channels // 4, 4)
        )

    def forward(self, x):
        x = self.act(self.conv_out(self.conv1(self.conv_in(x)))).view(
            self.batch_size if x.shape[0] == self.batch_size else x.shape[0], -1)
        return x


class Critic(BaseNetwork):
    """
    Critic model for WGAN-GP
    """

    def __init__(self, batch_size=8, img_size=256):
        super().__init__()
        self.batch_size = batch_size
        self.ratio = img_size / 256.  # Store ratio as an instance variable

        # 定义层
        self.conv_in = nn.Sequential(
            Disconv(3, 8, bn=False, act=True),  # WGAN 通常避免输入层的 BN
            RepViTBlock(8, 16, 3, 2)
        )
        self.conv1 = self._make_sequential(16, 32, 3)
        self.conv_out = Disconv(4, 1, bn=False, act=False)  # 输出实数值，无激活

        # 初始化权重
        self.apply(self.init_weights)

    def _make_sequential(self, in_channels, out_channels, weight):
        """
        Creates a sequential block for the discriminator.
        """
        return nn.Sequential(
            RepViTBlock(in_channels, out_channels, weight, 2),
            Disconv(out_channels, out_channels * 2),
            RepViTBlock(out_channels * 2, in_channels, weight, 2),
            Disconv(in_channels, in_channels // 2),
            RepViTBlock(in_channels // 2, in_channels // 4, weight, 2),
            Disconv(in_channels // 4, 4)
        )

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv1(x)
        x = self.conv_out(x)
        # 调整输出形状
        batch_size = x.shape[0] if x.shape[0] != self.batch_size else self.batch_size
        x = x.view(batch_size, -1)
        return x.mean(dim=1, keepdim=True)  # 输出实数值

# GAN成器


class CNNTransformerGenerator(nn.Module):
    def __init__(self, depth=1, weight=1):
        super().__init__()
        self.depth = depth
        self.weight = weight

        # CNN 编码器部分
        self.conv1 = nn.Conv2d(3, math.ceil(
            8 * self.depth), kernel_size=3, stride=1, padding=1)
        self.conv2 = self._make_cnn_block(
            math.ceil(8 * self.depth), math.ceil(16 * self.depth))
        self.conv3 = self._make_cnn_block(
            math.ceil(16 * self.depth), math.ceil(32 * self.depth))
        self.conv4 = self._make_cnn_block(
            math.ceil(32 * self.depth), math.ceil(64 * self.depth))

        # Transformer 部分
        transformer_layers = nn.TransformerEncoderLayer(
            d_model=math.ceil(64 * self.depth),
            nhead=8,
            dim_feedforward=math.ceil(128 * self.depth),
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layers, num_layers=2)

        # CNN 解码器部分
        self.conv6 = nn.Sequential(
            nn.Conv2d(math.ceil(64 * self.depth), math.ceil(64 *
                      self.depth), kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(math.ceil(96 * self.depth), math.ceil(96 *
                      self.depth), kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(math.ceil(112 * self.depth),
                      math.ceil(112 * self.depth), kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv9 = nn.Conv2d(math.ceil(112 * self.depth),
                               3, kernel_size=3, padding=1)

        self.tanh = nn.Sigmoid()
        self.concat = lambda x: torch.cat(x, dim=1)  # 简单实现 Concat

    def _make_cnn_block(self, in_channels, out_channels):
        """创建 CNN 块：卷积 + BN + ReLU + 下采样"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        # 编码器 (CNN)
        x1 = self.conv1(x)  # (B, 8*d, H, W)
        x2 = self.conv2(x1)  # (B, 16*d, H/2, W/2)
        x3 = self.conv3(x2)  # (B, 32*d, H/4, W/4)
        x4 = self.conv4(x3)  # (B, 64*d, H/8, W/8)

        # Transformer 处理
        B, C, H, W = x4.shape
        x4_flat = x4.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, 64*d)
        x5 = self.transformer(x4_flat)  # (B, H*W, 64*d)
        x5 = x5.permute(0, 2, 1).view(B, C, H, W)  # (B, 64*d, H/8, W/8)

        # 解码器 (CNN + 上采样)
        x6 = self.conv6(x5)  # (B, 64*d, H/4, W/4)
        x7 = self.conv7(self.concat([x6, x3]))  # (B, 96*d, H/2, W/2)
        x8 = self.conv8(self.concat([x7, x2]))  # (B, 112*d, H, W)
        x9 = self.tanh(self.conv9(x8))  # (B, 3, H, W)

        return x9


class DualPrunedSelfAttn(nn.Module):
    def __init__(self, dim, dim_head, heads, height_top_k, width_top_k, dropout=0.):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.height_top_k = height_top_k
        self.width_top_k = width_top_k

    def forward(self, x):
        B, N, C = x.size()
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.heads,
                      self.dim_head).transpose(1, 2), qkv)
        attn = torch.einsum('b h i d, b h j d -> b h i j',
                            q, k) * (self.dim_head ** -0.5)
        attn = self.prune_attn(attn, self.height_top_k, self.width_top_k)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.to_out(out)
        return out

    def prune_attn(self, attn, height_top_k, width_top_k):
        B, H, N, _ = attn.shape
        device = attn.device
        row_scores = attn.mean(dim=-1)
        _, top_row_idx = row_scores.topk(height_top_k, dim=-1)
        row_mask = torch.zeros(B, H, N, device=device)
        row_mask.scatter_(-1, top_row_idx, 1)
        row_mask = row_mask.unsqueeze(-1).expand(B, H, N, N)
        col_scores = attn.mean(dim=-2)
        _, top_col_idx = col_scores.topk(width_top_k, dim=-1)
        col_mask = torch.zeros(B, H, N, device=device)
        col_mask.scatter_(-1, top_col_idx, 1)
        col_mask = col_mask.unsqueeze(-2).expand(B, H, N, N)
        mask = row_mask * col_mask
        return attn * mask

# Hybrid Perception Block


class HybridPerceptionBlock(nn.Module):
    def __init__(self, dim, dim_head, heads, attn_height_top_k, attn_width_top_k, attn_dropout, ff_mult, ff_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DualPrunedSelfAttn(
            dim, dim_head, heads, attn_height_top_k, attn_width_top_k, dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ff = nn.ModuleList([
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),  # 使用 GELU 避免梯度消失
            nn.Dropout(ff_dropout),
            nn.Linear(dim * ff_mult, dim),
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        B, C, H, W = x.size()
        x_attn = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x_attn = self.norm1(x_attn)
        x_attn = self.attn(x_attn)
        x_attn = x_attn.view(B, H, W, C).permute(0, 3, 1, 2)
        x_conv = self.conv(x)
        x = x_attn + x_conv
        normed = self.norm2(x.permute(0, 2, 3, 1))
        normed = normed.contiguous().view(B * H * W, C)
        ff_out = self.ff[0](normed)
        ff_out = self.ff[1](ff_out)
        ff_out = self.ff[2](ff_out)
        ff_out = self.ff[3](ff_out)
        ff_out = ff_out.view(B, H, W, C).permute(0, 3, 1, 2)
        x = x + ff_out
        return x

# 更新后的生成器


class SWTformerGenerator(nn.Module):
    def __init__(self, depth=1, weight=1):
        super().__init__()
        self.depth = depth
        self.weight = weight

        # 编码器
        self.conv1 = nn.Conv2d(
            3, int(math.ceil(8 * self.depth)), kernel_size=3, stride=1, padding=1)
        self.conv2 = self._make_cnn_block(
            int(math.ceil(8 * self.depth)), int(math.ceil(16 * self.depth)))
        self.conv3 = self._make_cnn_block(
            int(math.ceil(16 * self.depth)), int(math.ceil(32 * self.depth)))
        self.conv4 = self._make_cnn_block(
            int(math.ceil(32 * self.depth)), int(math.ceil(64 * self.depth)))

        # Transformer部分
        dim = int(math.ceil(64 * self.depth))
        dim_head = dim // 8
        self.hpb1 = HybridPerceptionBlock(dim, dim_head, heads=8, attn_height_top_k=64,  # 增大到 64
                                          attn_width_top_k=64, attn_dropout=0.1, ff_mult=2, ff_dropout=0.1)
        self.hpb2 = HybridPerceptionBlock(dim, dim_head, heads=8, attn_height_top_k=64,
                                          attn_width_top_k=64, attn_dropout=0.1, ff_mult=2, ff_dropout=0.1)

        # 解码器
        self.conv6 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv9 = nn.Conv2d(dim, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Identity()  # 输出映射到 (0, 1)

        # 投影层
        self.proj2 = nn.Conv2d(int(math.ceil(
            8 * self.depth)), int(math.ceil(16 * self.depth)), kernel_size=1, stride=2, padding=0)
        self.proj3 = nn.Conv2d(int(math.ceil(
            16 * self.depth)), int(math.ceil(32 * self.depth)), kernel_size=1, stride=2, padding=0)
        self.proj4 = nn.Conv2d(int(math.ceil(
            32 * self.depth)), int(math.ceil(64 * self.depth)), kernel_size=1, stride=2, padding=0)

        # 初始化权重
        self.apply(self.init_weights)

    def _make_cnn_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = x2 + 0.1 * self.proj2(x1)  # 添加残差缩放
        del x1

        x3 = self.conv3(x2)
        x3 = x3 + 0.1 * self.proj3(x2)
        del x2

        x4 = self.conv4(x3)
        x4 = x4 + 0.1 * self.proj4(x3)
        del x3

        x5 = self.hpb1(x4) + 0.1 * x4
        del x4
        x5 = self.hpb2(x5) + 0.1 * x5

        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x7 = x7 + 0.1 * \
            F.interpolate(x6, size=x7.shape[2:],
                          mode='bilinear', align_corners=True)
        del x6
        x8 = self.conv8(x7)
        x8 = x8 + 0.1 * \
            F.interpolate(x7, size=x8.shape[2:],
                          mode='bilinear', align_corners=True)
        del x7
        x9 = self.sigmoid(self.conv9(x8))  # 输出 (0, 1)
        del x8

        return x9
