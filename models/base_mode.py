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

# 改进的Hybrid Perception Block，添加了更多注意力机制和更好的残差连接


class EnhancedHybridPerceptionBlock(nn.Module):
    def __init__(self, dim, dim_head, heads, attn_height_top_k,
                 attn_width_top_k, attn_dropout, ff_mult, ff_dropout):
        super().__init__()
        # 第一个标准化和注意力块
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DualPrunedSelfAttn(
            dim, dim_head, heads, attn_height_top_k,
            attn_width_top_k, dropout=attn_dropout)

        # 第二个标准化和前馈网络
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        # 使用两层感知机代替ModuleList以提高性能
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * ff_mult, dim),
        )

        # 深度可分离卷积捕获局部信息
        self.dw_conv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1)

        # 添加空间注意力和通道注意力
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(dim // 8, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
            nn.Sigmoid()
        )

        # 伽马参数用于控制残差连接的强度
        self.gamma1 = nn.Parameter(torch.ones(1) * 0.5)
        self.gamma2 = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        # 保存输入用于残差连接
        input_x = x

        # 处理自注意力部分
        B, C, H, W = x.size()
        x_attn = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x_attn = self.norm1(x_attn)
        x_attn = self.attn(x_attn)
        x_attn = x_attn.view(B, H, W, C).permute(0, 3, 1, 2)

        # 卷积分支处理
        x_conv = self.dw_conv(x)
        x_conv = self.pw_conv(x_conv)

        # 融合自注意力和卷积分支
        x = x_attn + x_conv

        # 应用空间和通道注意力机制
        spatial_weight = self.spatial_gate(x)
        channel_weight = self.channel_gate(x)
        x = x * spatial_weight * channel_weight

        # 第一个残差连接
        x = input_x + self.gamma1 * x

        # 前馈网络处理
        identity = x
        normed = self.norm2(x.permute(0, 2, 3, 1))
        normed = normed.contiguous().view(B * H * W, C)
        ff_out = self.ff(normed)
        ff_out = ff_out.view(B, H, W, C).permute(0, 3, 1, 2)

        # 第二个残差连接
        x = identity + self.gamma2 * ff_out

        return x

# 改进的SWTformerGenerator网络


class SWTformerGenerator(nn.Module):
    def __init__(self, depth=1, weight=1):
        super().__init__()
        self.depth = depth
        self.weight = weight

        # 编码器 - 使用级联卷积结构
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(math.ceil(8 * self.depth)),
                      kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(int(math.ceil(8 * self.depth)))
        )

        # 使用更高效的下采样块
        self.conv2 = self._make_sequential(
            int(math.ceil(8 * self.depth)), int(math.ceil(16 * self.depth)))
        self.conv3 = self._make_sequential(
            int(math.ceil(16 * self.depth)), int(math.ceil(32 * self.depth)))
        self.conv4 = self._make_sequential(
            int(math.ceil(32 * self.depth)), int(math.ceil(64 * self.depth)))

        # 特征融合层
        dim = int(math.ceil(64 * self.depth))
        # 增加一个瓶颈层进行特征压缩，不增加计算量
        self.bottleneck = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(dim // 2, dim, kernel_size=1)
        )

        # Transformer部分 - 使用增强的HybridPerceptionBlock
        dim_head = dim // 8
        self.hpb1 = EnhancedHybridPerceptionBlock(
            dim, dim_head, heads=8,
            attn_height_top_k=64, attn_width_top_k=64,
            attn_dropout=0.1, ff_mult=2, ff_dropout=0.1
        )
        self.hpb2 = EnhancedHybridPerceptionBlock(
            dim, dim_head, heads=8,
            attn_height_top_k=64, attn_width_top_k=64,
            attn_dropout=0.1, ff_mult=2, ff_dropout=0.1
        )

        # 解码器 - 改进的上采样块
        self.up1 = self._make_up_block(dim, dim)
        self.up2 = self._make_up_block(dim, dim)
        self.up3 = self._make_up_block(dim, dim)

        # 细化网络 - 最终输出层
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim // 2, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # 确保输出在(0,1)范围内
        )

        # 跳跃连接投影层
        self.proj2 = nn.Conv2d(
            int(math.ceil(8 * self.depth)),
            int(math.ceil(16 * self.depth)),
            kernel_size=1, stride=2, padding=0
        )
        self.proj3 = nn.Conv2d(
            int(math.ceil(16 * self.depth)),
            int(math.ceil(32 * self.depth)),
            kernel_size=1, stride=2, padding=0
        )
        self.proj4 = nn.Conv2d(
            int(math.ceil(32 * self.depth)),
            int(math.ceil(64 * self.depth)),
            kernel_size=1, stride=2, padding=0
        )

        # 添加跳跃连接通道调整层 - 用于解决通道不匹配问题
        self.skip_proj2 = nn.Conv2d(
            int(math.ceil(16 * self.depth)),
            dim,  # 将编码器特征映射到相同的通道数
            kernel_size=1, padding=0
        )
        self.skip_proj3 = nn.Conv2d(
            int(math.ceil(32 * self.depth)),
            dim,  # 将编码器特征映射到相同的通道数
            kernel_size=1, padding=0
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _make_sequential(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # 编码器前向传播
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = x2 + self.proj2(x1)  # 添加残差缩放

        x3 = self.conv3(x2)
        x3 = x3 + self.proj3(x2)

        x4 = self.conv4(x3)
        x4 = x4 + self.proj4(x3)

        # 瓶颈层特征处理
        x4 = self.bottleneck(x4) + x4  # 残差连接

        # Transformer处理
        x5 = self.hpb1(x4)
        x5 = self.hpb2(x5)

        # 解码器前向传播 - 使用跳跃连接
        x6 = self.up1(x5)

        # 添加跳跃连接，使用插值调整尺寸并确保通道匹配
        x7 = self.up2(x6)
        size7 = (x7.shape[2], x7.shape[3])
        # 使用通道投影层解决通道不匹配问题
        skip_x3 = F.interpolate(
            self.skip_proj3(x3), size=size7, mode='bilinear', align_corners=True
        )
        x7 = x7 + self.res_scale * skip_x3

        x8 = self.up3(x7)
        size8 = (x8.shape[2], x8.shape[3])
        # 使用通道投影层解决通道不匹配问题
        skip_x2 = F.interpolate(
            self.skip_proj2(x2), size=size8, mode='bilinear', align_corners=True
        )
        x8 = x8 + self.res_scale * skip_x2

        # 最终输出
        out = self.refine(x8)

        return out
