import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils import spectral_norm
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


# 提取图像块的工具类
class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images):
        B, C, H, W = images.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "图像尺寸必须能被块大小整除"
        # 提取图像块
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(
            B, -1, C * self.patch_size * self.patch_size)
        return patches

# 修改后的双重裁剪自注意力模块，适配 ViT


class DualPrunedSelfAttn(nn.Module):
    def __init__(self, dim, num_heads, dim_head, height_top_k, width_top_k, dropout=0.):
        super().__init__()
        self.num_heads = num_heads  # 注意力头数
        self.dim_head = dim_head    # 每个头的维度
        self.scale = dim_head ** -0.5  # 缩放因子
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)  # Q、K、V 投影
        self.to_out = nn.Linear(dim, dim)  # 输出投影
        self.dropout = nn.Dropout(dropout)  # Dropout 层
        self.height_top_k = height_top_k  # 高度方向裁剪的 top-k
        self.width_top_k = width_top_k    # 宽度方向裁剪的 top-k

    def prune_attn(self, attn, height_top_k, width_top_k):
        B, H, N, _ = attn.shape
        device = attn.device
        # 动态调整 k，确保不超过 N
        height_k = min(height_top_k, N)
        width_k = min(width_top_k, N)

        # 裁剪行
        row_scores = attn.mean(dim=-1)  # 按列平均，形状 [B, H, N]
        _, top_row_idx = row_scores.topk(height_k, dim=-1)
        row_mask = torch.zeros(B, H, N, device=device)
        row_mask.scatter_(-1, top_row_idx, 1)
        row_mask = row_mask.unsqueeze(-1).expand(B, H, N, N)

        # 裁剪列
        col_scores = attn.mean(dim=-2)  # 按行平均，形状 [B, H, N]
        _, top_col_idx = col_scores.topk(width_k, dim=-1)
        col_mask = torch.zeros(B, H, N, device=device)
        col_mask.scatter_(-1, top_col_idx, 1)
        col_mask = col_mask.unsqueeze(-2).expand(B, H, N, N)

        # 合并掩码
        mask = row_mask * col_mask
        return attn * mask

    def forward(self, x):
        B, N, C = x.size()  # 输入形状：[批次，块数，通道]
        # 生成 Q、K、V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.num_heads,
                                       self.dim_head).transpose(1, 2), qkv)
        # 计算注意力分数
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # 应用裁剪
        attn = self.prune_attn(attn, self.height_top_k, self.width_top_k)
        # Softmax 和 Dropout
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        # 加权和
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.to_out(out)
        return out

# ViT 风格的混合感知块


class ViTHybridBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_head, height_top_k, width_top_k, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # 第一层归一化
        self.attn = DualPrunedSelfAttn(
            dim, num_heads, dim_head, height_top_k, width_top_k, dropout)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3,
                              padding=1, groups=dim)  # 深度卷积
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)  # 第二层归一化
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),  # 前馈网络扩展
            nn.GELU(),                      # GELU 激活
            nn.Dropout(dropout),            # Dropout
            nn.Linear(dim * ff_mult, dim)   # 前馈网络缩减
        )

    def forward(self, x):
        B, N, C = x.size()  # 输入为块序列
        # 注意力路径
        x_attn = self.attn(self.norm1(x))
        # 重塑为图像以进行卷积
        H = W = int(N ** 0.5)  # 假设块网格为方形
        x_conv = x.view(B, C, H, W)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.view(B, N, C)
        # 合并注意力与卷积结果
        x = x_attn + x_conv
        # 前馈路径
        x = x + self.ff(self.norm2(x))
        return x

# 更新后的生成器，固定输入尺寸为 [B, 3, 256, 256]


class SWTformerGenerator(nn.Module):
    def __init__(self, depth=1, weight=1, patch_size=16, num_heads=8, height_top_k=64, width_top_k=64):
        super().__init__()
        self.depth = depth
        self.weight = weight
        self.patch_size = patch_size
        self.img_size = (256, 256)  # 固定输入尺寸为 256x256

        # ----------------------- 编码器 -----------------------
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(math.ceil(8 * depth)),
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(math.ceil(8 * depth))),
            nn.SiLU(),
            PSA(int(math.ceil(8 * depth)), int(math.ceil(8 * depth)))  # 增加PSA
        )  # 256x256

        self.conv2 = self._make_cnn_block(
            int(math.ceil(8 * depth)), int(math.ceil(16 * depth)))  # 128x128
        self.conv3 = self._make_cnn_block(
            int(math.ceil(16 * depth)), int(math.ceil(32 * depth)))  # 64x64
        self.conv4 = self._make_cnn_block(
            int(math.ceil(32 * depth)), int(math.ceil(64 * depth)))  # 32x32

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ----------------------- Transformer -----------------------
        dim = int(math.ceil(64 * depth))
        self.patch_embed = nn.Linear(
            patch_size * patch_size * dim + dim, dim)  # 融合全局信息
        self.feat_h = 256 // 8  # 32
        self.feat_w = 256 // 8  # 32
        self.num_patches = (self.feat_h // patch_size) *\
            (self.feat_w // patch_size)  # 4 (对于 patch_size=16)
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.num_patches, dim))  # [1, 4, dim]

        self.transformer = nn.ModuleList([
            ViTHybridBlock(dim, num_heads, dim // num_heads,
                           height_top_k, width_top_k, ff_mult=4, dropout=0.1)
            for _ in range(2)  # 两层 Transformer
        ])

        # ----------------------- 解码器 -----------------------
        self.upconv6 = Disconv(
            dim, int(math.ceil(32 * depth)))  # 32x32 -> 64x64
        self.upconv7 = Disconv(int(math.ceil(32 * depth)),
                               int(math.ceil(16 * depth)))  # 64x64 -> 128x128
        self.upconv8 = Disconv(int(math.ceil(16 * depth)),
                               int(math.ceil(8*depth)))  # 128x128 -> 256x256
        self.conv9 = nn.Conv2d(int(math.ceil(8*depth)),
                               3, kernel_size=3, padding=1)  # 最后一层卷积

        self.sigmoid = nn.Sigmoid()  # 输出映射到 (0, 1)

        # 跳跃连接的 1x1 卷积
        self.skip_proj2 = nn.Conv2d(
            int(math.ceil(8 * depth)), int(math.ceil(8 * depth)), kernel_size=1, padding=0)
        self.skip_proj3 = nn.Conv2d(
            int(math.ceil(16 * depth)), int(math.ceil(16 * depth)), kernel_size=1, padding=0)
        self.skip_proj4 = nn.Conv2d(
            int(math.ceil(32 * depth)), int(math.ceil(32 * depth)), kernel_size=1, padding=0)

        # 初始化尺寸调整
        self.adjust6 = nn.Conv2d(
            dim // 2, int(math.ceil(32*depth)), kernel_size=1, padding=0)
        self.adjust7 = nn.Conv2d(
            int(math.ceil(16*depth)), int(math.ceil(16*depth)), kernel_size=1, padding=0)
        self.adjust8 = nn.Conv2d(
            int(math.ceil(8*depth)), int(math.ceil(8*depth)), kernel_size=1, padding=0)

    def _make_cnn_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            PSA(out_channels, out_channels)  # 增加PSA
        )

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 输入检查
        assert x.shape[2:] == torch.Size(
            [256, 256]), "输入尺寸必须为 [B, 3, 256, 256]"

        # ----------------------- 编码器 -----------------------
        x1 = self.conv1(x)  # 256x256 -> 256x256
        x2 = self.conv2(x1)  # 256x256 -> 128x128
        x3 = self.conv3(x2)  # 128x128 -> 64x64
        x4 = self.conv4(x3)  # 64x64 -> 32x32

        # 全局平均池化
        global_feat = self.global_pool(x4)
        global_feat = torch.flatten(global_feat, 1)  # [B, dim]

        # ----------------------- Transformer -----------------------
        patches = Patches(self.patch_size)(x4)  # 使用 Patches 提取图像块
        x5 = torch.cat([patches, global_feat[:, None, :].expand(-1,
                       patches.shape[1], -1)], dim=-1)  # 融合全局信息
        x5 = self.patch_embed(x5) + self.pos_embed  # 融合位置编码
        for block in self.transformer:
            x5 = block(x5)

        # 重塑回图像格式
        B, N, C = x5.shape
        x5 = x5.view(B, self.feat_h // self.patch_size,
                     self.feat_w // self.patch_size, C).permute(0, 3, 1, 2)

        # ----------------------- 解码器 -----------------------
        skip2 = self.skip_proj2(x1)  # 256x256 -> 256x256
        skip3 = self.skip_proj3(x2)  # 128x128 -> 128x128
        skip4 = self.skip_proj4(x3)  # 64x64 -> 64x64

        x6 = self.upconv6(x5)  # 32x32 -> 64x64
        x6 = F.interpolate(
            x6, size=skip4.shape[2:], mode='bilinear', align_corners=True)  # 插值到对应的大小
        x6 = self.adjust6(x6) + skip4  # 添加skip链接

        x7 = self.upconv7(x6)  # 64x64 -> 128x128
        x7 = F.interpolate(
            x7, size=skip3.shape[2:], mode='bilinear', align_corners=True)  # 插值到对应的大小
        x7 = self.adjust7(x7) + skip3

        x8 = self.upconv8(x7)  # 128x128 -> 256x256
        x8 = F.interpolate(
            x8, size=skip2.shape[2:], mode='bilinear', align_corners=True)
        x8 = self.adjust8(x8) + skip2

        x9 = self.conv9(x8)  # 调整到 256x256
        x9 = self.sigmoid(x9)

        return x9
