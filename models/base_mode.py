import math
import torch.nn as nn
import torch
from models.common import SPPELAN, C2f, C2fCIB, Concat, Disconv, Gencov, SCDown


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

    def __init__(self, depth=1, weight=1) -> None:
        super(Generator, self).__init__()
        depth = depth
        weight = weight
        self.conv1 = Gencov(3, math.ceil(8 * depth))
        self.conv2 = nn.Sequential(
            Gencov(math.ceil(8 * depth), math.ceil(16 * depth),
                   math.ceil(weight), 2),
            C2f(math.ceil(16 * depth), math.ceil(32 * depth), 1, True)
        )

        self.conv3 = nn.Sequential(
            Gencov(math.ceil(32 * depth),
                   math.ceil(64 * depth), math.ceil(weight), 2),
            C2f(math.ceil(64 * depth), math.ceil(128 * depth), 1, True)
        )

        self.conv4 = nn.Sequential(
            SCDown(math.ceil(128 * depth),
                   math.ceil(256 * depth), math.ceil(weight), 2),
            C2f(math.ceil(256 * depth), math.ceil(512 * depth), 1, True)
        )

        self.conv5 = nn.Sequential(
            SPPELAN(math.ceil(512 * depth), math.ceil(512 * depth),
                    math.ceil(256 * depth)),
            Gencov(math.ceil(512 * depth), math.ceil(256 * depth)), )
        self.conv6 = nn.Sequential(
            Gencov(math.ceil(256 * depth),
                   math.ceil(128 * depth), math.ceil(3 * weight)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.conv7 = nn.Sequential(
            Gencov(math.ceil(256 * depth),
                   math.ceil(64 * depth), math.ceil(weight)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.conv8 = nn.Sequential(
            C2fCIB(math.ceil(96 * depth), math.ceil(32 * depth)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.conv9 = Gencov(math.ceil(32 * depth), 3,
                            math.ceil(weight), act=False, bn=False)
        self.sigmoid = nn.Sigmoid()
        self.concat = Concat()

    def forward(self, x):

        # head net
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        # neck net

        x7 = self.conv7(self.concat([x6, x3]))
        x8 = self.conv8(self.concat([x2, x7]))
        x9 = self.sigmoid(self.conv9(x8))

        return x9.view(-1, 3, x.shape[2], x.shape[3])


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
        self.sigmoid = nn.Sigmoid()  # 使用 Identity，配合 BCEWithLogitsLoss

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
