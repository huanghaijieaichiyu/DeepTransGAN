import math
import torch.nn as nn

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
    def __init__(self, depth=0.8, weight=1):
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
        self.act = nn.Sigmoid()

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

        # Define layers
        self.conv_in = nn.Sequential(Disconv(3, 8), RepViTBlock(8, 16, 3, 2))
        self.conv1 = self._make_sequential(16, 32, 3)
        self.conv_out = Disconv(4, 1, bn=False, act=False)
        self.flat = nn.Flatten()
        self.liner = nn.Sequential(
            nn.Linear(math.ceil(16 * self.ratio) **
                      2, 16 * 16), nn.LeakyReLU(),
            nn.Linear(16 * 16, 8 * 16), nn.LeakyReLU(),
            nn.Linear(8 * 16, 8 * 8), nn.LeakyReLU(),
            nn.Linear(8 * 8, 4 * 8), nn.LeakyReLU(),
            nn.Linear(4 * 8, 8), nn.LeakyReLU(),
            nn.Linear(8, 1)
        )
        self.act = nn.Identity()

    def _make_sequential(self, in_channels, out_channels, weight):
        """
        Creates a sequential block for the critic.
        """
        return nn.Sequential(
            RepViTBlock(in_channels, out_channels, weight, 2),
            Disconv(out_channels, out_channels * 2),  # Increased out_channels
            RepViTBlock(out_channels * 2, in_channels, weight, 2),
            Disconv(in_channels, in_channels // 2),  # Reduced out_channels
            RepViTBlock(in_channels // 2, in_channels // 4, weight, 2),
            Disconv(in_channels // 4, in_channels // 8)
        )

    def forward(self, x):
        """
        :param x: input image
        :return: output
        """
        x = self.conv_out(self.conv1(self.conv_in(x))).view(
            self.batch_size if x.shape[0] == self.batch_size else x.shape[0], -1)
        return x
