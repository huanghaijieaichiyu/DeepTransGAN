import torch
import torch.nn as nn

from models.common import Conv, C2f, CA, SPPELAN, RepNCSPELAN4, Concat, EMA, ADown, SPPF, SimAM, CBAM
from utils.model_map import model_structure


class Generator(nn.Module):

    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.conv1 = Conv(1, 8)
        self.conv2 = nn.Sequential(Conv(8, 16, 3, 2),
                                   C2f(16, 32, shortcut=True)
                                   )
        self.conv3 = nn.Sequential(Conv(32, 64, 3, 2),
                                   C2f(64, 128, shortcut=True)
                                   )
        self.conv4 = nn.Sequential(Conv(128, 256, 3, 2),
                                   C2f(256, 512, shortcut=True)
                                   )
        self.conv5 = nn.Sequential(SPPELAN(512, 512, 256),
                                   Conv(512, 256, 3))
        self.conv6 = nn.Sequential(C2f(256, 128, shortcut=False),
                                   nn.Upsample(scale_factor=2),
                                   Conv(128, 64))
        self.conv7 = nn.Sequential(C2f(64, 32, shortcut=False),
                                   nn.Upsample(scale_factor=2))
        self.conv8 = nn.Sequential(C2f(64, 64, shortcut=True),
                                   nn.Upsample(scale_factor=2),
                                   Conv(64, 128)
                                   )
        self.conv9 = nn.Sequential(C2f(128, 64, shortcut=False),
                                   Conv(64, 32)
                                   )
        self.conv10 = nn.Sequential(C2f(32, 16, shortcut=False),
                                    Conv(16, 8),
                                    Conv(8, 2, act=False)
                                    )
        self.tanh = nn.Tanh()
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
        x8 = self.concat([x2, x7])
        x9 = self.conv8(x8)
        x10 = self.conv9(x9)
        x11 = self.conv10(x10)
        x12 = self.tanh(x11)

        x = x12.view(-1, 2, x.shape[2], x.shape[3])

        return x


class Discriminator(nn.Module):
    """
    Discriminator model with no activation function
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_in = nn.Sequential(Conv(2, 16, 3)
                                     )
        self.conv1 = nn.Sequential(C2f(16, 32),
                                   Conv(32, 64, 3, 2),
                                   Conv(64, 128, 3),
                                   C2f(128, 64),
                                   Conv(64, 32, 3, 2),
                                   Conv(32, 16, 3, 2),
                                   C2f(16, 8),
                                   Conv(8, 4, 3, 2),
                                   )
        self.conv_out = Conv(4, 1, 3, act=False)  # 记得替换激活函数

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.conv1(x1)
        x3 = self.conv_out(x2)
        x = self.sigmoid(x3)

        return x


if __name__ == '__main__':
    model = Discriminator()
    # model_ = Generator()
    d_params, d_macs = model_structure(model, (2, 256, 256))
    # d_params, d_macs = model_structure(model_, (1, 256, 256))
    print(d_params, d_macs)
