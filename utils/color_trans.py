# Description: This file contains the functions for color transformation.
import torch
import torch.nn as nn


def F(X):  # X为任意形状的张量
    FX = 7.787 * X + 0.137931
    index = X > 0.008856
    FX[index] = torch.pow(X[index], 1.0 / 3.0)
    return FX


def anti_F(X):  # 逆操作。
    tFX = (X - 0.137931) / 7.787
    index = X > 0.206893
    tFX[index] = torch.pow(X[index], 3)
    return tFX


def gamma(r):
    r2 = r / 12.92
    index = r > 0.04045  # pow:0.0031308072830676845,/12.92:0.0031308049535603713
    r2[index] = torch.pow((r[index] + 0.055) / 1.055, 2.4)
    return r2


def anti_g(r):
    r2 = r * 12.92
    index = r > 0.0031308072830676845
    r2[index] = torch.pow(r[index], 1.0 / 2.4) * 1.055 - 0.055
    return r2


def PSrgb2lab(img):  # RGB img:[b,3,h,w]->lab,L[0,100],AB[-127,127]
    r = img[:, 0, :, :]
    g = img[:, 1, :, :]
    b = img[:, 2, :, :]

    r = gamma(r)
    g = gamma(g)
    b = gamma(b)

    X = r * 0.436052025 + g * 0.385081593 + b * 0.143087414
    Y = r * 0.222491598 + g * 0.716886060 + b * 0.060621486
    Z = r * 0.013929122 + g * 0.097097002 + b * 0.714185470
    X = X / 0.964221
    Z = Z / 0.825211

    F_X = F(X)
    F_Y = F(Y)
    F_Z = F(Z)

    # L = 903.3*Y
    # index = Y > 0.008856
    # L[index] = 116 * F_Y[index] - 16 # [0,100]
    L = 116 * F_Y - 16.0
    a = 500 * (F_X - F_Y)  # [-127,127]
    b = 200 * (F_Y - F_Z)  # [-127,127]

    # L = L
    # a = (a+128.0)
    # b = (b+128.0)
    return torch.stack([L, a, b], dim=1)


def PSlab2rgb(Lab):
    fY = (Lab[:, 0, :, :] + 16.0) / 116.0
    fX = Lab[:, 1, :, :] / 500.0 + fY
    fZ = fY - Lab[:, 2, :, :] / 200.0

    x = anti_F(fX)
    y = anti_F(fY)
    z = anti_F(fZ)
    x = x * 0.964221
    z = z * 0.825211
    #
    r = 3.13405134 * x - 1.61702771 * y - 0.49065221 * z
    g = -0.97876273 * x + 1.91614223 * y + 0.03344963 * z
    b = 0.07194258 * x - 0.22897118 * y + 1.40521831 * z
    #
    r = anti_g(r)
    g = anti_g(g)
    b = anti_g(b)
    return torch.stack([r, g, b], dim=1)


class RGB_HSV(nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB_HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):

        hue = torch.Tensor(img.shape[0], img.shape[2],
                           img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0]-img[:, 1]) / (
            img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2]-img[:, 0]) / (
            img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1]-img[:, 2]) / (
            img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue/6

        saturation = (img.max(1)[0] - img.min(1)[0]) / \
            (img.max(1)[0] + self.eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        # 对出界值的处理
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb
