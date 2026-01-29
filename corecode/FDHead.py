import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.utils.tal import dist2bbox, make_anchors


class Conv2d_cd(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        theta=1.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        if conv_weight.is_cuda:
            conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        else:
            conv_weight_cd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_cd = conv_weight_cd.to(conv_weight.dtype)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd
        )
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        theta=1.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_ad
        )
        return conv_weight_ad, self.conv.bias


class Conv2d_hd(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        theta=1.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        if conv_weight.is_cuda:
            conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        else:
            conv_weight_hd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_hd = conv_weight_hd.to(conv_weight.dtype)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_hd
        )
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        if conv_weight.is_cuda:
            conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        else:
            conv_weight_vd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_vd = conv_weight_vd.to(conv_weight.dtype)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_vd
        )
        return conv_weight_vd, self.conv.bias


class DEConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

        self.bn = nn.BatchNorm2d(dim)
        self.act = Conv.default_act

    def forward(self, x):
        if hasattr(self, 'conv1_1'):
            w1, b1 = self.conv1_1.get_weight()
            w2, b2 = self.conv1_2.get_weight()
            w3, b3 = self.conv1_3.get_weight()
            w4, b4 = self.conv1_4.get_weight()
            w5, b5 = self.conv1_5.weight, self.conv1_5.bias

            merged_weight = w1 + w2 + w3 + w4 + w5
            merged_bias = b1 + b2 + b3 + b4 + b5
            res = F.conv2d(x, merged_weight, merged_bias, stride=1, padding=1, groups=1)
        else:
            res = self.conv1_5(x)

        if hasattr(self, 'bn'):
            res = self.bn(res)
        return self.act(res)


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        return x * self.scale


def _make_divisible(v: int, divisor: int = 8) -> int:
    if divisor == 0:
        return int(v)
    return max(divisor, int(v) // divisor * divisor)


class Conv_GN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class DEConv_GN(DEConv):
    def __init__(self, dim):
        super().__init__(dim)
        self.bn = nn.GroupNorm(16, dim)

    def switch_to_deploy(self):
        if not hasattr(self, 'conv1_1'):
            return

        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        self.conv1_5.weight = torch.nn.Parameter(w1 + w2 + w3 + w4 + w5)
        self.conv1_5.bias = torch.nn.Parameter(b1 + b2 + b3 + b4 + b5)

        self.__delattr__('conv1_1')
        self.__delattr__('conv1_2')
        self.__delattr__('conv1_3')
        self.__delattr__('conv1_4')


class FDHead(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=(), freq_ratio=0.25, gate_reduction=8):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        self.hidc = hidc
        self.c_freq = _make_divisible(int(hidc * float(freq_ratio)), 8)
        self.c_freq = min(self.c_freq, hidc)

        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 1)) for x in ch)

        self.share_conv = nn.Sequential(
            DEConv_GN(hidc),
            nn.Sequential(
                DWConv(hidc, hidc, 3, 1),
                Conv(hidc, hidc, 1, 1),
            ),
        )

        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.c_freq > 0:
            hf_base = torch.tensor(
                [
                    [[0.5, 0.5], [0.5, 0.5]],
                    [[0.5, -0.5], [0.5, -0.5]],
                    [[0.5, 0.5], [-0.5, -0.5]],
                    [[0.5, -0.5], [-0.5, 0.5]],
                ],
                dtype=torch.float32,
            ).view(4, 1, 2, 2)
            weight = hf_base.repeat(self.c_freq, 1, 1, 1)
            self.register_buffer('haar_weight', weight)

            self.hf_w = nn.Parameter(torch.ones(3))

            c_mid = max(self.c_freq // int(gate_reduction), 1)
            self.hf_gate = nn.Sequential(
                nn.Conv2d(self.c_freq, c_mid, 1, 1, 0, bias=True),
                nn.SiLU(),
                nn.Conv2d(c_mid, self.c_freq, 1, 1, 0, bias=True),
                nn.Sigmoid(),
            )
            self.alpha_p2 = nn.Parameter(torch.tensor(0.5))

    def _haar_dwt(self, x):
        y = F.conv2d(x, self.haar_weight, stride=2, padding=0, groups=self.c_freq)
        B, _, H, W = y.shape
        y = y.view(B, self.c_freq, 4, H, W)
        ll, lh, hl, hh = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        return ll, lh, hl, hh

    def forward(self, x):
        for i in range(self.nl):
            xi = self.conv[i](x[i])
            fi = self.share_conv(xi)

            if i == 0 and self.c_freq > 0:
                fa = fi[:, : self.c_freq]
                fb = fi[:, self.c_freq :]

                _, lh, hl, hh = self._haar_dwt(fa)
                w = F.softmax(self.hf_w, dim=0)
                hf = w[0] * lh.abs() + w[1] * hl.abs() + w[2] * hh.abs()

                s = hf.mean((2, 3), keepdim=True)
                g = self.hf_gate(s)
                fa = fa * (1.0 + self.alpha_p2 * g)
                reg_f = torch.cat((fa, fb), 1)

                box = self.scale[i](self.cv2(reg_f))
                cls = self.cv3(fi)
            else:
                box = self.scale[i](self.cv2(fi))
                cls = self.cv3(fi)

            x[i] = torch.cat((box, cls), 1)

        if self.training:
            return x

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        self.cv2.bias.data[:] = 1.0
        self.cv3.bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


__all__ = [
    'FDHead',
]
