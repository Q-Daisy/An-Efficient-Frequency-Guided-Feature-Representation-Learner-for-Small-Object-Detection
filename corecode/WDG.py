# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
WDC-Block: Wavelet Difference Convolution Block
结合小波变换和中心差分卷积的轻量级模块，用于替换 YOLOv11 中的 C3k2

核心设计理念：
1. 使用 Haar 小波变换将特征分解为 LL、LH、HL、HH 四个分量
2. 在 LL（低频）分支使用中心差分卷积（CDC）增强边缘感知能力
3. 将高频分量（LH/HL/HH）作为注意力门控，修正低频特征
4. 使用重参数化技术，推理时 CDC 退化为普通卷积，保持轻量化

设计优势：
- CDC 可以在 H/2 分辨率下有效保留小目标边缘信息
- 高频门控机制使低频特征更聚焦于关键区域
- 重参数化保证推理速度与普通卷积相当
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class RDC(nn.Module):
    """
    Re-parameterized Central Difference Convolution (CDC)
    基于 DEA-Net 理论的重参数化中心差分卷积
    
    训练时：使用中心差分卷积增强边缘感知
    推理时：合并为普通卷积，保持速度
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super(RDC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 1. 普通卷积权重
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        )

        # 2. Theta 参数用于 CDC（控制强度和梯度的权衡）
        # 使用 1x1 的 theta 进行逐通道调整
        self.theta = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, 1, 1)
        )

        # 初始化
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.theta, 0.0)

        self.is_deploy = False

    def forward(self, x):
        if self.is_deploy:
            # 推理模式：直接使用合并后的权重
            return F.conv2d(
                x, self.reparam_weight, None, self.stride, self.padding, self.dilation, self.groups
            )

        # 训练模式：CDC = Vanilla Conv - Theta * Center
        # 构造 CDC 卷积核：在中心位置减去 theta
        kernel_cdc = self.weight.clone()
        center_idx = self.kernel_size // 2
        
        # 在卷积核中心位置减去 theta
        # theta 的形状是 [out_channels, in_channels//groups, 1, 1]
        # 需要广播到卷积核的中心位置
        kernel_cdc[:, :, center_idx, center_idx] -= self.theta.squeeze(-1).squeeze(-1)

        return F.conv2d(
            x, kernel_cdc, None, self.stride, self.padding, self.dilation, self.groups
        )

    def switch_to_deploy(self):
        """切换到部署模式：合并卷积核"""
        if not self.is_deploy:
            center_idx = self.kernel_size // 2
            kernel_final = self.weight.clone()
            kernel_final[:, :, center_idx, center_idx] -= self.theta.squeeze(-1).squeeze(-1)

            self.reparam_weight = nn.Parameter(kernel_final.detach())
            # 删除训练时的参数
            del self.weight
            del self.theta
            self.is_deploy = True


class WDG(nn.Module):
    """
    Wavelet Difference Gate (WDG) Block
    作为 Bottleneck 替换，结合小波变换和差分卷积
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            shortcut: 是否使用残差连接
            g: 分组卷积组数
            e: 扩展比例
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.add = shortcut and c1 == c2

        # 输入投影
        self.cv1 = Conv(c1, c_, 1, 1)

        # 输出投影
        self.cv2 = Conv(c_, c2, 1, 1)

        # --- 低频分支（LL）：使用 RDC 增强边缘感知 ---
        self.cdc_ll = nn.Sequential(
            RDC(c_, c_, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        # --- 高频分支（LH/HL/HH）：作为注意力门控 ---
        # 将 3 个高频分量融合为注意力图
        self.hf_gate = nn.Sequential(
            nn.Conv2d(c_ * 3, c_, 1, 1, groups=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.Sigmoid()  # 生成 (0,1) 的权重
        )

    def _haar_dwt(self, x):
        """
        Haar 小波变换（手动实现，避免依赖 pywt）
        将输入分解为 LL, LH, HL, HH 四个分量
        处理奇数尺寸的情况，确保所有切片尺寸一致
        """
        b, c, h, w = x.shape
        
        # 如果高度或宽度是奇数，裁剪到偶数尺寸
        h_even = h if h % 2 == 0 else h - 1
        w_even = w if w % 2 == 0 else w - 1
        x = x[:, :, :h_even, :w_even]
        
        # 使用切片实现 2x2 下采样
        # 由于已经裁剪到偶数，所有切片应该有相同的尺寸
        x0 = x[:, :, 0::2, 0::2]  # 左上
        x1 = x[:, :, 0::2, 1::2]  # 右上
        x2 = x[:, :, 1::2, 0::2]  # 左下
        x3 = x[:, :, 1::2, 1::2]  # 右下

        # 确保所有切片尺寸一致（取最小值，处理可能的边界情况）
        min_h = min(x0.shape[2], x1.shape[2], x2.shape[2], x3.shape[2])
        min_w = min(x0.shape[3], x1.shape[3], x2.shape[3], x3.shape[3])
        
        if min_h < x0.shape[2] or min_w < x0.shape[3]:
            x0 = x0[:, :, :min_h, :min_w]
            x1 = x1[:, :, :min_h, :min_w]
            x2 = x2[:, :, :min_h, :min_w]
            x3 = x3[:, :, :min_h, :min_w]

        # Haar 小波变换公式
        ll = (x0 + x1 + x2 + x3) / 2.0  # 低频
        lh = (x0 - x1 + x2 - x3) / 2.0  # 水平高频
        hl = (x0 + x1 - x2 - x3) / 2.0  # 垂直高频
        hh = (x0 - x1 - x2 + x3) / 2.0  # 对角高频

        return ll, lh, hl, hh, h_even, w_even  # 返回裁剪后的尺寸

    def _haar_idwt(self, ll, lh, hl, hh, target_h=None, target_w=None):
        """
        Haar 小波逆变换（重构）
        将四个分量重构为原始分辨率
        """
        # 逆变换公式
        y0 = (ll + lh + hl + hh) / 2.0
        y1 = (ll - lh + hl - hh) / 2.0
        y2 = (ll + lh - hl - hh) / 2.0
        y3 = (ll - lh - hl + hh) / 2.0

        # 获取空间尺寸
        b, c, h, w = ll.shape

        # 重构为 2H x 2W
        out_h = target_h if target_h is not None else h * 2
        out_w = target_w if target_w is not None else w * 2
        
        out = torch.zeros((b, c, out_h, out_w), device=ll.device, dtype=ll.dtype)
        
        # 使用插值或直接赋值重构
        # 如果目标尺寸与 2H x 2W 不同，使用插值
        if out_h == h * 2 and out_w == w * 2:
            out[:, :, 0::2, 0::2] = y0
            out[:, :, 0::2, 1::2] = y1
            out[:, :, 1::2, 0::2] = y2
            out[:, :, 1::2, 1::2] = y3
        else:
            # 先重构到 2H x 2W，然后插值到目标尺寸
            temp = torch.zeros((b, c, h * 2, w * 2), device=ll.device, dtype=ll.dtype)
            temp[:, :, 0::2, 0::2] = y0
            temp[:, :, 0::2, 1::2] = y1
            temp[:, :, 1::2, 0::2] = y2
            temp[:, :, 1::2, 1::2] = y3
            out = F.interpolate(temp, size=(out_h, out_w), mode='bilinear', align_corners=False)

        return out

    def forward(self, x):
        # 1. 输入投影
        x_in = self.cv1(x)

        # 2. Haar DWT 分解（内部会处理奇数尺寸）
        ll, lh, hl, hh, h_even, w_even = self._haar_dwt(x_in)

        # 3. 处理低频分支：使用 CDC 增强边缘
        feat_ll = self.cdc_ll(ll)

        # 4. 处理高频分支：生成注意力门控
        hf_cat = torch.cat([lh, hl, hh], dim=1)  # [B, 3*c_, H/2, W/2]
        feat_gate = self.hf_gate(hf_cat)  # [B, c_, H/2, W/2]

        # 5. 频率交互：用高频信息修正低频特征
        # 使用加性门控：(1 + gate) 增强关键区域
        ll_refined = feat_ll * (1.0 + feat_gate)

        # 6. 保持原始高频信息（可以选择性地使用原始或更新的高频）
        # 这里使用原始高频，保持纹理细节
        r_lh, r_hl, r_hh = lh, hl, hh

        # 7. 逆小波变换重构（恢复到裁剪后的偶数尺寸）
        out = self._haar_idwt(ll_refined, r_lh, r_hl, r_hh, target_h=h_even, target_w=w_even)

        # 8. 如果原始输入是奇数尺寸，需要插值或裁剪回原始尺寸
        _, _, h_orig, w_orig = x_in.shape
        if out.shape[2] != h_orig or out.shape[3] != w_orig:
            out = F.interpolate(out, size=(h_orig, w_orig), mode='bilinear', align_corners=False)

        # 9. 输出投影
        out = self.cv2(out)

        # 10. 残差连接（如果需要）
        return x + out if self.add else out


class C3_WDG(nn.Module):
    """
    C3 模块，使用 WDG 作为 Bottleneck
    替换原始的 C3k2 中的 Bottleneck
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            n: Bottleneck 重复次数
            c3k: 是否使用 C3k（兼容参数，当前不支持）
            e: 扩展比例
            g: 分组数
            shortcut: 是否使用残差连接
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # 使用 WDG 替换原始 Bottleneck
        if c3k:
            # 如果要求使用 C3k，我们仍然使用 WDG（因为 C3k 也是 Bottleneck 的变种）
            self.m = nn.ModuleList(
                WDG(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                WDG(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
            )

    def forward(self, x):
        """前向传播"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """使用 split 的前向传播"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":
    # 测试 WDG 模块
    print("测试 WDG 模块...")

    # 生成测试输入
    B, C1, H, W = 2, 64, 64, 64
    x = torch.randn(B, C1, H, W)
    print(f"输入形状: {x.shape}")

    # 测试 RDC
    print("\n测试 RDC...")
    cdc = RDC(64, 64, kernel_size=3, padding=1)
    out_cdc = cdc(x)
    print(f"RDC 输出形状: {out_cdc.shape}")
    
    # 测试切换到部署模式
    cdc.switch_to_deploy()
    out_cdc_deploy = cdc(x)
    print(f"RDC (部署模式) 输出形状: {out_cdc_deploy.shape}")
    print(f"输出差异: {torch.abs(out_cdc - out_cdc_deploy).max().item():.6f}")

    # 测试 WDG
    print("\n测试 WDG...")
    wdg = WDG(c1=64, c2=64, shortcut=True)
    out_wdg = wdg(x)
    print(f"WDG 输出形状: {out_wdg.shape}")

    # 测试 C3_WDG
    print("\n测试 C3_WDG...")
    c3_wdg = C3_WDG(c1=64, c2=128, n=2, shortcut=True)
    out_c3_wdg = c3_wdg(x)
    print(f"C3_WDG 输出形状: {out_c3_wdg.shape}")

    # 计算参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n参数量统计:")
    print(f"RDC: {count_parameters(cdc):,} 参数")
    print(f"WDG: {count_parameters(wdg):,} 参数")
    print(f"C3_WDG: {count_parameters(c3_wdg):,} 参数")

    print("\n测试完成！")

__all__ = [
    'RDC',
    'WDG',
    'C3_WDG',
]