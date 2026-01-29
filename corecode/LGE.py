# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
LGT-Conv Neck Lite Module for YOLO11
轻量化版本的Neck LGT模块
保留核心Log-Gabor频域处理，移除冗余的注意力和低频分支
让YOLO自己的concat处理特征融合，LGT专注于高频增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class LGF(nn.Module):
    """
    轻量化Log-Gabor滤波器组
    保持原有的频域处理能力，但优化了实现
    """
    def __init__(self, in_channels, kernel_size=5, num_orientations=2, num_scales=1):
        super(LGF, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        
        # Log-Gabor分组卷积
        self.total_filters = num_orientations * num_scales
        self.grouped_conv = nn.Conv2d(
            in_channels, 
            in_channels * self.total_filters, 
            kernel_size, 
            padding=autopad(kernel_size), 
            groups=in_channels,
            bias=False
        )
        
        # 初始化Log-Gabor滤波器
        self._init_loggabor_filters()
        
    def _init_loggabor_filters(self):
        """初始化Log-Gabor滤波器"""
        with torch.no_grad():
            filters = []
            for s in range(self.num_scales):
                for k in range(self.num_orientations):
                    orientation = k * math.pi / self.num_orientations
                    scale = 0.5 + s * 0.3
                    lg_filter = self._create_loggabor_kernel(
                        self.kernel_size, orientation, scale
                    )
                    filters.append(lg_filter)
            
            filter_tensor = torch.stack(filters, dim=0)
            filter_tensor = filter_tensor.squeeze(2)
            repeated_filters = filter_tensor.repeat(self.in_channels, 1, 1, 1)
            self.grouped_conv.weight.data = repeated_filters
    
    def _create_loggabor_kernel(self, kernel_size, orientation, scale):
        """创建单个Log-Gabor滤波器核"""
        center = kernel_size // 2
        x, y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size), indexing='ij')
        x = x.float() - center
        y = y.float() - center
        
        # 旋转坐标
        cos_orient = math.cos(orientation)
        sin_orient = math.sin(orientation)
        x_rot = x * cos_orient + y * sin_orient
        y_rot = -x * sin_orient + y * cos_orient
        
        # Log-Gabor函数
        r = torch.sqrt(x_rot**2 + y_rot**2)
        theta = torch.atan2(y_rot, x_rot)
        r = torch.clamp(r, min=1e-6)
        
        scale_tensor = torch.tensor(scale, dtype=torch.float32)
        log_gabor = torch.exp(-(torch.log(r / scale_tensor)**2) / (2 * torch.log(torch.tensor(2.0))**2))
        log_gabor = log_gabor * torch.cos(theta)
        
        return log_gabor.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        """前向传播"""
        out = self.grouped_conv(x)
        B, C_total, H, W = out.shape
        C = self.in_channels
        out = out.view(B, C, self.total_filters, H, W)
        return out


class LGE(nn.Module):
    """
    轻量化LGT Neck模块
    
    核心设计：
    1. 保留Log-Gabor滤波器（核心高频增强）
    2. 移除低频分支（由后续C3k2处理）
    3. 移除注意力机制（用简单缩放因子）
    4. 移除内部融合（让YOLO的concat处理）
    5. 只处理Ci，输出增强后的特征
    
    使用方式：
    在 YAML 中（示例）：
    - [-1, 1, LGE, [128, 128, 3, 1, 1]]  # 处理 Ci
    - [[-1, 11], 1, Concat, [1]]        # 与 Pi+1 融合
    """
    def __init__(self, c1, c2=None, kernel_size=5, num_orientations=2, num_scales=1):
        super(LGE, self).__init__()
        
        # 如果c2未指定，默认等于c1（保持通道数不变）
        if c2 is None:
            c2 = c1
            
        self.c1 = c1
        self.c2 = c2
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        
        # Log-Gabor滤波器（核心组件，保持不变）
        self.loggabor_filter = LGF(
            c1, kernel_size, num_orientations, num_scales
        )
        
        # 方向和尺度的可学习权重
        self.orientation_weights = nn.Parameter(torch.ones(num_orientations))
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
        # 简单的缩放因子（替代注意力机制）
        self.scale_factor = nn.Parameter(torch.ones(1) * 0.5)
        
        # 高频特征处理（3x3 DWConv）
        self.high_conv = Conv(c1, c2, 3, 1, g=c1 if c1 == c2 else 1)
        
        # 残差连接（如果通道数匹配）
        if c1 != c2:
            self.shortcut = Conv(c1, c2, 1, 1)
            print(f"use normal conv for shortcut")
            print(f"c1: {c1}, c2: {c2}, kernel_size: {kernel_size}")
        else:
            self.shortcut = nn.Identity()
            print(f"use identity for shortcut")
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征 [B, C1, H, W]
        Returns:
            增强后的特征 [B, C2, H, W]
        """
        # 保存输入用于残差连接
        identity = self.shortcut(x)
        
        # Log-Gabor滤波器进行子带分解
        subbands = self.loggabor_filter(x)  # [B, C, K*S, H, W]
        
        # 加权聚合不同方向和尺度
        B, C, total_filters, H, W = subbands.shape
        subbands_reshaped = subbands.view(B, C, self.num_scales, self.num_orientations, H, W)
        
        # 归一化权重
        orientation_weights = F.softmax(self.orientation_weights, dim=0)
        scale_weights = F.softmax(self.scale_weights, dim=0)
        
        # 加权聚合
        f_high = torch.zeros_like(subbands_reshaped[:, :, 0, 0, :, :])
        for s in range(self.num_scales):
            for k in range(self.num_orientations):
                f_high += scale_weights[s] * orientation_weights[k] * subbands_reshaped[:, :, s, k, :, :]
        
        # 应用简单缩放因子（替代注意力）
        f_high = f_high * torch.sigmoid(self.scale_factor)
        
        # 高频特征处理
        f_high = self.high_conv(f_high)
        
        # 残差连接
        out = identity + f_high
        
        return out


class LGE_V2(nn.Module):
    """
    LGT Neck Lite V2 - 进一步简化版本
    
    相比V1的改进：
    - 移除复杂的加权聚合，直接对子带求平均
    - 移除可学习的方向/尺度权重
    - 进一步减少参数
    """
    def __init__(self, c1, c2=None, kernel_size=5, num_orientations=2, num_scales=1):
        super(LGE_V2, self).__init__()
        
        if c2 is None:
            c2 = c1
            
        self.c1 = c1
        self.c2 = c2
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        
        # Log-Gabor滤波器
        self.loggabor_filter = LGF(
            c1, kernel_size, num_orientations, num_scales
        )
        
        # 高频特征处理
        self.high_conv = Conv(c1, c2, 3, 1, g=c1 if c1 == c2 else 1)
        
        # 残差连接
        if c1 != c2:
            self.shortcut = Conv(c1, c2, 1, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        """前向传播"""
        identity = self.shortcut(x)
        
        # Log-Gabor子带分解
        subbands = self.loggabor_filter(x)  # [B, C, K*S, H, W]
        
        # 简单平均聚合（无可学习权重）
        f_high = subbands.mean(dim=2)  # [B, C, H, W]
        
        # 高频特征处理
        f_high = self.high_conv(f_high)
        
        # 残差连接
        out = identity + f_high
        
        return out


class LGE_U(nn.Module):
    """
    LGT Neck Ultra Lite - 极致轻量版本
    
    极简设计：
    - 保留Log-Gabor核心
    - 最小化其他所有组件
    - 适合对参数量和速度有极致要求的场景
    """
    def __init__(self, c1, c2=None, kernel_size=5):
        super(LGE_U, self).__init__()
        
        if c2 is None:
            c2 = c1
            
        # 简化：只用1个方向1个尺度
        self.loggabor_filter = LGF(c1, kernel_size, num_orientations=1, num_scales=1)
        
        # 最简单的处理
        self.conv = Conv(c1, c2, 3, 1, g=c1 if c1 == c2 else 1)
        
        if c1 != c2:
            self.shortcut = Conv(c1, c2, 1, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        """前向传播"""
        identity = self.shortcut(x)
        
        # Log-Gabor处理（只有1个filter）
        subbands = self.loggabor_filter(x)  # [B, C, 1, H, W]
        f_high = subbands.squeeze(2)  # [B, C, H, W]
        
        # 简单卷积
        f_high = self.conv(f_high)
        
        # 残差
        return identity + f_high


# 导出模块（论文命名）
__all__ = [
    'LGE',     # Log-Gabor Enhancer (main neck module)
    'LGE_V2',  # 更轻量：移除可学习权重
    'LGE_U',   # 极致轻量：1方向1尺度
    'LGF',     # Log-Gabor Filter bank
]

