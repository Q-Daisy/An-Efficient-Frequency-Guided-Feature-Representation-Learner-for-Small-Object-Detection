# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .WTConv import WTConv


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
    
    def __init__(self, in_channels, kernel_size=5, num_orientations=2, num_scales=1):
        super(LGF, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        
        
        self.total_filters = num_orientations * num_scales
        self.grouped_conv = nn.Conv2d(
            in_channels, 
            in_channels * self.total_filters, 
            kernel_size, 
            padding=autopad(kernel_size), 
            groups=in_channels,
            bias=False
        )
        
        
        self._init_loggabor_filters()
        
    def _init_loggabor_filters(self):
        
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
        
        center = kernel_size // 2
        x, y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size), indexing='ij')
        x = x.float() - center
        y = y.float() - center
        
        
        cos_orient = math.cos(orientation)
        sin_orient = math.sin(orientation)
        x_rot = x * cos_orient + y * sin_orient
        y_rot = -x * sin_orient + y * cos_orient
        
        
        r = torch.sqrt(x_rot**2 + y_rot**2)
        theta = torch.atan2(y_rot, x_rot)
        r = torch.clamp(r, min=1e-6)
        
        scale_tensor = torch.tensor(scale, dtype=torch.float32)
        log_gabor = torch.exp(-(torch.log(r / scale_tensor)**2) / (2 * torch.log(torch.tensor(2.0))**2))
        log_gabor = log_gabor * torch.cos(theta)
        
        return log_gabor.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        
        out = self.grouped_conv(x)
        B, C_total, H, W = out.shape
        C = self.in_channels
        out = out.view(B, C, self.total_filters, H, W)
        return out


class LGE_W(nn.Module):
   
    def __init__(self, c1, c2=None, kernel_size=5, num_orientations=2, num_scales=1):
        super(LGE_W, self).__init__()
        
        
        if c2 is None:
            c2 = c1
            
        self.c1 = c1
        self.c2 = c2
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        
       
        self.loggabor_filter = LGF(
            c1, kernel_size, num_orientations, num_scales
        )
        
        
        self.orientation_weights = nn.Parameter(torch.ones(num_orientations))
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
        
        self.scale_factor = nn.Parameter(torch.ones(1) * 0.5)
        
      
        if c1 == c2:
            self.high_conv = WTConv(c1, c2, kernel_size=3, stride=1)
            print(f"use WTConv2d for high_conv")
        else:
            self.high_conv = Conv(c1, c2, 3, 1, g=1)
            print(f"use normal conv for high_conv")
        
        if c1 != c2:
            self.shortcut = Conv(c1, c2, 1, 1)
            print(f"use normal conv for shortcut")
        else:
            self.shortcut = nn.Identity()
            print(f"use identity for shortcut")
    
    def forward(self, x):
      
        identity = self.shortcut(x)
        
       
        subbands = self.loggabor_filter(x)  # [B, C, K*S, H, W]
        
        
        B, C, total_filters, H, W = subbands.shape
        subbands_reshaped = subbands.view(B, C, self.num_scales, self.num_orientations, H, W)
        
        
        orientation_weights = F.softmax(self.orientation_weights, dim=0)
        scale_weights = F.softmax(self.scale_weights, dim=0)
        
        
        f_high = torch.zeros_like(subbands_reshaped[:, :, 0, 0, :, :])
        for s in range(self.num_scales):
            for k in range(self.num_orientations):
                f_high += scale_weights[s] * orientation_weights[k] * subbands_reshaped[:, :, s, k, :, :]
        
       
        f_high = f_high * torch.sigmoid(self.scale_factor)
        
      
        f_high = self.high_conv(f_high)
        
      
        out = identity + f_high
        
        return out


__all__ = [
    'LGE_W',
    'LGF',
]

