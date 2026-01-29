# Ultralytics YOLO 🚀, AGPL-3.0 license


import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class RDC(nn.Module):
    

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super(RDC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

       
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        )

       
        self.theta = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, 1, 1)
        )

       
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.theta, 0.0)

        self.is_deploy = False

    def forward(self, x):
        if self.is_deploy:
           
            return F.conv2d(
                x, self.reparam_weight, None, self.stride, self.padding, self.dilation, self.groups
            )

       
        kernel_cdc = self.weight.clone()
        center_idx = self.kernel_size // 2
        
        
        kernel_cdc[:, :, center_idx, center_idx] -= self.theta.squeeze(-1).squeeze(-1)

        return F.conv2d(
            x, kernel_cdc, None, self.stride, self.padding, self.dilation, self.groups
        )

    def switch_to_deploy(self):
        
        if not self.is_deploy:
            center_idx = self.kernel_size // 2
            kernel_final = self.weight.clone()
            kernel_final[:, :, center_idx, center_idx] -= self.theta.squeeze(-1).squeeze(-1)

            self.reparam_weight = nn.Parameter(kernel_final.detach())
           
            del self.weight
            del self.theta
            self.is_deploy = True


class WDG(nn.Module):
    """
    Wavelet Difference Gate (WDG) Block
   
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
       
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.add = shortcut and c1 == c2

       
        self.cv1 = Conv(c1, c_, 1, 1)

        
        self.cv2 = Conv(c_, c2, 1, 1)

       
        self.cdc_ll = nn.Sequential(
            RDC(c_, c_, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

       
        self.hf_gate = nn.Sequential(
            nn.Conv2d(c_ * 3, c_, 1, 1, groups=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.Sigmoid()  
        )

    def _haar_dwt(self, x):
       
        b, c, h, w = x.shape
        
        
        h_even = h if h % 2 == 0 else h - 1
        w_even = w if w % 2 == 0 else w - 1
        x = x[:, :, :h_even, :w_even]
      
        x0 = x[:, :, 0::2, 0::2]  # 左上
        x1 = x[:, :, 0::2, 1::2]  # 右上
        x2 = x[:, :, 1::2, 0::2]  # 左下
        x3 = x[:, :, 1::2, 1::2]  # 右下

       
        min_h = min(x0.shape[2], x1.shape[2], x2.shape[2], x3.shape[2])
        min_w = min(x0.shape[3], x1.shape[3], x2.shape[3], x3.shape[3])
        
        if min_h < x0.shape[2] or min_w < x0.shape[3]:
            x0 = x0[:, :, :min_h, :min_w]
            x1 = x1[:, :, :min_h, :min_w]
            x2 = x2[:, :, :min_h, :min_w]
            x3 = x3[:, :, :min_h, :min_w]

        # Haar 
        ll = (x0 + x1 + x2 + x3) / 2.0  
        lh = (x0 - x1 + x2 - x3) / 2.0  
        hl = (x0 + x1 - x2 - x3) / 2.0 
        hh = (x0 - x1 - x2 + x3) / 2.0  

        return ll, lh, hl, hh, h_even, w_even  

    def _haar_idwt(self, ll, lh, hl, hh, target_h=None, target_w=None):
        
        y0 = (ll + lh + hl + hh) / 2.0
        y1 = (ll - lh + hl - hh) / 2.0
        y2 = (ll + lh - hl - hh) / 2.0
        y3 = (ll - lh - hl + hh) / 2.0

        
        b, c, h, w = ll.shape

       
        out_h = target_h if target_h is not None else h * 2
        out_w = target_w if target_w is not None else w * 2
        
        out = torch.zeros((b, c, out_h, out_w), device=ll.device, dtype=ll.dtype)
        
       
        if out_h == h * 2 and out_w == w * 2:
            out[:, :, 0::2, 0::2] = y0
            out[:, :, 0::2, 1::2] = y1
            out[:, :, 1::2, 0::2] = y2
            out[:, :, 1::2, 1::2] = y3
        else:
           
            temp = torch.zeros((b, c, h * 2, w * 2), device=ll.device, dtype=ll.dtype)
            temp[:, :, 0::2, 0::2] = y0
            temp[:, :, 0::2, 1::2] = y1
            temp[:, :, 1::2, 0::2] = y2
            temp[:, :, 1::2, 1::2] = y3
            out = F.interpolate(temp, size=(out_h, out_w), mode='bilinear', align_corners=False)

        return out

    def forward(self, x):
      
        x_in = self.cv1(x)

       
        ll, lh, hl, hh, h_even, w_even = self._haar_dwt(x_in)

       
        feat_ll = self.cdc_ll(ll)

      
        hf_cat = torch.cat([lh, hl, hh], dim=1)  # [B, 3*c_, H/2, W/2]
        feat_gate = self.hf_gate(hf_cat)  # [B, c_, H/2, W/2]

        ll_refined = feat_ll * (1.0 + feat_gate)

       
        r_lh, r_hl, r_hh = lh, hl, hh

       
        out = self._haar_idwt(ll_refined, r_lh, r_hl, r_hh, target_h=h_even, target_w=w_even)

      
        _, _, h_orig, w_orig = x_in.shape
        if out.shape[2] != h_orig or out.shape[3] != w_orig:
            out = F.interpolate(out, size=(h_orig, w_orig), mode='bilinear', align_corners=False)

       
        out = self.cv2(out)

     
        return x + out if self.add else out


class C3_WDG(nn.Module):
   

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
       
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

       
        if c3k:
            
            self.m = nn.ModuleList(
                WDG(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                WDG(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
            )

    def forward(self, x):
       
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
       
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":
   
    print("Testing WDG ...")

  
    B, C1, H, W = 2, 64, 64, 64
    x = torch.randn(B, C1, H, W)
    print(f"Input shape: {x.shape}")

   
    print("\nTesting RDC...")
    cdc = RDC(64, 64, kernel_size=3, padding=1)
    out_cdc = cdc(x)
    print(f"RDC output shape: {out_cdc.shape}")
    
  
    cdc.switch_to_deploy()
    out_cdc_deploy = cdc(x)
    print(f"RDC output shape: {out_cdc_deploy.shape}")
    print(f"output diff: {torch.abs(out_cdc - out_cdc_deploy).max().item():.6f}")

  
    print("\nTest WDG...")
    wdg = WDG(c1=64, c2=64, shortcut=True)
    out_wdg = wdg(x)
    print(f"WDG output shape: {out_wdg.shape}")

   
    print("\nTest C3_WDG...")
    c3_wdg = C3_WDG(c1=64, c2=128, n=2, shortcut=True)
    out_c3_wdg = c3_wdg(x)
    print(f"C3_WDG output shape: {out_c3_wdg.shape}")

   
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nParameters:")
    print(f"RDC: {count_parameters(cdc):,} ")
    print(f"WDG: {count_parameters(wdg):,} ")
    print(f"C3_WDG: {count_parameters(c3_wdg):,} ")

    print("\nTest Over!")

__all__ = [
    'RDC',
    'WDG',
    'C3_WDG',
]