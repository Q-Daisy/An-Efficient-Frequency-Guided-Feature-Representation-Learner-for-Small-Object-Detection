# 自定义模块总文档（Frequency-Guided YOLO11 改进版）

本文件用于完整说明 `corecode/` 中相对“普通/原生 YOLO（Ultralytics YOLO11 风格）”新增或改造的模块。目标是：**把本文件交给另一个 AI 或新同学，即可在不阅读源码的情况下理解每个模块的作用、接口、张量形状、配置方式以及与 YOLO 结构的交互关系**。

本目录包含的自定义实现：
- `WDG.py`（WDG / C3_WDG / RDC）
- `LGE.py`（LGE / LGF 等 Log-Gabor 相关模块）
- `LGE_W.py`（LGE_W：LGE-W 波小波变体）
- `WTConv.py`（WTConv：Wavelet-Transform Convolution）
- `FDHead.py`（FDHead：Frequency-Driven Head）
- `yolo11s-head2-LSDECD-DWTP2GateLite-LGTConv-Neck-Lite-WTConv-Internal-WDC-lite2.yaml`

---

## 1. 你在 YAML 里做了什么改动（相对普通 YOLO）

当前 YAML（`yolo11s-head2-LSDECD-DWTP2GateLite-LGTConv-Neck-Lite-WTConv-Internal-WDC.yaml`）体现的主要差异：

### 1.1 Backbone：用 `C3k2_WDC` 替换部分 `C3k2`
在 Backbone 的部分 stage 里：
- 用 `C3k2_WDC` 替换 `C3k2` 的 Bottleneck，使其内部引入 **Haar 小波频域分解 + CDC（Central Difference Convolution）+ 高频门控**。

### 1.2 Neck：插入 `LGTConvNeckLite` / `LGTConvNeckLite_WT`
在上采样后的各个尺度特征上，先做一层轻量的 **Log-Gabor 高频增强**，再与对应的 lateral 特征 `Concat`，再进入 `C3k2`/`C3k2_WDC`。

### 1.3 Head：使用自定义检测头 `Detect_LSDECD_DWTP2GateLite`
检测头不是默认 YOLO Detect，而是：
- 共享卷积干路（`share_conv`）
- 在 **P2（低层高分辨率）分支**上做 `Haar DWT` + 高频统计 + 门控（Gate）增强
- 仍保留 Ultralytics 风格的 `DFL`、anchor/stride 逻辑、推理 decode 等

---

## 2. 依赖与环境假设（非常重要）

### 2.1 Python 包依赖
- `torch`
- `ultralytics`（你在多个模块里直接 `from ultralytics... import ...`）
- `einops`（`Detect_LSDECD_DWTP2GateLite.py` 里用到 `Rearrange`）
- `pywt`（PyWavelets，`WTConv_original.py` 里用到）

如果缺少 `einops` 或 `pywt`：
- `LGTConvNeckLite_WT`（依赖 `WTConv_original`）以及 `WTConv2d_original` 会直接导入失败或运行失败。
- `Detect_LSDECD_DWTP2GateLite` 的 re-parameterization 权重重排依赖 `einops`。

### 2.2 输入张量约定
以下文档都默认输入是 PyTorch 的 NCHW：
- 特征张量：`[B, C, H, W]`
- YOLO 检测头输入：一个 list/tuple，包含多尺度特征图 `x[i]`。

---

## 3. 模块之间的调用/依赖关系（结构图）

- `C3k2_WDC` 依赖：
  - `WDC_Block`
    - `RepCDC`
    - Haar DWT/IDWT（手写实现）

- `LGTConvNeckLite` / `LGTConvNeckLiteV2` / `LGTConvNeckUltraLite` 依赖：
  - `LogGaborFilterLite`
  - （内部自定义 `Conv`，与 Ultralytics 的 `Conv` 不是同一个类）

- `LGTConvNeckLite_WT` 依赖：
  - `LogGaborFilterLite`
  - `WTConv2d_original`（当 `c1 == c2` 时作为高频分支卷积）

- `WTConv2d_original` 依赖：
  - `pywt` 生成小波滤波器
  - 手写 wavelet transform / inverse wavelet transform

- `Detect_LSDECD_DWTP2GateLite` 依赖：
  - `ultralytics.nn.modules.block.DFL`
  - `ultralytics.nn.modules.conv.Conv`、`DWConv`
  - `ultralytics.utils.tal.dist2bbox`、`make_anchors`
  - 自定义：`DEConv_GN`（可切 deploy）、`Conv_GN`、Haar DWT（conv 实现）与 Gate

---

## 4. 详细模块文档

### 4.1 `LGF`（Log-Gabor Filter，轻量实现）

**文件**：`LGE.py` 与 `LGE_W.py`（两处都实现了一份同名类，逻辑一致）

**作用**：
- 用一组固定（非学习）的 Log-Gabor 核对每个输入通道做 depthwise 卷积，得到多个方向/尺度的子带响应。
- 目的是提取/增强 **高频纹理、边缘与方向性结构**。

**构造参数**：
- `in_channels`: 输入通道数 `C`
- `kernel_size`: 卷积核大小 `k`（默认 5）
- `num_orientations`: 方向数 `K`（默认 2）
- `num_scales`: 尺度数 `S`（默认 1）

**内部结构要点**：
- 用 `groups=in_channels` 的 depthwise 卷积：
  - 输入：`[B, C, H, W]`
  - 输出通道：`C * (K*S)`
  - 然后 reshape 为 `[..., C, K*S, H, W]`
- 滤波器初始化 `_init_loggabor_filters()`：
  - 为每个 `(scale, orientation)` 生成一个 Log-Gabor kernel
  - 再 repeat 到每个输入通道（因此每个通道共享同一组滤波器形状，但卷积是逐通道执行）

**前向输出形状**：
- 输入 `x`: `[B, C, H, W]`
- 输出 `subbands`: `[B, C, K*S, H, W]`

**关键注意点**：
- 这是“固定滤波器”（weight 在 `no_grad()` 初始化后直接写入 conv weight），不是学习得到的卷积核。
- 两个文件各自定义了 `Conv` 与 `autopad`，与 Ultralytics 的实现可能存在差异；如果未来要合并到 Ultralytics 工程，建议统一依赖来源。

---

### 4.2 `LGE`（Log-Gabor Enhancer，高频增强 Neck 模块，带可学习方向/尺度加权）

**文件**：`LGE.py`

**动机**：
- 让 Neck 侧在拼接融合前先获得更强的高频/边缘信息（对小目标更敏感）。
- 相比更复杂的频域+注意力设计，这里强调 **轻量化**：
  - 不做低频分支
  - 不做内部融合（交给 YOLO 的 `Concat`）
  - 用一个 `scale_factor` 替代复杂注意力

**构造参数**：
- `c1`: 输入通道数
- `c2`: 输出通道数（默认 `None`，表示等于 `c1`）
- `kernel_size`: Log-Gabor 核大小
- `num_orientations`: 方向数
- `num_scales`: 尺度数

**内部结构**：
- `loggabor_filter`: `LogGaborFilterLite(c1, kernel_size, K, S)`
- `orientation_weights`: `[K]` 可学习参数，softmax 后为方向权重
- `scale_weights`: `[S]` 可学习参数，softmax 后为尺度权重
- `scale_factor`: 标量参数，经过 `sigmoid` 后作为全局缩放
- `high_conv`: 一个 3x3 卷积（当 `c1==c2` 时是 depthwise：`g=c1`，否则普通卷积）
- `shortcut`: 当 `c1!=c2` 时用 1x1 conv 对齐通道，否则 identity

**前向过程（可直接作为伪代码理解）**：
- `identity = shortcut(x)`
- `subbands = loggabor_filter(x)`，得到 `[B, C, S*K, H, W]`
- reshape 为 `[B, C, S, K, H, W]`
- softmax 归一化权重：`w_s`、`w_k`
- 加权求和：
  - `f_high = Σ_s Σ_k (w_s[s] * w_k[k] * subbands[:, :, s, k])` 得到 `[B, C, H, W]`
- `f_high = f_high * sigmoid(scale_factor)`
- `f_high = high_conv(f_high)` 得到 `[B, c2, H, W]`
- 输出：`out = identity + f_high`

**输入输出形状**：
- 输入：`[B, c1, H, W]`
- 输出：`[B, c2, H, W]`

**YAML 用法（来自你的 YAML）**：
```yaml
- [-1, 1, LGTConvNeckLite, [192, 192, 3, 1, 1]]
```
对应 `__init__(c1=192, c2=192, kernel_size=3, num_orientations=1, num_scales=1)`。

**注意点 / 可能的坑**：
- forward 中用双重 for-loop 聚合子带，K*S 较大时会慢；如果未来需要加速，可向量化（但当前先按你的实现定义）。
- 模块中有 `print(...)`，训练时会打印很多日志（若你追求整洁日志，可以考虑后续移除/加开关）。

---

### 4.3 `LGTConvNeckLiteV2` 与 `LGTConvNeckUltraLite`

**文件**：`LGTConv_Neck_Lite.py`

这两个属于同一思想的简化版。

#### 4.3.1 `LGTConvNeckLiteV2`
- 与 V1 的差异：
  - 移除方向/尺度的可学习权重
  - 直接 `subbands.mean(dim=2)` 对 `K*S` 维做平均
- 输入输出形状与 `LGTConvNeckLite` 相同。

#### 4.3.2 `LGTConvNeckUltraLite`
- 极简：固定 `num_orientations=1`、`num_scales=1`
- `subbands.squeeze(2)` 得到 `[B, C, H, W]`
- 仍然做 `Conv(3x3)` + 残差

**适用建议**：
- 如果你论文强调“性能/精度”，V1（可学习权重）解释空间更大。
- 如果你论文强调“速度/参数”，V2/UltraLite 更容易叙述为轻量消融。

---

### 4.4 `WTConv`（Wavelet-Transform Convolution）

**文件**：`WTConv.py`

**作用**：
- 用离散小波变换（DWT/IWT）在多个 level 上分解/重构特征，并在小波域做 depthwise conv，再与 base depthwise conv 残差融合。

**构造参数**：
- `in_channels`, `out_channels`：要求 **必须相等**（代码 `assert in_channels == out_channels`）
- `kernel_size`：depthwise conv 核大小（默认 5）
- `stride`：输出 stride；若 `stride>1`，最后用 `do_stride` 实现下采样
- `wt_levels`：小波分解层数（默认 1）
- `wt_type`：小波类型，默认 `'db1'`（Haar）

**核心流程**：
1. 对 `curr_x_ll` 做 wavelet transform，得到 `[B, C, 4, H/2, W/2]`（4 对应 LL/LH/HL/HH）。
2. 将其 reshape 为 `[B, C*4, H/2, W/2]` 做 depthwise conv（groups=`C*4`），再 reshape 回小波格式。
3. 逐级 inverse wavelet transform 回到原分辨率。
4. 与 `base_conv` 输出相加：`x = base_conv(x) + x_tag`。

**输入输出形状**：
- 一般情况下：输入 `[B, C, H, W]` 输出 `[B, C, H, W]`
- 若 `stride>1`：输出空间尺寸额外下采样。

**注意点**：
- 依赖 `pywt`。
- 内部会对奇数尺寸做 pad（保证能被 2 整除）。

---

### 4.5 `LGE_W`（LGE 的 WTConv 高频卷积变体，对应论文中的 LGE-W）

**文件**：`LGE_W.py`

**作用**：
- 与 `LGTConvNeckLite` 基本一致；区别在于 `high_conv`：
  - 如果 `c1 == c2`：使用 `WTConv2d_original(c1, c2, kernel_size=3, stride=1)`
  - 否则回退到普通 `Conv(c1, c2, 3, 1)`

**输入输出形状**：
- 输入 `[B, c1, H, W]` 输出 `[B, c2, H, W]`

**YAML 用法（来自你的 YAML）**：
```yaml
- [-1, 1, LGTConvNeckLite_WT, [96, 96, 3, 1, 1]]
```

**注意点 / 可能的坑**：
- 只有 `c1==c2` 时才启用 WTConv；否则用普通卷积，这会导致“模块名里写 WT，但实际没用 WTConv”的情况。写论文时建议明确说明这是“可替换卷积（when possible）”。
- 模块里同样存在 `print(...)`。

---

### 4.6 `RDC` / RepCDC（可重参数化的中心差分卷积）

**文件**：`WDG.py`

**动机**：
- 中心差分卷积（CDC）强调边缘/梯度信息，但直接实现可能带来额外代价。
- 你采用的策略是：训练时用 CDC 形式，推理时把权重合并成普通卷积核（re-parameterization），以保持推理效率。

**接口**：
- `__init__(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)`
- `forward(x)`：
  - deploy 前：使用 CDC 权重（中心元素减去 `theta`）
  - deploy 后：使用 `self.reparam_weight`
- `switch_to_deploy()`：将 CDC 合并后的权重写入 `self.reparam_weight` 并删除训练参数 `weight/theta`。

**张量形状**：
- 输入 `[B, Cin, H, W]` 输出 `[B, Cout, H, W]`

**关键点**：
- 你的 CDC 核构造为：`kernel_final = weight; kernel_final[..., center] -= theta`。
- `theta` 初始为 0，训练后逐步学习差分强度。

---

### 4.7 `WDG`（Wavelet-Difference Gate Block）

**文件**：`WDG.py`

**一句话定义**：
- 在 Bottleneck 内部对特征做 Haar 小波分解：
  - **LL（低频）**：走 `RepCDC` 强化边缘感知
  - **LH/HL/HH（高频）**：融合成一个门控权重，对 LL 特征做加性门控增强
  - 最后 IDWT 重构回原分辨率

**构造参数**：
- `c1`: 输入通道
- `c2`: 输出通道
- `shortcut`: 是否残差（只有 `c1==c2` 且 `shortcut=True` 才 add）
- `g`: 分组参数（当前主要用于上层兼容；本模块内部 gate 用的是 groups=1）
- `e`: expansion 比例（隐层通道 `c_ = int(c2 * e)`）

**内部结构**：
- `cv1`: 1x1 Conv，将 `c1 -> c_`
- `_haar_dwt(x_in)`：手写 Haar 分解（会裁剪到偶数尺寸）
- `cdc_ll`: `RepCDC + BN + SiLU`，作用于 `ll`
- `hf_gate`: `Conv1x1(c_*3 -> c_) + BN + Sigmoid`，作用于 `cat(lh,hl,hh)`
- `_haar_idwt(...)`：手写逆变换（必要时插值回目标尺寸）
- `cv2`: 1x1 Conv，将 `c_ -> c2`

**前向张量形状（关键）**：
- 输入 `x`: `[B, c1, H, W]`
- `x_in = cv1(x)`: `[B, c_, H, W]`
- Haar 分解（内部裁剪到偶数 `H_even, W_even`）：
  - `ll/lh/hl/hh`: `[B, c_, H_even/2, W_even/2]`
- 低频增强：`feat_ll`: `[B, c_, H_even/2, W_even/2]`
- 高频门控：
  - `hf_cat`: `[B, 3*c_, H_even/2, W_even/2]`
  - `feat_gate`: `[B, c_, H_even/2, W_even/2]`
- 交互：`ll_refined = feat_ll * (1 + feat_gate)`
- 逆变换重构：`out`: `[B, c_, H_even, W_even]`，若原输入奇数尺寸，则再插值到 `[B, c_, H, W]`
- 输出投影：`cv2(out)` -> `[B, c2, H, W]`
- 残差：若 `shortcut` 且 `c1==c2`：`return x + out`，否则 `return out`

**可能的坑**：
- `_haar_dwt` 会对奇数尺寸裁剪到偶数，这在某些训练配置下可能引入边界差异；你已在 IDWT 后用插值对齐回原尺寸。

---

### 4.8 `C3_WDG`（将 YOLO 的 C3k2 Bottleneck 替换为 WDG）

**文件**：`WDG.py`

**作用**：
- 保持 Ultralytics 的 `C2f/C3k2` 风格的“分支切片 + 多 bottleneck + concat”骨架。
- 但将内部的 bottleneck 实现替换为 `WDC_Block`。

**构造参数**（与你 YAML 对齐）：
- `c1`: 输入通道
- `c2`: 输出通道
- `n`: bottleneck 重复次数
- `c3k`: 兼容参数（代码中无论 True/False 都用 `WDC_Block`）
- `e`: expansion 比例（决定 hidden channels `self.c=int(c2*e)`）
- `g`: group（传给 `WDC_Block`）
- `shortcut`: 是否残差

**张量形状**：
- 输入 `[B, c1, H, W]`
- 输出 `[B, c2, H, W]`

**前向逻辑**：
- `y = cv1(x).chunk(2,1)` 形成 2 个分支
- 逐个追加 `WDC_Block` 的输出
- concat 后经 `cv2` 输出

---

### 4.9 `FDHead`（Frequency-Driven Head：DEConv + P2 高频门控 + DFL）

**文件**：`FDHead.py`

**定位**：
- 这是你“最像 YOLO Head”的核心改造点之一：把频域门控引入检测头的特征提炼过程，尤其针对小目标更重要的 **P2 分支**。

#### 4.9.1 构造参数
- `nc`: 类别数
- `hidc`: head 隐层通道（你的 YAML 里传了 192）
- `ch`: 输入特征通道列表（由 Ultralytics 的 parse_model 在构建时自动填充）
- `freq_ratio`: 频域通道占比（默认 0.25）
- `gate_reduction`: 门控 MLP 的 reduction ratio（默认 8）

内部派生量：
- `nl = len(ch)`：输入层数（你的 YAML 用了 `[P2,P3]`，所以 `nl=2`）
- `reg_max = 16`，`no = nc + 4*reg_max`
- `c_freq = make_divisible(int(hidc * freq_ratio), 8)` 且 `c_freq <= hidc`

#### 4.9.2 子模块组成
- `self.conv`: 对每个输入尺度做 `Conv_GN(x_i_channels -> hidc, k=1)`
- `self.share_conv`: 一个共享的特征提炼模块：
  - `DEConv_GN(hidc)`：由多种差分卷积权重（cd/hd/vd/ad + normal）组合，支持 `switch_to_deploy()` 合并
  - `DWConv(hidc->hidc,3)` + `Conv(hidc->hidc,1)`
- `self.cv2`: `Conv2d(hidc -> 4*reg_max, 1)`（回归分支）
- `self.cv3`: `Conv2d(hidc -> nc, 1)`（分类分支）
- `self.scale`: 每层一个可学习 scale
- `self.dfl`: `DFL(reg_max)`

P2 的额外频域门控组件（仅当 `c_freq > 0`）：
- `haar_weight`：注册 buffer，形状 `[c_freq*4, 1, 2, 2]`，用于 Haar DWT（stride=2, groups=c_freq）
- `hf_w`: `[3]` 可学习权重（softmax 后作为 LH/HL/HH 的加权系数）
- `hf_gate`: 1x1 两层门控网络：`c_freq -> c_mid -> c_freq`，最后 sigmoid
- `alpha_p2`: 标量门控强度

#### 4.9.3 Haar DWT（P2 内部）
- `_haar_dwt(x)`：
  - 输入 `x`: `[B, c_freq, H, W]`
  - `y = conv2d(x, haar_weight, stride=2, groups=c_freq)` -> `[B, c_freq*4, H/2, W/2]`
  - reshape -> `[B, c_freq, 4, H/2, W/2]`
  - 输出 `ll, lh, hl, hh`，每个 `[B, c_freq, H/2, W/2]`

#### 4.9.4 forward（训练与推理）
输入：`x` 是一个 list，长度 `nl`；每个 `x[i]` 是 `[B, ch[i], Hi, Wi]`。

对每个尺度 i：
1. `xi = Conv_GN(x[i])` -> `[B, hidc, Hi, Wi]`
2. `fi = share_conv(xi)` -> `[B, hidc, Hi, Wi]`
3. 若 `i == 0`（即 P2）且 `c_freq > 0`：
   - 将通道切分：
     - `fa = fi[:, :c_freq]`
     - `fb = fi[:, c_freq:]`
   - Haar DWT 取高频：`_, lh, hl, hh = _haar_dwt(fa)`
   - 高频统计：
     - `w = softmax(hf_w)`
     - `hf = w0*|lh| + w1*|hl| + w2*|hh|`
   - 全局池化：`s = hf.mean((2,3), keepdim=True)` -> `[B, c_freq, 1, 1]`
   - 门控：`g = hf_gate(s)` -> `[B, c_freq, 1, 1]`
   - 增强：`fa = fa * (1 + alpha_p2 * g)`
   - 拼回回归特征：`reg_f = cat(fa, fb, dim=1)` -> `[B, hidc, Hi, Wi]`
   - 回归：`box = scale[i](cv2(reg_f))` -> `[B, 4*reg_max, Hi, Wi]`
   - 分类：`cls = cv3(fi)` -> `[B, nc, Hi, Wi]`
4. 其它层（或 i!=0）：
   - `box = scale[i](cv2(fi))`
   - `cls = cv3(fi)`
5. 每层输出拼接：`x[i] = cat(box, cls, dim=1)` -> `[B, no, Hi, Wi]`

训练模式：返回 list `x`（每层 `[B,no,Hi,Wi]`）。

推理模式：
- 将各层展平后 concat 成 `x_cat: [B, no, Σ(Hi*Wi)]`
- 生成 anchors/strides（Ultralytics 逻辑）
- `box, cls` 切分
- `dbox = decode_bboxes(box)`（内部使用 DFL + dist2bbox）
- 输出 `y = cat(dbox, sigmoid(cls))`

#### 4.9.5 deploy（推理加速）
- `DEConv_GN.switch_to_deploy()` 会把多分支差分卷积权重合并写入 `conv1_5` 并删除其它卷积分支，从而减少推理开销。

---

## 5. 建议的短命名方案（旧名 -> 新名）

你当前模块名非常长，确实不利于写论文/画结构图/给别人复现。下面给出一套“短但可读”的命名规则与映射。

### 5.1 命名规则（建议）
- 频域相关统一前缀：
  - `LG*` 表示 Log-Gabor 系列
  - `WD*` 表示 Wavelet-Difference 系列
  - `WT*` 表示 Wavelet Transform Convolution（WTConv）
  - `FG*` 表示 Frequency-Gated / Frequency-Guided 系列
- 尽量保持 3~10 个字符：适合写 YAML 和画图。
- 避免用过于抽象的单字母，保证“看到名字能猜出作用”。

### 5.2 模块映射表（推荐）
- `LogGaborFilterLite` -> `LGF`
  - 含义：Log-Gabor Filter
- `LGTConvNeckLite` -> `LGN`
  - 含义：Log-Gabor Neck
- `LGTConvNeckLiteV2` -> `LGN2`
- `LGTConvNeckUltraLite` -> `LGNU`

- `WTConv2d_original` -> `WTConv`
  - 若你后续还有改进版，可写成 `WTConvV1/WTConvV2`

- `LGTConvNeckLite_WT` -> `LGNWT`
  - 含义：Log-Gabor Neck + WTConv

- `RepCDC` -> `rCDC`
  - 含义：re-parameterized CDC

- `WDC_Block` -> `WDG`
  - 含义：Wavelet-Diff Gate（强调“高频门控低频”）

- `C3k2_WDC` -> `C3WD`
  - 含义：C3k2 with WDC

- `Detect_LSDECD_DWTP2GateLite` -> `FGDHead`
  - 含义：Frequency-Gated Detect Head
  - 如果你论文必须保留“P2 Gate”特征：也可以叫 `FGDHeadP2`

### 5.3 论文写作建议（命名如何落地）
- 在论文图中：用短名（例如 `LGN`、`C3WD`、`FGDHead`）。
- 在正文首次出现：
  - “LGN（Log-Gabor Neck Lite, implemented as `LGTConvNeckLite`）”
  - “C3WD（C3k2 with Wavelet Difference Convolution, implemented as `C3k2_WDC`）”
- 在开源代码里：
  - 你可以保留原类名以兼容旧 YAML，同时在同一个文件里再提供一个 alias（新类名继承旧类名或直接变量别名）。
  - 但是否要做“全仓库重命名”，建议你先确认：
    - 训练脚本/parse_model 是通过 `globals()`、`eval()` 还是显式 registry 找类。
    - 否则贸然改名可能导致 YAML 找不到模块。

---

## 6. 最小集成示例（只说明结构，不包含训练脚本）

### 6.1 YAML 片段示例
- `C3WD`（即 `C3k2_WDC`）用于 backbone：
```yaml
- [-1, 2, C3k2_WDC, [256, False, 0.25]]
```

- `LGN`（即 `LGTConvNeckLite`）用于 neck：
```yaml
- [-1, 1, nn.Upsample, [None, 2, "nearest"]]
- [-1, 1, LGTConvNeckLite, [128, 128, 3, 1, 1]]
- [[-1, 4], 1, Concat, [1]]
```

- `FGDHead`（即 `Detect_LSDECD_DWTP2GateLite`）用于检测头：
```yaml
- [[22, 25], 1, Detect_LSDECD_DWTP2GateLite, [nc, 192]]
```

---

## 7. 常见问题（排错清单）

- **导入失败：pywt / einops 不存在**
  - `WTConv_original.py` 需要 `pywt`
  - `Detect_LSDECD_DWTP2GateLite.py` 需要 `einops`

- **训练日志刷屏**
  - `LGTConvNeckLite` / `LGTConvNeckLite_WT` / `WTConv_original` 内有多处 `print`。

- **输入尺寸为奇数导致的边界差异**
  - `WDC_Block` 的 Haar DWT 会裁剪到偶数，再在输出时插值回去。

- **`LGTConvNeckLite_WT` 未必真的启用 WTConv**
  - 只有 `c1==c2` 才启用 `WTConv2d_original`。

---

## 8. 当前进度说明

本文件已覆盖 `corecode/` 下所有自定义模块与 YAML 引用到的改动点。
如果你的真实训练工程（parse_model/registry/ultralytics 改动）在 `corecode` 之外，请把对应目录也加入工作区，否则我无法进一步把“模块如何被注册与 YAML 解析”这一层补齐到文档中。
