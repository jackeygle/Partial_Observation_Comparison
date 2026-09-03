"""
model.py — DINCAE 2.0 的网络结构（PyTorch 重写）
=================================================

对照参考实现 `reference/DINCAE.jl/src/model.jl`。**照抄的是 2.0 的结构，不是 1.0 论文的
Table 1**——两者差别很大，2.0 §4 给了理由：

  * 1.0 的**全连接瓶颈 + dropout 去掉了**。理由是 FC 要求训练输入矩阵和推理输入矩阵尺寸
    完全一致，大区域没法用（2.0 §4，引 Long et al. 2015 FCN）。代码里
    `dropout_rate_train = 0.3` 是注释掉的。
  * skip connection 从 **concat 改成加法**（SumSkip，2.0 Eq.2）。2.0 §5.1：在 1.0 的同一个
    Ligurian 算例上 0.3835 -> **0.3604**。机制是残差网络那套（He et al. 2016），缓解梯度消失。
  * pooling 用 **MeanPool**（`model.jl:268` 硬编码）。注意 2.0 正文 Table 1 写的是 max
    pooling —— **论文与代码不一致**。这里跟代码走（1.0 的消融也支持 avg：0.3835 vs 0.3900），
    但这条在 2.0 里没被重新验证过，`--pool max` 可以 A/B。
  * **精化步**（refinement，2.0 §2.2 Eq.4）：第二个同结构网络吃 `cat(第一级的原始输出, 输入)`，
    权重不共享，损失对**每一级输出**都算并加权（默认 α=0.3, α'=0.7）。2.0 Table 2：
    0.60 -> **0.55**，比多喂物理辅助场（只值 0.03）大得多。

输出参数化（2.0 Eq.6-7；`model.jl:16-29`）：网络第一片输出的**不是均值，而是 m/σ²**
（信息形式），与输入/目标同一个坐标系。

    invσ² = exp(min(x₂, γ));   σ² = 1/max(invσ², µ);   m = x₁·σ²

γ = log(min_std_err⁻²) = 10，µ = 1e-3（对应 σ 的有效区间 0.0067~31.6）。min/max 只在训练
最初几个 epoch 权重还接近随机时起作用。

尺寸：ATC 走廊网格 36×12，3 级池化 -> 18×6 -> 9×3 -> 5×2。参考 2.0 的 altimetry 算例
（177×69 -> 3 级 -> 23×9，滤波器 32/64/96）。奇数尺寸靠 `ceil_mode` 池化 + 上采样后裁剪
处理，等价于参考实现的 `sz_small = sz÷2 + odd` 与 `croppadding`。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_STD_ERR = 0.006737946999085467          # exp(-5)，参考实现的默认值
GAMMA = float(torch.log(torch.tensor(MIN_STD_ERR ** -2)))       # = 10.0
MU = 1e-3


def transform_msigma2(x, gamma: float = GAMMA, mu: float = MU):
    """(B, 2*nvar, H, W) 的原始输出 -> (mean, σ²) 各 nvar 通道。

    参考 `model.jl:16-41`：每个输出变量占**两片**，第 1 片是 m/σ²，第 2 片是 log(1/σ²)。
    """
    assert x.shape[1] % 2 == 0, "输出通道数必须是 2×变量数"
    x1 = x[:, 0::2]                                     # m/σ²
    x2 = x[:, 1::2]                                     # log(1/σ²)
    inv_s2 = torch.exp(torch.clamp(x2, max=gamma))
    s2 = 1.0 / torch.clamp(inv_s2, min=mu)
    return x1 * s2, s2


class _Level(nn.Module):
    """一级编码-解码块，等价于 `recmodel4` 里的 `inner` chain（可选 SumSkip）。

        x -> conv(enc_in->enc_out) -> ReLU -> pool2 -> inner -> up2 -> crop
          -> conv(dec_in->dec_out) -> [ReLU]
        若 skip: 返回 x + 上式（要求 enc_in == dec_out）
    """

    def __init__(self, enc_in, enc_out, dec_in, dec_out, inner,
                 out_relu=True, skip=False, pool="mean"):
        super().__init__()
        self.down = nn.Conv2d(enc_in, enc_out, 3, padding=1)
        self.inner = inner
        self.up = nn.Conv2d(dec_in, dec_out, 3, padding=1)
        self.out_relu = out_relu
        self.pool = pool
        self.skip = skip
        if skip and enc_in != dec_out:
            raise ValueError(f"SumSkip 要求 enc_in==dec_out，得到 {enc_in} vs {dec_out}")

    def forward(self, x):
        h = F.relu(self.down(x))
        hw = h.shape[-2:]                                     # 记住池化前的尺寸，供裁剪用
        if self.pool == "mean":
            h = F.avg_pool2d(h, 2, ceil_mode=True, count_include_pad=False)
        else:
            h = F.max_pool2d(h, 2, ceil_mode=True)
        h = self.inner(h)
        h = F.interpolate(h, scale_factor=2, mode="nearest")   # 参考实现默认 :nearest
        h = h[..., : hw[0], : hw[1]]                           # = croppadding(x, odd)
        h = self.up(h)
        if self.out_relu:
            h = F.relu(h)
        return x + h if self.skip else h


def build_unet(n_in, n_out_raw, enc_internal=(32, 64, 96), skip_levels=None, pool="mean"):
    """按 `recmodel4` 的递归结构搭一个 U-Net。

    enc = [n_in, *enc_internal]，dec = [n_out_raw, *enc_internal]。第 l 级：
    conv enc[l]->enc[l+1]，池化，递归，上采样，conv dec[l+1]->dec[l]。最内层是 identity。
    最外层（l=1）输出层不加激活（参考实现 `f = l==1 ? identity : relu`）且不加 skip
    （n_in != n_out_raw）。
    """
    enc = [n_in, *enc_internal]
    dec = [n_out_raw, *enc_internal]
    L = len(enc)                                    # 1..L-1 建块，第 L 级是 identity
    if skip_levels is None:
        skip_levels = range(2, L + 1)               # 参考默认 2:(len(enc_internal)+1)

    net: nn.Module = nn.Identity()
    for l in range(L - 1, 0, -1):                   # 由内向外搭
        net = _Level(enc_in=enc[l - 1], enc_out=enc[l],
                     dec_in=dec[l], dec_out=dec[l - 1],
                     inner=net, out_relu=(l != 1),
                     skip=(l in skip_levels and l != 1), pool=pool)
    return net


class DINCAE(nn.Module):
    """DINCAE 2.0：U-Net（+ 可选精化步）。

    forward 返回**每一级**的 (mean, σ²)，因为损失要对每一级都算（2.0 Eq.4）。推理时用
    最后一级。精化级的输入是 `cat(上一级的原始输出, 原始输入)`（`model.jl:182-189`）。
    """

    def __init__(self, n_in, n_var, enc_internal=(32, 64, 96),
                 loss_weights=(0.3, 0.7), pool="mean", gamma=GAMMA, mu=MU):
        super().__init__()
        self.n_var = n_var
        self.n_out_raw = 2 * n_var
        self.loss_weights = tuple(loss_weights)
        self.gamma, self.mu = gamma, mu
        nets = [build_unet(n_in, self.n_out_raw, enc_internal, pool=pool)]
        for _ in range(len(self.loss_weights) - 1):            # 精化级：输入多了上一级的输出
            nets.append(build_unet(n_in + self.n_out_raw, self.n_out_raw,
                                   enc_internal, pool=pool))
        self.nets = nn.ModuleList(nets)

    def forward(self, xin):
        outs = []
        raw = self.nets[0](xin)
        outs.append(transform_msigma2(raw, self.gamma, self.mu))
        for net in self.nets[1:]:
            raw = net(torch.cat([raw, xin], dim=1))
            outs.append(transform_msigma2(raw, self.gamma, self.mu))
        return outs

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
