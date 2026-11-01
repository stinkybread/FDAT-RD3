#!/usr/bin/env python3
"""FDAT-RD3 converter (self-contained; FDATRD3 architecture embedded verbatim below).

Accepts FDAT-RD3, FDAT-RD2 and FDAT-RD 2x-unshuffle checkpoints and writes FDAT-RD3 ONNX / weights.

Usage:
    python fdat_rd3_converter.py model.safetensors                     # static ONNX 540x720 -> 1080x1440, verified
    python fdat_rd3_converter.py model.safetensors --input-size 1080 1440
    python fdat_rd3_converter.py model.safetensors --attn-scale pre     # previous attention form, for A/B
    python fdat_rd3_converter.py rd2_model.safetensors -f safetensors   # port RD/RD2 weights to an FDAT-RD3 checkpoint
    python fdat_rd3_converter.py model.safetensors --info
"""
# --- FDATRD3 architecture (verbatim from fdat_rd3_arch.py) ---
# ruff: noqa
# type: ignore
"""FDAT-RD3 (FDATRD3): FDAT-RD2's network with an inference-optimised graph.

Same function and the same weights as FDAT-RD2 (or a 2x unshuffle FDAT-RD), reformulated
where TensorRT 11 BF16 engines were losing time:

  1. Upsampler (x4 "transpose+conv" tail). VapourKit's BF16 builder had to keep both
     ConvTranspose2d layers in FP32 (no BF16 tactic), costing ~6.6 ms of a 61 ms 540x720 frame
     plus FP32<->BF16 casts around the GELU. Each ConvTranspose2d(k=4, s=2, p=1) is computed
     exactly as ordinary convolutions + a manual depth-to-space:
       * "phase": four 2x2 convs, one per output sub-pixel phase (same MACs as the deconv);
         used when the output has many channels (the 128->128 layer).
       * "dense": one 3x3 conv to 4*Cout channels (2.25x MACs, one read of the input);
         used when the output has few channels (the 128->3 layer).
     Parameters stay in ConvTranspose2d layout, so checkpoints, init and training dynamics
     are unchanged.
  2. Channel attention. Instead of normalising q/k over all tokens, transposing, two
     matmuls and a projection on full-resolution tensors, FDATRD3 forms one Gram matrix of
     [q|k] over the tokens (2C x 2C), derives the cosine similarities and per-head softmax
     on that tiny matrix, and folds attention, value projection and output projection into
     one C x C weight applied with a single matmul. v is never materialised.
     Mathematically identical to FDAT-RD's FastChannelAttention (incl. F.normalize eps).

Everything else (pre-resize, unshuffle stem, spatial window attention, token dictionary,
AIM gates, FFN, groups) is FDAT-RD/RD2 verbatim, except pixel (un)shuffle is done with
explicit reshape/permute (nn.PixelUnshuffle mis-exports to ONNX).

Stems
  stem="resize"    (FDAT-RD2 layout): LR -> fixed zimg-exact 2x resize -> unshuffle 4 -> body
  stem="unshuffle" (FDAT-RD 2x unshuffle layout): LR -> unshuffle 2 -> body
Both run the body at LR/2 and upsample x4, output = 2x LR.

Checkpoints
  FDATRD3 state dict = FDAT-RD/RD2 keys, minus 'upsampler.MetaUpsample', plus 'MetaFDATRD3'
  ([3, stem_id, version]). FDAT-RD and FDAT-RD2 spandrel detectors require
  'upsampler.MetaUpsample', so they never claim an FDATRD3 file; FDATRD3 requires 'MetaFDATRD3',
  so it never claims theirs. FDATRD3.load_state_dict accepts RD2 and RD 2x-unshuffle
  checkpoints directly (marker conversion only, no weight changes).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint

RD3_META_KEY = "MetaFDATRD3"
RD3_VERSION = 1
STEMS = ("resize", "unshuffle")
RESIZE_META_KEY = "pre.MetaResize"


# ============================================================================================
# Helpers
# ============================================================================================

def pixel_unshuffle(x: Tensor, r: int) -> Tensor:
    """== F.pixel_unshuffle, as explicit reshape/permute (clean ONNX/TensorRT)."""
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // r, r, w // r, r).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * r * r, h // r, w // r)


def pixel_shuffle(x: Tensor, r: int) -> Tensor:
    """== F.pixel_shuffle, as explicit reshape/permute."""
    b, c, h, w = x.shape
    oc = c // (r * r)
    x = x.reshape(b, oc, r, r, h, w).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, oc, h * r, w * r)


class PixelUnshuffle(nn.Module):
    def __init__(self, r: int) -> None:
        super().__init__()
        self.r = r

    def forward(self, x: Tensor) -> Tensor:
        return pixel_unshuffle(x, self.r)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x.div(keep) * (keep + torch.rand(shape, dtype=x.dtype, device=x.device)).floor_()


# ONNX export form of attention scaling (training always uses SDPA):
#   "post": softmax((q @ k^T) * scale [+ bias]) @ v  -- canonical order TensorRT's fused-attention
#           pattern matching expects (default)
#   "pre":  softmax((q * scale) @ k^T [+ bias]) @ v  -- FDAT-RD/RD2 converters' form
ATTN_EXPORT_SCALE = "post"


def _attention(q: Tensor, k: Tensor, v: Tensor, bias: Tensor | None = None, scale: float = 1.0) -> Tensor:
    if torch.onnx.is_in_onnx_export():
        if ATTN_EXPORT_SCALE == "pre":
            a = (q * scale) @ k.transpose(-2, -1)
        else:
            a = (q @ k.transpose(-2, -1)) * scale
        if bias is not None:
            a = a + bias
        return F.softmax(a, dim=-1) @ v
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale)


# ============================================================================================
# Fixed 2x pre-resize (FDAT-RD2, verbatim math)
# ============================================================================================

RESIZE_KERNELS = ("bicubic", "spline16", "spline36", "spline64", "lanczos")
RESIZE_ALIASES = {"mitchell": ("bicubic", 1 / 3, 1 / 3), "catmull_rom": ("bicubic", 0.0, 0.5)}


def resolve_resize(kernel="bicubic", b=0.0, c=0.5, taps=3):
    if kernel in RESIZE_ALIASES:
        kernel, b, c = RESIZE_ALIASES[kernel]
    if kernel not in RESIZE_KERNELS:
        raise ValueError(f"resize kernel must be one of {RESIZE_KERNELS + tuple(RESIZE_ALIASES)}, got {kernel!r}")
    if kernel == "lanczos":
        if int(taps) != taps or not 1 <= taps <= 8:
            raise ValueError(f"lanczos taps must be an integer in 1..8, got {taps}")
        return kernel, 0.0, 0.0, int(taps)
    if kernel != "bicubic":
        b, c = 0.0, 0.0
    return kernel, float(b), float(c), 0


def _resize_kernel(kernel, b=0.0, c=0.5, taps=3):
    if kernel == "bicubic":
        def k(x):
            x = abs(x)
            if x < 1:
                return ((12 - 9 * b - 6 * c) * x**3 + (-18 + 12 * b + 6 * c) * x**2 + (6 - 2 * b)) / 6
            if x < 2:
                return ((-b - 6 * c) * x**3 + (6 * b + 30 * c) * x**2 + (-12 * b - 48 * c) * x + (8 * b + 24 * c)) / 6
            return 0.0
        return 2, k
    if kernel == "spline16":
        def k(x):
            x = abs(x)
            if x < 1:
                return ((x - 9 / 5) * x - 1 / 5) * x + 1
            if x < 2:
                x -= 1
                return ((-1 / 3 * x + 4 / 5) * x - 7 / 15) * x
            return 0.0
        return 2, k
    if kernel == "spline36":
        def k(x):
            x = abs(x)
            if x < 1:
                return ((13 / 11 * x - 453 / 209) * x - 3 / 209) * x + 1
            if x < 2:
                x -= 1
                return ((-6 / 11 * x + 270 / 209) * x - 156 / 209) * x
            if x < 3:
                x -= 2
                return ((1 / 11 * x - 45 / 209) * x + 26 / 209) * x
            return 0.0
        return 3, k
    if kernel == "spline64":
        def k(x):
            x = abs(x)
            if x < 1:
                return ((49 / 41 * x - 6387 / 2911) * x - 3 / 2911) * x + 1
            if x < 2:
                x -= 1
                return ((-24 / 41 * x + 4032 / 2911) * x - 2328 / 2911) * x
            if x < 3:
                x -= 2
                return ((6 / 41 * x - 1008 / 2911) * x + 582 / 2911) * x
            if x < 4:
                x -= 3
                return ((-1 / 41 * x + 168 / 2911) * x - 97 / 2911) * x
            return 0.0
        return 4, k
    if kernel == "lanczos":
        def sinc(x):
            return 1.0 if x == 0 else math.sin(math.pi * x) / (math.pi * x)

        def k(x):
            x = abs(x)
            return sinc(x) * sinc(x / taps) if x < taps else 0.0
        return int(taps), k
    raise ValueError(f"unknown resize kernel {kernel!r}")


def resize2x_weight(ch=3, kernel="bicubic", b=0.0, c=0.5, taps=3, dtype=torch.float32):
    kernel, b, c, taps = resolve_resize(kernel, b, c, taps)
    r, k = _resize_kernel(kernel, b, c, taps)
    ph = torch.tensor([[k(o - p) for o in range(-r, r + 1)] for p in (-0.25, 0.25)], dtype=torch.float64)
    ph = ph / ph.sum(1, keepdim=True)
    w = torch.einsum("ai,bj->abij", ph, ph).reshape(4, 1, 2 * r + 1, 2 * r + 1)
    return w.repeat(ch, 1, 1, 1).to(dtype)


def _sym_pad(x, r):
    for dim in (-2, -1):
        n = x.shape[dim]
        idx = torch.cat([torch.arange(r - 1, -1, -1), torch.arange(n), torch.arange(n - 1, n - 1 - r, -1)])
        x = x.index_select(dim, idx.clamp(0, n - 1).to(x.device))
    return x


def resize_meta(kernel="bicubic", b=0.0, c=0.5, taps=3):
    kernel, b, c, taps = resolve_resize(kernel, b, c, taps)
    p1, p2 = (float(taps), 0.0) if kernel == "lanczos" else (b, c)
    return torch.tensor([2.0, float(RESIZE_KERNELS.index(kernel)), p1, p2], dtype=torch.float32)


def normalize_resize_meta(t):
    t = t.detach().float().flatten().cpu()
    if t.numel() == 3:
        return torch.tensor([float(t[0]), 0.0, float(t[1]), float(t[2])], dtype=torch.float32)
    if t.numel() != 4:
        raise ValueError(f"bad {RESIZE_META_KEY} marker of length {t.numel()}")
    return t


def resize_kwargs_from_meta(t):
    t = normalize_resize_meta(t)
    kernel = RESIZE_KERNELS[int(round(float(t[1])))]
    if kernel == "lanczos":
        return dict(resize_kernel=kernel, bicubic_b=0.0, bicubic_c=0.0, lanczos_taps=int(round(float(t[2]))))
    b, c = float(t[2]), float(t[3])
    for _, ab, ac in RESIZE_ALIASES.values():  # marker is float32: snap back to the exact 1/3 etc.
        if abs(b - ab) < 1e-6 and abs(c - ac) < 1e-6:
            b, c = ab, ac
    return dict(resize_kernel=kernel, bicubic_b=b, bicubic_c=c, lanczos_taps=3)


def describe_resize(t):
    kw = resize_kwargs_from_meta(t)
    kern = kw["resize_kernel"]
    if kern == "lanczos":
        return f"lanczos{kw['lanczos_taps']}"
    if kern == "bicubic":
        for name, (_, b, c) in RESIZE_ALIASES.items():
            if abs(kw["bicubic_b"] - b) < 1e-6 and abs(kw["bicubic_c"] - c) < 1e-6:
                return name
        return f"bicubic(b={kw['bicubic_b']:g},c={kw['bicubic_c']:g})"
    return kern


class FixedResize2x(nn.Module):
    """Fixed, deterministic, center-aligned 2x resize; bit-matches zimg / VapourSynth resize.*."""

    def __init__(self, ch=3, kernel="bicubic", b=0.0, c=0.5, taps=3):
        super().__init__()
        kernel, b, c, taps = resolve_resize(kernel, b, c, taps)
        self.register_buffer("w", resize2x_weight(ch, kernel, b, c, taps), persistent=False)
        self.register_buffer("MetaResize", resize_meta(kernel, b, c, taps))

    def forward(self, x):
        r = (self.w.shape[-1] - 1) // 2
        y = F.conv2d(_sym_pad(x, r), self.w.to(x.dtype), groups=x.shape[1])
        return pixel_shuffle(y, 2)


# ============================================================================================
# Upsampler: ConvTranspose2d(k=4, s=2, p=1) computed exactly with ordinary convolutions
# ============================================================================================

# Output sub-pixel phase a in {0,1}, input offset d: kernel tap k = a + 1 - 2d.
#   phase-split (2 taps):  a=0 uses input rows (i-1, i) -> taps (3, 1); a=1 uses (i, i+1) -> taps (2, 0)
#   dense 3x3 (3 offsets): offsets (i-1, i, i+1); index 4 = zero tap
_PHASE_TAPS = ((3, 1), (2, 0))
_DENSE_TAPS = ((3, 1, 4), (4, 2, 0))


class ConvTranspose4x2(nn.ConvTranspose2d):
    """Exact ConvTranspose2d(cin, cout, 4, 2, 1) evaluated as convs + depth-to-space.
    Parameters, init and state-dict keys are those of nn.ConvTranspose2d."""

    def __init__(self, cin: int, cout: int, impl: str = "auto") -> None:
        super().__init__(cin, cout, 4, 2, 1)
        if impl == "auto":
            impl = "dense" if cout * 4 <= 64 else "phase"
        if impl not in ("phase", "dense"):
            raise ValueError(f"impl must be auto|phase|dense, got {impl!r}")
        self.impl = impl
        self.register_buffer("_phase_idx", torch.tensor(_PHASE_TAPS), persistent=False)
        self.register_buffer("_dense_idx", torch.tensor(_DENSE_TAPS), persistent=False)

        self._frozen = False

    def _conv_params(self):
        """Derived conv weights/biases from the ConvTranspose2d parameters."""
        wt, bias = self.weight, self.bias  # (cin, cout, 4, 4), (cout,)
        cout = self.out_channels
        if self.impl == "dense":
            we = F.pad(wt, (0, 1, 0, 1))                          # tap index 4 = 0
            k = we[:, :, self._dense_idx][..., self._dense_idx]    # (cin, cout, a, di, b, dj)
            k = k.permute(1, 2, 4, 0, 3, 5).reshape(cout * 4, -1, 3, 3)
            return [k], [None if bias is None else bias.repeat_interleave(4)]
        ks = [wt[:, :, self._phase_idx[a]][..., self._phase_idx[c]].transpose(0, 1).contiguous()
              for a in range(2) for c in range(2)]                 # (cout, cin, 2, 2) per phase
        return ks, [bias] * 4

    @torch.no_grad()
    def freeze(self) -> None:
        """Bake the derived conv kernels into buffers (ONNX export needs static kernels).
        Call after the final weights are loaded; inference-only afterwards."""
        ks, bs = self._conv_params()
        for i, (k, b) in enumerate(zip(ks, bs)):
            self.register_buffer(f"_fk{i}", k.detach().clone(), persistent=False)
            self.register_buffer(f"_fb{i}", None if b is None else b.detach().clone(), persistent=False)
        self._frozen = True

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        cout = self.out_channels
        if self._frozen:
            n = 1 if self.impl == "dense" else 4
            ks = [getattr(self, f"_fk{i}") for i in range(n)]
            bs = [getattr(self, f"_fb{i}") for i in range(n)]
        else:
            ks, bs = self._conv_params()
        ks = [k.to(x.dtype) for k in ks]
        bs = [None if v is None else v.to(x.dtype) for v in bs]
        if self.impl == "dense":
            return pixel_shuffle(F.conv2d(x, ks[0], bs[0], padding=1), 2)
        xp = F.pad(x, (1, 1, 1, 1))
        phases = []
        for a in range(2):
            for c in range(2):
                xs = xp[:, :, a:a + h + 1, c:c + w + 1]  # phase 0 starts at input row -1
                phases.append(F.conv2d(xs, ks[a * 2 + c], bs[a * 2 + c]))
        y = torch.stack(phases, dim=2)                             # (b, cout, 4, h, w), phase = a*2 + c
        return pixel_shuffle(y.reshape(b, cout * 4, h, w), 2)


class Upsampler4x(nn.Module):
    """FDAT-RD 'transpose+conv' x4 tail, keys upsampler.{0,2,3}: ConvT, GELU, ConvT, Conv3x3."""

    def __init__(self, embed_dim: int, num_out_ch: int) -> None:
        super().__init__()
        self.add_module("0", ConvTranspose4x2(embed_dim, embed_dim))
        self.add_module("1", nn.GELU())
        self.add_module("2", ConvTranspose4x2(embed_dim, num_out_ch))
        self.add_module("3", nn.Conv2d(num_out_ch, num_out_ch, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        for m in self._modules.values():
            x = m(x)
        return x


# ============================================================================================
# Body (FDAT-RD, verbatim except FastChannelAttention)
# ============================================================================================

class TokenDictionaryCrossAttention(nn.Module):
    def __init__(self, dim, num_tokens=128, num_heads=4, qkv_bias=False) -> None:
        super().__init__()
        self.nh = num_heads
        self.hd = dim // num_heads
        self.scale = self.hd ** -0.5
        self.dictionary = nn.Parameter(torch.zeros(num_tokens, dim))
        trunc_normal_(self.dictionary, std=0.02)
        self.r_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.r_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.r_norm = nn.LayerNorm(dim)
        self.e_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.e_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def _mha(self, q, k, v):
        B, Lq, C = q.shape
        Lk = k.shape[1]
        q = q.reshape(B, Lq, self.nh, self.hd).permute(0, 2, 1, 3)
        k = k.reshape(B, Lk, self.nh, self.hd).permute(0, 2, 1, 3)
        v = v.reshape(B, Lk, self.nh, self.hd).permute(0, 2, 1, 3)
        return _attention(q, k, v, None, self.scale).permute(0, 2, 1, 3).reshape(B, Lq, C)

    def forward(self, x, H, W):
        B, N, C = x.shape
        d = self.dictionary.unsqueeze(0).expand(B, -1, -1)
        rk, rv = self.r_kv(self.r_norm(x)).chunk(2, dim=-1)
        d = d + self._mha(self.r_q(d), rk, rv)
        ek, ev = self.e_kv(d).chunk(2, dim=-1)
        return self.proj(self._mha(self.e_q(x), ek, ev))


class FastSpatialWindowAttention(nn.Module):
    def __init__(self, dim, window_size=(16, 16), num_heads=4, qkv_bias=False) -> None:
        super().__init__()
        self.wh, self.ww = window_size
        self.nh = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        n = self.wh * self.ww
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, n, n))
        trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self, x, H, W):
        B, L, C = x.shape
        wh, ww = self.wh, self.ww
        pad_b = (wh - H % wh) % wh
        pad_r = (ww - W % ww) % ww
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x.reshape(B, H, W, C), (0, 0, 0, pad_r, 0, pad_b)).reshape(B, -1, C)
        Hp, Wp = H + pad_b, W + pad_r
        x = (x.reshape(B, Hp // wh, wh, Wp // ww, ww, C)
             .permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, wh * ww, C))
        N = wh * ww
        q, k, v = self.qkv(x).reshape(-1, N, 3, self.nh, C // self.nh).permute(2, 0, 3, 1, 4).unbind(0)
        x = _attention(q, k, v, self.rel_pos_bias.unsqueeze(0).to(q.dtype), self.scale)
        x = x.transpose(1, 2).reshape(-1, N, C)
        x = (self.proj(x).reshape(B, Hp // wh, Wp // ww, wh, ww, C)
             .permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, Hp, Wp, C))
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x.reshape(B, L, C)


class FastChannelAttention(nn.Module):
    """FDAT-RD channel (transposed) attention, Gram-folded. Same parameters and function:

        q, k, v = split(qkv(x));  qn, kn = normalize(q, k over tokens, per channel)
        y = proj( per-head softmax(temp * qn^T kn) applied to v )

    computed as  S = [q|k]^T [q|k]  (one 2C x 2C Gram over tokens), cos = S_qk / (|q| |k|^T),
    A = blockdiag_h softmax(temp_h * cos_h),  y = x @ (W_p A W_v)^T + (W_p A b_v + b_p).
    """

    def __init__(self, dim, num_heads=4, qkv_bias=False) -> None:
        super().__init__()
        self.nh = num_heads
        self.temp = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        hd = dim // num_heads
        head = torch.arange(dim) // hd
        self.register_buffer("_head_of", head, persistent=False)
        block = head[:, None] == head[None, :]
        self.register_buffer("_offblock", torch.zeros(dim, dim).masked_fill(~block, -1e4), persistent=False)
        self.register_buffer("_eye2", torch.eye(2 * dim), persistent=False)

    def forward(self, x, H, W):
        B, N, C = x.shape
        w, bias = self.qkv.weight, self.qkv.bias
        # q,k pre-scaled by N^-1/2 (folded into the weight): the Gram becomes a mean over tokens,
        # FP16-safe at full-frame token counts; cosine similarities are scale-invariant.
        s = float(N) ** -0.5
        qk = F.linear(x, w[: 2 * C] * s, None if bias is None else bias[: 2 * C] * s)  # (B, N, 2C)
        S = qk.transpose(1, 2) @ qk                                          # (B, 2C, 2C)
        sq = (S * self._eye2.to(S.dtype)).sum(-1)                            # (B, 2C) = mean_n x^2
        nrm = torch.sqrt(sq).clamp_min(1e-12 * s)                            # F.normalize eps (scaled)
        cos = S[:, :C, C:] / (nrm[:, :C, None] * nrm[:, None, C:])           # (B, C, C)
        t = self.temp.reshape(-1)[self._head_of].to(cos.dtype)               # (C,) per-row temp
        A = F.softmax(cos * t[:, None] + self._offblock.to(cos.dtype), dim=-1)  # block-diag (B, C, C)
        PA = self.proj.weight.to(A.dtype) @ A                                # (B, C, C)
        W_eff = PA @ w[2 * C:].to(A.dtype)                                   # (B, C, C)
        b_eff = self.proj.bias.to(A.dtype).expand(B, C)
        if bias is not None:
            b_eff = b_eff + (PA @ bias[2 * C:].to(A.dtype)[:, None]).squeeze(-1)
        return x @ W_eff.transpose(1, 2) + b_eff[:, None, :]


class SimplifiedAIM(nn.Module):
    def __init__(self, dim, reduction_ratio=8) -> None:
        super().__init__()
        self.sg = nn.Sequential(nn.Conv2d(dim, 1, 1, bias=False), nn.Sigmoid())
        self.cg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim // reduction_ratio, 1, bias=False),
            nn.GELU(), nn.Conv2d(dim // reduction_ratio, dim, 1, bias=False), nn.Sigmoid())

    def forward(self, attn_feat, conv_feat, interaction_type, H, W):
        B, L, C = attn_feat.shape
        if interaction_type == "spatial_modulates_channel":
            sm = self.sg(attn_feat.transpose(1, 2).reshape(B, C, H, W)).reshape(B, 1, L).transpose(1, 2)
            return attn_feat + (conv_feat * sm)
        cm = self.cg(conv_feat.transpose(1, 2).reshape(B, C, H, W)).reshape(B, C, 1).transpose(1, 2)
        return (attn_feat * cm) + conv_feat


class SimplifiedFFN(nn.Module):
    def __init__(self, dim, expansion_ratio=2.0, drop=0.0) -> None:
        super().__init__()
        hd = int(dim * expansion_ratio)
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hd, False), nn.GELU(), nn.Linear(hd, dim, False)
        self.drop = nn.Dropout(drop)
        self.smix = nn.Conv2d(hd, hd, 3, 1, 1, groups=hd, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.drop(self.act(self.fc1(x)))
        x_s = self.smix(x.transpose(1, 2).reshape(B, x.shape[-1], H, W)).reshape(B, x.shape[-1], L).transpose(1, 2)
        return self.drop(self.fc2(x_s))


class SimplifiedDATBlock(nn.Module):
    def __init__(self, dim, nh, window_size, ffn_exp, aim_re, btype, dp, num_dict_tokens=128, qkv_b=False) -> None:
        super().__init__()
        self.btype = btype
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        if btype == "spatial":
            self.attn = FastSpatialWindowAttention(dim, window_size, nh, qkv_b)
        elif btype == "channel":
            self.attn = FastChannelAttention(dim, nh, qkv_b)
        else:
            self.attn = TokenDictionaryCrossAttention(dim, num_dict_tokens, nh, qkv_b)
        if btype != "dictionary":
            self.conv = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False), nn.GELU())
            self.inter = SimplifiedAIM(dim, aim_re)
        self.dp = DropPath(dp) if dp > 0.0 else nn.Identity()
        self.ffn = SimplifiedFFN(dim, ffn_exp)

    def _conv_fwd(self, x, H, W):
        B, L, C = x.shape
        return self.conv(x.transpose(1, 2).reshape(B, C, H, W)).reshape(B, C, L).transpose(1, 2)

    def forward(self, x, H, W):
        if self.btype == "dictionary":
            x = x + self.dp(self.attn(self.n1(x), H, W))
            return x + self.dp(self.ffn(self.n2(x), H, W))
        n1 = self.n1(x)
        itype = "channel_modulates_spatial" if self.btype == "spatial" else "spatial_modulates_channel"
        x = x + self.dp(self.inter(self.attn(n1, H, W), self._conv_fwd(n1, H, W), itype, H, W))
        return x + self.dp(self.ffn(self.n2(x), H, W))


class SimplifiedResidualGroup(nn.Module):
    def __init__(self, dim, depth, nh, split_size, ffn_exp, aim_re, pattern, dp_rates,
                 num_dict_tokens=128, start_parity=0, use_checkpoint=False) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        s0, s1 = split_size
        blocks, sp = [], start_parity
        for i in range(depth):
            btype = pattern[i % len(pattern)]
            if btype == "spatial":
                ws = (s0, s1) if sp % 2 == 0 else (s1, s0)
                sp += 1
            else:
                ws = (s0, s1)
            blocks.append(SimplifiedDATBlock(dim, nh, ws, ffn_exp, aim_re, btype, dp_rates[i], num_dict_tokens))
        self.blocks = nn.ModuleList(blocks)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_seq = x.reshape(B, C, H * W).transpose(1, 2).contiguous()
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x_seq = checkpoint(block, x_seq, H, W, use_reentrant=False)
            else:
                x_seq = block(x_seq, H, W)
        return self.conv(x_seq.transpose(1, 2).reshape(B, C, H, W)) + x


# ============================================================================================
# FDATRD3
# ============================================================================================

class FDATRD3(nn.Module):
    """2x SR. LR in, 2x out. stem='resize' (FDAT-RD2 layout) or 'unshuffle' (FDAT-RD 2x layout)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=2, embed_dim=128, num_groups=4,
                 depth_per_group=2, num_heads=4, split_size=(10, 30), num_dict_tokens=128,
                 ffn_expansion_ratio=2.0, aim_reduction_ratio=8, group_block_pattern=None,
                 drop_path_rate=0.1, stem="resize", resize_kernel="bicubic", bicubic_b=0.0,
                 bicubic_c=0.5, lanczos_taps=3, use_checkpoint=False, **legacy) -> None:
        super().__init__()
        # tolerate FDAT-RD/RD2 config keys that have a single valid value here
        unknown = set(legacy) - {"unshuffle_mod", "upsampler_type", "mid_dim", "img_range"}
        if unknown:
            raise TypeError(f"FDATRD3 got unexpected arguments: {sorted(unknown)}")
        if legacy.get("upsampler_type", "transpose+conv") != "transpose+conv":
            raise ValueError("FDATRD3 implements only the 'transpose+conv' x4 upsampler")
        if scale != 2:
            raise ValueError("FDATRD3 is a 2x model; use scale: 2")
        if stem not in STEMS:
            raise ValueError(f"stem must be one of {STEMS}, got {stem!r}")
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        split_size = tuple(split_size)
        if split_size != (10, 30):
            raise ValueError("FDATRD3 window split_size is fixed at (10, 30)")
        if group_block_pattern is None:
            group_block_pattern = ["spatial", "channel", "dictionary"]
        self.stem = stem
        self.upscale = 2
        align = math.lcm(*split_size)
        if stem == "resize":
            self.pre = FixedResize2x(num_in_ch, resize_kernel, bicubic_b, bicubic_c, lanczos_taps)
            r = 4
        else:
            r = 2
        self.pad = r * align  # in the coordinates of the tensor fed to the stem
        self.conv_first = nn.Sequential(PixelUnshuffle(r), nn.Conv2d(num_in_ch * r * r, embed_dim, 3, 1, 1, bias=True))

        ad = depth_per_group * len(group_block_pattern)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_groups * ad)]
        spg = sum(1 for j in range(ad) if group_block_pattern[j % len(group_block_pattern)] == "spatial")
        self.groups = nn.Sequential(*[
            SimplifiedResidualGroup(
                embed_dim, ad, num_heads, split_size, ffn_expansion_ratio, aim_reduction_ratio,
                group_block_pattern, dpr[i * ad:(i + 1) * ad], num_dict_tokens,
                start_parity=(i * spg) % 2, use_checkpoint=use_checkpoint)
            for i in range(num_groups)])
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)
        self.upsampler = Upsampler4x(embed_dim, num_out_ch)
        self.register_buffer(RD3_META_KEY, torch.tensor(
            [3.0, float(STEMS.index(stem)), float(RD3_VERSION)], dtype=torch.float32))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):  # ConvTranspose2d keeps PyTorch default, as FDAT-RD
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Accepts FDATRD3, FDAT-RD2 and FDAT-RD 2x-unshuffle checkpoints (marker conversion only)."""
        sd = dict(state_dict)
        up_meta = sd.pop("upsampler.MetaUpsample", None)
        if up_meta is not None:  # FDAT-RD / RD2 file: tail must be transpose+conv x4
            m = [int(v) for v in up_meta.tolist()]
            if m[1] != 5 or m[2] != 4:
                raise ValueError(f"FDATRD3 needs the 'transpose+conv' x4 tail; checkpoint has MetaUpsample={m}")
        if RD3_META_KEY in sd:
            stem_ckpt = STEMS[int(round(float(sd[RD3_META_KEY].flatten()[1])))]
            if stem_ckpt != self.stem:
                raise ValueError(f"checkpoint stem '{stem_ckpt}' != model stem '{self.stem}'")
        sd[RD3_META_KEY] = getattr(self, RD3_META_KEY).clone()
        if self.stem == "resize":
            mine = self.pre.MetaResize
            if RESIZE_META_KEY in sd:
                theirs = normalize_resize_meta(sd[RESIZE_META_KEY])
                if not torch.allclose(theirs, mine.float().cpu(), atol=1e-6):
                    raise ValueError(f"checkpoint resize {describe_resize(theirs)} != model's {describe_resize(mine)}")
                sd[RESIZE_META_KEY] = theirs.to(mine.device)
            else:
                sd[RESIZE_META_KEY] = mine.clone()
        elif RESIZE_META_KEY in sd:
            raise ValueError("checkpoint has a pre-resize (FDAT-RD2 layout); use stem='resize'")
        return super().load_state_dict(sd, *args, **kwargs)

    def freeze_for_export(self) -> "FDATRD3":
        """Bake upsampler conv kernels as constants. Call once, after loading, before ONNX export."""
        for m in self.modules():
            if isinstance(m, ConvTranspose4x2):
                m.freeze()
        return self.eval()

    def check_img_size(self, x: Tensor, h: int, w: int) -> Tensor:
        mph = (self.pad - h % self.pad) % self.pad
        mpw = (self.pad - w % self.pad) % self.pad
        if mph == 0 and mpw == 0:
            return x
        mode = "replicate" if (mph >= h or mpw >= w) else "reflect"
        return F.pad(x, (0, mpw, 0, mph), mode)

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        if self.stem == "resize":
            x = self.pre(x)
            oh, ow = x.shape[-2:]
        else:
            oh, ow = 2 * h, 2 * w
        x = self.check_img_size(x, x.shape[-2], x.shape[-1])
        x_shallow = self.conv_first(x)
        x_deep = self.conv_after(self.groups(x_shallow))
        return self.upsampler(x_deep + x_shallow)[:, :, :oh, :ow]


# ============================================================================================
# Config detection (converter / spandrel)
# ============================================================================================

def fdat_rd3_config_from_state_dict(sd) -> dict:
    """Infer FDATRD3 kwargs from an FDATRD3, FDAT-RD2 or FDAT-RD 2x-unshuffle state dict."""
    if "conv_first.1.weight" not in sd:
        raise ValueError("not an unshuffle-stem FDAT-RD/RD2/FDATRD3 checkpoint (no conv_first.1.weight)")
    w = sd["conv_first.1.weight"]
    embed_dim, stem_in = w.shape[0], w.shape[1]
    num_out_ch = sd["upsampler.3.weight"].shape[0]
    if RD3_META_KEY in sd:
        stem = STEMS[int(round(float(sd[RD3_META_KEY].flatten()[1])))]
    elif RESIZE_META_KEY in sd:
        stem = "resize"
    elif stem_in == num_out_ch * 4:
        stem = "unshuffle"  # FDAT-RD 2x unshuffle
    elif stem_in == num_out_ch * 16:
        raise ValueError("unshuffle-4 stem without a resize marker is a 1x FDAT-RD; "
                         "port it with port_to_fdat_rd2.py (adds the resize marker) first")
    else:
        raise ValueError(f"unexpected stem input channels {stem_in}")
    r = 4 if stem == "resize" else 2
    num_in_ch = stem_in // (r * r)
    if num_in_ch * r * r != stem_in:
        raise ValueError(f"stem input channels {stem_in} do not match stem '{stem}'")
    up = "upsampler.MetaUpsample"
    if up in sd:
        m = [int(v) for v in sd[up].tolist()]
        if m[1] != 5 or m[2] != 4:
            raise ValueError(f"FDATRD3 needs the 'transpose+conv' x4 tail; checkpoint has MetaUpsample={m}")

    def count(prefix):
        n = 0
        while any(k.startswith(f"{prefix}.{n}.") for k in sd):
            n += 1
        return n

    num_groups = count("groups")
    num_blocks = count("groups.0.blocks")

    def btype(i):
        b = f"groups.0.blocks.{i}.attn."
        if b + "dictionary" in sd:
            return "dictionary"
        if b + "temp" in sd:
            return "channel"
        return "spatial"

    types = [btype(i) for i in range(num_blocks)]
    period = next(p for p in range(1, num_blocks + 1)
                  if num_blocks % p == 0 and all(types[i] == types[i % p] for i in range(num_blocks)))
    sp = types.index("spatial")
    num_heads = sd[f"groups.0.blocks.{sp}.attn.rel_pos_bias"].shape[0]
    di = types.index("dictionary")
    cfg = dict(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        embed_dim=embed_dim, num_groups=num_groups, depth_per_group=num_blocks // period,
        num_heads=num_heads, split_size=(10, 30),
        num_dict_tokens=sd[f"groups.0.blocks.{di}.attn.dictionary"].shape[0],
        ffn_expansion_ratio=float(sd["groups.0.blocks.0.ffn.fc1.weight"].shape[0] / embed_dim),
        aim_reduction_ratio=embed_dim // sd[f"groups.0.blocks.{sp}.inter.cg.1.weight"].shape[0],
        group_block_pattern=types[:period], stem=stem,
    )
    if stem == "resize":
        meta = sd.get(RESIZE_META_KEY)
        cfg.update(resize_kwargs_from_meta(meta) if meta is not None else {})
    return cfg


def fdat_rd3_from_state_dict(sd, **override) -> FDATRD3:
    cfg = fdat_rd3_config_from_state_dict(sd)
    cfg.update(override)
    model = FDATRD3(**cfg)
    model.load_state_dict(sd, strict=True)
    return model


# ============================================================================================
# Converter CLI
# ============================================================================================

import argparse
import os
from collections import Counter, OrderedDict
from pathlib import Path


def load_checkpoint(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path, device="cpu")
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt
        if isinstance(ckpt, dict):
            for key in ("params_ema", "params", "state_dict", "model", "net"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    sd = ckpt[key]
                    break
    out = OrderedDict()
    for k, v in sd.items():
        for pre in ("module.", "model.", "net.", "network.", "state_dict."):
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def source_arch(sd):
    if RD3_META_KEY in sd:
        return "FDATRD3"
    return "FDAT-RD2" if RESIZE_META_KEY in sd else "FDAT-RD (2x unshuffle)"


def export_onnx(model, path, input_size, dynamic=False, opset=17):
    model.freeze_for_export()
    dummy = torch.rand(*input_size)
    dyn = ({"input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"}} if dynamic else None)
    kw = dict(input_names=["input"], output_names=["output"], dynamic_axes=dyn,
              opset_version=opset, do_constant_folding=True)
    # legacy (TorchScript) exporter only: dynamo graphs are rejected by TensorRT/VapourKit
    for extra in (dict(dynamo=False, external_data=False), dict(dynamo=False), {}):
        try:
            with torch.no_grad():
                torch.onnx.export(model, dummy, path, **extra, **kw)
            break
        except TypeError:
            continue
    print(f"  Saved: {path} ({os.path.getsize(path) / 1024 / 1024:.2f} MB) "
          f"[{'dynamic' if dynamic else 'static ' + 'x'.join(map(str, input_size[2:]))}, opset {opset}]")


def check_onnx(model, path, input_size):
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  onnx/onnxruntime not installed; skipping checks.")
        return
    g = onnx.load(path)
    onnx.checker.check_model(g)
    ops = Counter(n.op_type for n in g.graph.node)
    print(f"  graph: {sum(ops.values())} nodes | ConvTranspose {ops.get('ConvTranspose', 0)} | "
          f"Conv {ops.get('Conv', 0)} | MatMul {ops.get('MatMul', 0)} | Softmax {ops.get('Softmax', 0)} | "
          f"DepthToSpace/SpaceToDepth {ops.get('DepthToSpace', 0)}/{ops.get('SpaceToDepth', 0)}")
    x = torch.rand(*input_size)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    print(f"  ORT parity: max|torch-onnx| = {diff:.3e}  [{'OK' if diff < 1e-3 else 'WARN'}]")


def main():
    ap = argparse.ArgumentParser(
        description="FDATRD3 converter: FDATRD3 / FDAT-RD2 / FDAT-RD 2x-unshuffle checkpoint -> FDATRD3 ONNX or weights")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="output folder (default: next to input)")
    ap.add_argument("-f", "--formats", nargs="+", default=["onnx-static"],
                    choices=["onnx-static", "onnx-dynamic", "safetensors", "pth"])
    ap.add_argument("--input-size", type=int, nargs=2, default=[540, 720], metavar=("H", "W"),
                    help="static ONNX LR input size (default 540 720 -> 1080x1440)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--attn-scale", choices=["post", "pre"], default="post",
                    help="attention scaling order in the ONNX graph: post = (q@k^T)*scale, TensorRT's fused-attention pattern (default); pre = (q*scale)@k^T, as earlier exports")
    ap.add_argument("--name", help="output base name (default: input stem)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--info", action="store_true")
    args = ap.parse_args()

    global ATTN_EXPORT_SCALE
    ATTN_EXPORT_SCALE = args.attn_scale
    sd = load_checkpoint(args.input)
    print(f"Loading {args.input}  [{source_arch(sd)}]")
    model = fdat_rd3_from_state_dict(sd).eval()
    cfg = fdat_rd3_config_from_state_dict(sd)
    n = sum(p.numel() for p in model.parameters())
    pre = (describe_resize(model.pre.MetaResize) if model.stem == "resize" else "-")
    print(f"  FDATRD3 stem={model.stem} resize={pre} embed_dim={cfg['embed_dim']} groups={cfg['num_groups']} "
          f"depth={cfg['depth_per_group']} heads={cfg['num_heads']} dict_tokens={cfg['num_dict_tokens']} "
          f"pattern={cfg['group_block_pattern']} params={n / 1e6:.2f}M")
    print(f"  upsampler: ConvT 0 -> {model.upsampler._modules['0'].impl}, ConvT 2 -> {model.upsampler._modules['2'].impl}")
    if args.info:
        return

    out_dir = Path(args.output) if args.output else Path(args.input).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.name or Path(args.input).stem
    for fmt in args.formats:
        if fmt == "safetensors":
            from safetensors.torch import save_file
            p = out_dir / f"{base}_rd3.safetensors"
            save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(p))
            print(f"  Saved: {p}  (FDATRD3 weights; use as pretrain_network_g)")
        elif fmt == "pth":
            p = out_dir / f"{base}_rd3.pth"
            torch.save(model.state_dict(), str(p))
            print(f"  Saved: {p}")
        else:
            h, w = args.input_size
            isize = (1, cfg["num_in_ch"], h, w)
            dyn = fmt == "onnx-dynamic"
            tag = "" if args.attn_scale == "post" else "_prescale"
            p = str(out_dir / (f"{base}_rd3{tag}_dynamic.onnx" if dyn else f"{base}_rd3{tag}_static_{h}x{w}.onnx"))
            export_onnx(model, p, isize, dynamic=dyn, opset=args.opset)
            if not args.no_verify:
                check_onnx(model, p, isize)
    print("Done.")


if __name__ == "__main__":
    main()
