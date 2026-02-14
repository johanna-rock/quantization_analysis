#!/usr/bin/env python3
"""
Reproduce "FP amax value" vs "Reconstructed FP value" staircase plots
for MXFP4, NVFP4, BF16, and BFP8/BFP4/BFP2 with (A) ideal exponent and
(B) shared exponent from a random 16-value normal block.

Dependencies:
  pip install numpy matplotlib safetensors
Optional (for float8 weights):
  pip install torch
Optional (to auto-download DeepSeek weights):
  pip install huggingface_hub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from typing import Sequence


# ---------------------------
# Utilities
# ---------------------------

def _safe_log2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    out[x > 0] = np.log2(x[x > 0])
    out[x <= 0] = -np.inf
    return out


def _nearest(x: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """
    Map each x to the nearest value in 'levels' (ties break toward the smaller level).
    """
    x = np.asarray(x, dtype=np.float32)
    levels = np.asarray(levels, dtype=np.float32)
    # Broadcast abs diffs: (N, L)
    diffs = np.abs(x[..., None] - levels[None, ...])
    idx = np.argmin(diffs, axis=-1)
    return levels[idx]


# ---------------------------
# BF16
# ---------------------------

def fp32_to_bf16_round_to_nearest_even(x: np.ndarray) -> np.ndarray:
    """
    Convert FP32 -> BF16 (stored in uint16) using round-to-nearest-even,
    then return as BF16-bit-pattern-in-uint16.
    BF16 layout: sign(1), exponent(8), mantissa(7).  [oai_citation:3‡Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus?utm_source=chatgpt.com)
    """
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)

    # Round-to-nearest-even when truncating lower 16 bits.
    lsb = (u >> 16) & 1
    rounding_bias = 0x7FFF + lsb  # ties-to-even
    u_rounded = u + rounding_bias

    bf16 = (u_rounded >> 16).astype(np.uint16)
    return bf16


def bf16_to_fp32(bf16: np.ndarray) -> np.ndarray:
    bf16 = np.asarray(bf16, dtype=np.uint16)
    u = (bf16.astype(np.uint32) << 16)
    return u.view(np.float32)


def quantize_dequantize_bf16(x: np.ndarray) -> np.ndarray:
    return bf16_to_fp32(fp32_to_bf16_round_to_nearest_even(x))


# ---------------------------
# FP4 E2M1 element quantizer
# ---------------------------

# Common E2M1 representable magnitudes (no NaN/Inf handling here).
# NVFP4 blog lists example values: 0, 0.5, 1.0, 1.5, 2, 3, 4, 6.  [oai_citation:4‡NVIDIA Developer](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
_FP4_E2M1_LEVELS_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

def quantize_fp4_e2m1(x: np.ndarray) -> np.ndarray:
    """
    Quantize to FP4 E2M1 using nearest representable value in {-6..6}.
    """
    x = np.asarray(x, dtype=np.float32)
    sign = np.sign(x)
    ax = np.abs(x)
    q = _nearest(ax, _FP4_E2M1_LEVELS_POS)
    return sign * q


# ---------------------------
# FP8 E4M3 quantizer (for NVFP4 scale)
# ---------------------------

def quantize_fp8_e4m3(x: np.ndarray) -> np.ndarray:
    """
    Quantize to an IEEE-like FP8 E4M3 (1 sign, 4 exponent, 3 mantissa).
    Used here for NVFP4's micro-block scale.  [oai_citation:5‡NVIDIA Developer](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

    Notes:
    - This is a practical minifloat quantizer (supports subnormals).
    - Special values (NaN/Inf) are not emphasized for this plotting use-case.
    """
    x = np.asarray(x, dtype=np.float32)
    sign = np.sign(x)
    ax = np.abs(x)

    ebits = 4
    mbits = 3
    bias = (2 ** (ebits - 1)) - 1  # bias=7 is commonly used for E4M3  [oai_citation:6‡Emergent Mind](https://www.emergentmind.com/topics/mxfp8-e4m3-floating-point-format?utm_source=chatgpt.com)

    # Handle zeros
    out = np.zeros_like(ax, dtype=np.float32)
    nz = ax > 0
    ax_nz = ax[nz]

    # Compute exponent
    e = np.floor(np.log2(ax_nz)).astype(np.int32)
    # Normal exponent range (excluding exp=0 subnorm and exp=all-ones specials)
    e_min = 1 - bias
    e_max = (2 ** ebits - 2) - bias

    # Normal numbers
    normal = (e >= e_min) & (e <= e_max)

    # Subnormals (too small): exponent field = 0
    sub = e < e_min

    # Too large: clamp to max finite
    big = e > e_max

    # ---- normals
    if np.any(normal):
        e_n = e[normal]
        m = ax_nz[normal] / (2.0 ** e_n)  # in [1,2)
        frac = m - 1.0                    # in [0,1)
        frac_q = np.round(frac * (2 ** mbits)) / (2 ** mbits)
        # Handle rounding that bumps to 2.0
        bumped = frac_q >= 1.0
        if np.any(bumped):
            frac_q[bumped] = 0.0
            e_n = e_n + 1
            e_n = np.minimum(e_n, e_max)
        out_n = (1.0 + frac_q) * (2.0 ** e_n)
        out[nz.nonzero()[0][normal]] = out_n

    # ---- subnormals (exp field = 0): value = frac * 2^(e_min)
    # quantize to steps of 2^(-mbits) * 2^(e_min)
    if np.any(sub):
        step = (2.0 ** e_min) / (2 ** mbits)
        out_s = np.round(ax_nz[sub] / step) * step
        out[nz.nonzero()[0][sub]] = out_s

    # ---- big: clamp
    if np.any(big):
        max_frac = (2 ** mbits - 1) / (2 ** mbits)
        max_val = (1.0 + max_frac) * (2.0 ** e_max)
        out[nz.nonzero()[0][big]] = max_val

    return sign * out


# ---------------------------
# Scales: E8M0 (power-of-two) and block quantization
# ---------------------------

def quantize_scale_e8m0_pow2(s: np.ndarray) -> np.ndarray:
    """
    E8M0 scale is power-of-two in practice (used as MXFP4 shared scale).  [oai_citation:7‡NVIDIA Developer](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
    We quantize to nearest power-of-two: 2^round(log2(s)).
    """
    s = np.asarray(s, dtype=np.float32)
    out = np.zeros_like(s, dtype=np.float32)
    nz = s > 0
    out[nz] = 2.0 ** np.round(_safe_log2(s[nz]))
    return out


def build_block_with_amax(am: float, block_size: int, rng: np.random.Generator | None, mode: str) -> np.ndarray:
    """
    mode:
      - "all_equal": all entries are +am
      - "rand16": only valid when block_size==16; create N(0,1) block scaled to max=am
    """
    if mode == "all_equal":
        return np.full((block_size,), float(am), dtype=np.float32)

    if mode == "rand16":
        if block_size != 16:
            raise ValueError("rand16 mode expects block_size=16")
        assert rng is not None
        v = rng.normal(loc=0.0, scale=1.0, size=(16,)).astype(np.float32)
        m = np.max(np.abs(v))
        if m == 0:
            v[0] = 1.0
            m = 1.0
        v = v * (float(am) / m)
        return v

    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------
# Format simulations
# ---------------------------

def quantize_scale_e8m0_pow2_round_up(s: np.ndarray) -> np.ndarray:
    """
    Quantize scale to E8M0 power-of-two, rounding UP to avoid overflow:
      s_q = 2^ceil(log2(s))
    """
    s = np.asarray(s, dtype=np.float32)
    out = np.zeros_like(s, dtype=np.float32)
    nz = s > 0
    out[nz] = 2.0 ** np.ceil(np.log2(s[nz]))
    return out

def simulate_mxfp4_amax(am: float) -> float:
    """
    MXFP4: 32 values share E8M0 (power-of-two) scale, values are FP4 E2M1.
    Choose smallest power-of-two scale >= amax/6 so normalized max <= 6 (no saturation).
    """
    block = np.full((32,), float(am), dtype=np.float32)
    amax = float(np.max(np.abs(block)))
    if amax == 0.0:
        return 0.0

    s = amax / 6.0
    s_q = float(quantize_scale_e8m0_pow2_round_up(np.array([s], dtype=np.float32))[0])

    xq = quantize_fp4_e2m1(block / s_q)
    xhat = xq * s_q
    return float(np.max(np.abs(xhat)))


def simulate_nvfp4_amax(am: float) -> float:
    """
    NVFP4: block size 16, shared FP8 E4M3 scale, data values in FP4 E2M1.  [oai_citation:9‡NVIDIA Developer](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
    (The NVIDIA blog also describes an additional per-tensor FP32 scale; omitted here to keep the
     "local amax mapping" plot comparable and simple.)
    """
    block = build_block_with_amax(am, 16, rng=None, mode="all_equal")
    amax = np.max(np.abs(block))
    s = amax / 6.0 if amax > 0 else 0.0
    s_q = float(quantize_fp8_e4m3(np.array([s], dtype=np.float32))[0])

    if s_q == 0:
        return 0.0
    xq = quantize_fp4_e2m1(block / s_q)
    xhat = xq * s_q
    return float(np.max(np.abs(xhat)))


def simulate_bf16_amax(am: float) -> float:
    x = np.array([am], dtype=np.float32)
    xhat = quantize_dequantize_bf16(x)
    return float(xhat[0])


def simulate_bfp_amax(
    am: float,
    mant_bits: int,
    mode: str,
    rng: np.random.Generator | None,
    rand_samples: int = 100,
) -> float:
    """
    BFP: sign + mantissa, shared exponent scale E8M0 (power-of-two) over 16 values.
    Your spec:
      - bfp8: 0E7M + sign, scaled with shared exponent E8M0
      - bfp4: 0E3M + sign, scaled with shared exponent E8M0
      - bfp2: 0E1M + sign, scaled with shared exponent E8M0

    Two exponent modes:
      A) mode="ideal": exponent from the value itself (equivalent to all-16-equal case)
      B) mode="rand": exponent from random N(0,1) blocks of 16 values (no rescale),
         averaged over rand_samples blocks
    """
    def _reconstruct_with_amax(amax: float) -> float:
        if amax == 0:
            return 0.0

        # max representable normalized magnitude with M mantissa bits
        max_norm = 2.0 - 2.0 ** (-mant_bits)

        # choose power-of-two shared scale so that amax/scale <= max_norm (no saturation)
        e = int(np.ceil(np.log2(amax / max_norm)))
        scale = float(2.0 ** e)

        # Normalize value to roughly [0,2) then quantize mantissa bits as fixed-point in [0,2).
        ax = abs(float(am)) / scale
        ax = np.clip(ax, 0.0, 2.0 - 2.0 ** (-mant_bits))
        step = 2.0 ** (-mant_bits)
        ax_q = np.round(ax / step) * step
        xhat = np.sign(float(am)) * ax_q * scale
        return float(abs(xhat))

    if mode == "ideal":
        return _reconstruct_with_amax(float(abs(am)))
    if mode == "rand":
        assert rng is not None
        total = 0.0
        for _ in range(rand_samples):
            block = rng.normal(loc=0.0, scale=1.0, size=(16,)).astype(np.float32)
            amax = float(np.max(np.abs(block)))
            total += _reconstruct_with_amax(amax)
        return total / float(rand_samples)

    raise ValueError("mode must be 'ideal' or 'rand'")


# ---------------------------
# DS weights curve (BFP4 from real blocks)
# ---------------------------

def _find_default_ds_weights() -> str | None:
    base = Path("data/deepseek-r1")
    if not base.exists():
        return None
    matches = list(base.rglob("model-00001-of-000163.safetensors"))
    if not matches:
        return None
    return str(matches[0])


def _resolve_ds_weights_path(path: str) -> str | None:
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return str(p)
    if p.is_dir():
        matches = list(p.rglob("model-00001-of-000163.safetensors"))
        if matches:
            return str(matches[0])
    return None


def _load_ds_index(cache_dir: Path) -> dict:
    from huggingface_hub import hf_hub_download
    import json

    index_path = hf_hub_download(
        repo_id="deepseek-ai/DeepSeek-R1",
        filename="model.safetensors.index.json",
        cache_dir=str(cache_dir),
    )
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _download_ds_shard(shard: str, cache_dir: Path) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id="deepseek-ai/DeepSeek-R1",
        filename=shard,
        cache_dir=str(cache_dir),
    )


def _load_safetensors_weight(path: str, tensor_name: str) -> np.ndarray:
    def _resolve_name(keys: set[str]) -> str:
        if tensor_name in keys:
            return tensor_name
        raise KeyError(f"Tensor '{tensor_name}' not found in {path}")

    # Prefer torch backend to support float8 weights.
    try:
        from safetensors.torch import safe_open as safe_open_t
        import torch

        with safe_open_t(path, framework="pt", device="cpu") as f:
            name = _resolve_name(set(f.keys()))
            t = f.get_tensor(name)
            return t.to(dtype=torch.float32).cpu().numpy()
    except Exception:
        from safetensors.numpy import safe_open as safe_open_np

        with safe_open_np(path, framework="numpy") as f:
            name = _resolve_name(set(f.keys()))
            t = f.get_tensor(name)
            return t


def _bfp4_scales_from_weights(
    weights: np.ndarray,
    block_size: int = 16,
    target_mean: float = 0.5,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32)
    last = w.shape[-1]
    trunc = (last // block_size) * block_size
    if trunc == 0:
        return np.empty((0,), dtype=np.float32)

    w = w[..., :trunc]
    blocks = w.reshape(-1, block_size)

    # Scale each block so mean(abs) == target_mean.
    mean_abs = np.mean(np.abs(blocks), axis=1)
    scale = np.zeros_like(mean_abs)
    nz = mean_abs > 0
    scale[nz] = target_mean / mean_abs[nz]
    blocks_scaled = blocks * scale[:, None]

    amax = np.max(np.abs(blocks_scaled), axis=1)
    scales = np.zeros_like(amax, dtype=np.float32)
    nz_amax = amax > 0
    if np.any(nz_amax):
        max_norm = 2.0 - 2.0 ** (-3)
        e = np.ceil(np.log2(amax[nz_amax] / max_norm)).astype(np.int32)
        scales[nz_amax] = (2.0 ** e).astype(np.float32)
    return scales


def _bfp4_reconstruct_weights(
    weights: np.ndarray,
    ds_scales: np.ndarray,
    block_size: int = 16,
    target_mean: float = 0.5,
    global_scale: float = 1.0,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32)
    last = w.shape[-1]
    trunc = (last // block_size) * block_size
    if trunc == 0:
        return np.empty((0,), dtype=np.float32)

    w = w[..., :trunc]
    blocks = w.reshape(-1, block_size)

    mean_abs = np.mean(np.abs(blocks), axis=1)
    scale_block = np.zeros_like(mean_abs)
    nz = mean_abs > 0
    scale_block[nz] = target_mean / (mean_abs[nz] * global_scale)

    blocks_scaled = blocks * (global_scale * scale_block[:, None])

    max_norm = 2.0 - 2.0 ** (-3)
    step = 2.0 ** (-3)

    ax = np.abs(blocks_scaled) / ds_scales[:, None]
    ax = np.clip(ax, 0.0, max_norm)
    ax_q = np.round(ax / step) * step
    xhat_scaled = np.sign(blocks_scaled) * ax_q * ds_scales[:, None]

    xhat = xhat_scaled / (scale_block[:, None] * global_scale)
    return xhat.reshape(w.shape)


def _best_global_bfp8_scale(
    weights: np.ndarray,
    ds_scales: np.ndarray,
    block_size: int = 16,
    target_mean: float = 0.5,
) -> float:
    # With per-block mean normalization to target_mean, the global scale cancels out.
    # We still return a valid BFP8 scale (1.0) for clarity.
    return float(quantize_fp8_e4m3(np.array([1.0], dtype=np.float32))[0])


def simulate_bfp4_ds_curve(xs: np.ndarray, ds_scales: np.ndarray, global_scale: float = 1.0) -> np.ndarray:
    ds_scales = np.asarray(ds_scales, dtype=np.float32)
    ds_scales = ds_scales[ds_scales > 0]
    if ds_scales.size == 0:
        return np.zeros_like(xs, dtype=np.float32)

    max_norm = 2.0 - 2.0 ** (-3)
    step = 2.0 ** (-3)

    ys = np.empty_like(xs, dtype=np.float32)
    for i, x in enumerate(xs):
        ax = (x * global_scale) / ds_scales
        ax = np.clip(ax, 0.0, max_norm)
        ax_q = np.round(ax / step) * step
        ys[i] = float(np.mean((ax_q * ds_scales) / global_scale))
    return ys


def _read_tensor_list(path: str) -> list[tuple[str, str | None]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tensor list file not found: {path}")
    out: list[tuple[str, str | None]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "|" in s:
            name, label = [x.strip() for x in s.split("|", 1)]
            out.append((name, label or None))
        else:
            out.append((s, None))
    return out


def _dequantize_tensor(
    tensor, inv_scale, block_shape: Sequence[int]
):
    """Dequantize a tensor using the provided scale (matches codebase helper)."""
    assert tensor.ndim == inv_scale.ndim
    assert len(block_shape) == tensor.ndim and all(
        inv_scale.shape[i] * block_shape[i] >= tensor.shape[i] for i in range(tensor.ndim)
    )
    for i, block_dim in enumerate(block_shape):
        inv_scale = inv_scale.repeat_interleave(block_dim, dim=i)
    tensor = tensor.float() * inv_scale[tuple(slice(0, s) for s in tensor.shape)].float()
    del inv_scale
    return tensor


def _load_tensor_from_shard(
    tensor_name: str,
    weight_map: dict[str, str],
    cache_dir: Path,
    shard_cache: dict[str, str],
):
    shard = weight_map.get(tensor_name)
    if shard is None:
        raise KeyError(f"Tensor '{tensor_name}' not found in index.")
    if shard not in shard_cache:
        shard_cache[shard] = _download_ds_shard(shard, cache_dir)
    shard_path = shard_cache[shard]

    from safetensors.torch import safe_open as safe_open_t

    with safe_open_t(shard_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        if tensor_name not in keys:
            raise KeyError(f"Tensor '{tensor_name}' not found in shard {shard}")
        return f.get_tensor(tensor_name)


def _load_ds_weights_auto_from_file(path: str, tensor_name: str) -> np.ndarray:
    # For requested *_fp32 tensors: prefer pre-dequantized key, else reconstruct from
    # {base, base_scale_inv} found in the same safetensors file.
    if tensor_name.endswith("_fp32"):
        base = tensor_name[:-5]
        scale_name = f"{base}_scale_inv"

        from safetensors.torch import safe_open as safe_open_t
        import torch

        with safe_open_t(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if tensor_name in keys:
                return f.get_tensor(tensor_name).to(dtype=torch.float32).cpu().numpy()
            if base in keys:
                if scale_name not in keys:
                    raise KeyError(f"Scale tensor '{scale_name}' not found in {path}")
                w = f.get_tensor(base)
                s = f.get_tensor(scale_name)
                w = _dequantize_tensor(w, s, block_shape=(128, 128))
                return w.to(dtype=torch.float32).cpu().numpy()
            raise KeyError(f"Tensor '{base}' not found in {path}")

    return _load_safetensors_weight(path, tensor_name)


def _load_ds_weights_auto_from_index(
    cache_dir: Path,
    weight_map: dict[str, str],
    tensor_name: str,
    shard_cache: dict[str, str],
) -> np.ndarray:
    # For requested *_fp32 tensors: prefer pre-dequantized key if present in index,
    # else reconstruct from base + base_scale_inv (these may reside in different shards).
    if tensor_name.endswith("_fp32"):
        base = tensor_name[:-5]
        scale_name = f"{base}_scale_inv"

        import torch

        if tensor_name in weight_map:
            t = _load_tensor_from_shard(tensor_name, weight_map, cache_dir, shard_cache)
            return t.to(dtype=torch.float32).cpu().numpy()

        w = _load_tensor_from_shard(base, weight_map, cache_dir, shard_cache)
        s = _load_tensor_from_shard(scale_name, weight_map, cache_dir, shard_cache)
        w = _dequantize_tensor(w, s, block_shape=(128, 128))
        return w.to(dtype=torch.float32).cpu().numpy()

    import torch
    t = _load_tensor_from_shard(tensor_name, weight_map, cache_dir, shard_cache)
    return t.to(dtype=torch.float32).cpu().numpy()


# ---------------------------
# Main: make the plot
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot amax reconstruction for low-precision formats.")
    parser.add_argument(
        "--rand-samples",
        type=int,
        default=100,
        help="Number of random blocks to average for BFP rand curves (default: 100).",
    )
    parser.add_argument(
        "--ds-weights-path",
        type=str,
        default=None,
        help="Path to DeepSeek safetensors shard for ds weights curve.",
    )
    parser.add_argument(
        "--ds-tensors-file",
        type=str,
        default=None,
        help="Text file listing DeepSeek tensor names (one per line) for ds curves. "
             "Optional label via 'tensor_name | Label'.",
    )
    parser.add_argument(
        "--ds-tensor",
        type=str,
        default="model.layers.0.self_attn.q_proj.weight",
        help="Tensor name for ds weights curve (used if --ds-tensors-file is not set).",
    )
    args = parser.parse_args()

    xs = np.linspace(0.0, 1.0, 400, dtype=np.float32)

    y_mx = np.array([simulate_mxfp4_amax(float(x)) for x in xs], dtype=np.float32)
    y_nv = np.array([simulate_nvfp4_amax(float(x)) for x in xs], dtype=np.float32)
    y_bf = np.array([simulate_bf16_amax(float(x)) for x in xs], dtype=np.float32)

    rng = np.random.default_rng(0)

    # BFP formats (A = ideal exponent, B = random block exponent)
    y_bfp8_A = np.array([simulate_bfp_amax(float(x), mant_bits=7, mode="ideal", rng=None) for x in xs], dtype=np.float32)
    y_bfp8_B = np.array([simulate_bfp_amax(float(x), mant_bits=7, mode="rand", rng=rng, rand_samples=args.rand_samples)  for x in xs], dtype=np.float32)

    rng = np.random.default_rng(0)
    y_bfp4_A = np.array([simulate_bfp_amax(float(x), mant_bits=3, mode="ideal", rng=None) for x in xs], dtype=np.float32)
    y_bfp4_B = np.array([simulate_bfp_amax(float(x), mant_bits=3, mode="rand", rng=rng, rand_samples=args.rand_samples)  for x in xs], dtype=np.float32)

    rng = np.random.default_rng(0)
    y_bfp2_A = np.array([simulate_bfp_amax(float(x), mant_bits=1, mode="ideal", rng=None) for x in xs], dtype=np.float32)
    y_bfp2_B = np.array([simulate_bfp_amax(float(x), mant_bits=1, mode="rand", rng=rng, rand_samples=args.rand_samples)  for x in xs], dtype=np.float32)

    ds_curves: list[tuple[str, np.ndarray]] = []
    ds_cache = Path("data/deepseek-r1")
    ds_tensors: list[tuple[str, str | None]]
    if args.ds_tensors_file:
        try:
            ds_tensors = _read_tensor_list(args.ds_tensors_file)
        except Exception as exc:
            print(f"Warning: failed to read ds tensors file: {exc}", file=sys.stderr)
            ds_tensors = []
    else:
        ds_tensors = [(args.ds_tensor, None)]

    ds_weights_path = args.ds_weights_path
    ds_weights_file = None
    ds_weights_dir = None
    if ds_weights_path:
        p = Path(ds_weights_path)
        if p.is_file():
            ds_weights_file = str(p.resolve())
        elif p.is_dir():
            ds_weights_dir = p.resolve()

    if ds_tensors:
        if ds_weights_dir is not None:
            ds_cache = ds_weights_dir

        try:
            index = _load_ds_index(ds_cache)
            weight_map = index.get("weight_map", {})
        except Exception as exc:
            print(f"Warning: failed to load DeepSeek index: {exc}", file=sys.stderr)
            weight_map = {}

        shard_cache: dict[str, str] = {}
        for tensor_name, label_override in ds_tensors:
            used_name = tensor_name
            file_exc = None

            try:
                if ds_weights_file is not None:
                    try:
                        weights = _load_ds_weights_auto_from_file(ds_weights_file, used_name)
                    except Exception as exc:
                        # If the local file is incomplete, fall back to index/shards.
                        file_exc = exc
                        if not weight_map:
                            raise
                        weights = _load_ds_weights_auto_from_index(ds_cache, weight_map, used_name, shard_cache)
                else:
                    weights = _load_ds_weights_auto_from_index(ds_cache, weight_map, used_name, shard_cache)

                ds_scales = _bfp4_scales_from_weights(weights, block_size=16, target_mean=0.5)
                global_scale = _best_global_bfp8_scale(weights, ds_scales, block_size=16, target_mean=0.5)
                y_ds = simulate_bfp4_ds_curve(xs, ds_scales, global_scale=global_scale)
                label = label_override or f"BFP4 DS {used_name.split('.')[-2]}"
                ds_curves.append((label, y_ds))
            except Exception as exc:
                if file_exc is not None:
                    print(
                        f"Warning: local file fallback for {used_name} failed ({file_exc}); "
                        f"index/shard load failed: {exc}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Warning: failed to build ds curve for {used_name}: {exc}",
                        file=sys.stderr,
                    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    lines = []
    labels = []

    def _add_line(y, label, **kwargs):
        line = ax.plot(xs, y, label=label, **kwargs)[0]
        lines.append(line)
        labels.append(label)

    _add_line(y_mx, "MXFP4")
    _add_line(y_nv, "NVFP4")
    _add_line(y_bf, "BF16")

    _add_line(y_bfp8_A, "BFP8 (ideal exp)")
    _add_line(y_bfp8_B, "BFP8 (rand16 exp)")

    _add_line(y_bfp4_A, "BFP4 (ideal exp)")
    _add_line(y_bfp4_B, "BFP4 (rand16 exp)")

    _add_line(y_bfp2_A, "BFP2 (ideal exp)")
    _add_line(y_bfp2_B, "BFP2 (rand16 exp)")

    for i, (label, y_ds) in enumerate(ds_curves):
        _add_line(y_ds, label, linewidth=2.5, linestyle="--", zorder=5)

    _add_line(xs, "IDEAL", linewidth=2)

    ax.set_xlabel("FP amax value")
    ax.set_ylabel("Reconstructed FP value")
    ax.set_title("amax reconstruction under low-precision formats")
    ax.grid(True, alpha=0.3)

    # Interactive checkboxes to toggle lines
    rax = fig.add_axes([0.82, 0.15, 0.17, 0.7])
    visibility = [line.get_visible() for line in lines]
    check = CheckButtons(rax, labels, visibility)

    def _refresh_legend():
        visible_lines = [line for line in lines if line.get_visible()]
        visible_labels = [lbl for line, lbl in zip(lines, labels) if line.get_visible()]
        ax.legend(handles=visible_lines, labels=visible_labels, loc="upper left")

    def _toggle(label):
        idx = labels.index(label)
        line = lines[idx]
        line.set_visible(not line.get_visible())
        _refresh_legend()
        fig.canvas.draw_idle()

    check.on_clicked(_toggle)

    _refresh_legend()
    plt.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])
    plt.show()


if __name__ == "__main__":
    main()
