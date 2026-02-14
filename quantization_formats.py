#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


SUPPORTED_FORMATS = ["mxfp4", "nvfp4", "bf16", "bfp8", "bfp4", "bfp2", "fp0"]

_FP4_E2M1_LEVELS_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _safe_log2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    out[x > 0] = np.log2(x[x > 0])
    out[x <= 0] = -np.inf
    return out


def _nearest(x: np.ndarray, levels: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    levels = np.asarray(levels, dtype=np.float32)
    diffs = np.abs(x[..., None] - levels[None, ...])
    idx = np.argmin(diffs, axis=-1)
    return levels[idx]


def fp32_to_bf16_round_to_nearest_even(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lsb = (u >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    u_rounded = u + rounding_bias
    return (u_rounded >> 16).astype(np.uint16)


def bf16_to_fp32(bf16: np.ndarray) -> np.ndarray:
    bf16 = np.asarray(bf16, dtype=np.uint16)
    u = (bf16.astype(np.uint32) << 16)
    return u.view(np.float32)


def quantize_dequantize_bf16(x: np.ndarray) -> np.ndarray:
    return bf16_to_fp32(fp32_to_bf16_round_to_nearest_even(x))


def quantize_dequantize_bfp_ideal(x: np.ndarray, mant_bits: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    ax = np.abs(x)
    out = np.zeros_like(ax, dtype=np.float32)
    nz = ax > 0
    if not np.any(nz):
        return np.zeros_like(x, dtype=np.float32)

    max_norm = 2.0 - 2.0 ** (-mant_bits)
    step = 2.0 ** (-mant_bits)
    e = np.ceil(np.log2(ax[nz] / max_norm)).astype(np.int32)
    scale = np.exp2(e).astype(np.float32)
    norm = ax[nz] / scale
    norm = np.clip(norm, 0.0, max_norm)
    norm_q = np.round(norm / step) * step
    out[nz] = norm_q * scale
    return np.sign(x) * out


def quantize_fp0(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(x, dtype=np.float32), dtype=np.float32)


def quantize_weight_values(x: np.ndarray, fmt: str) -> np.ndarray:
    fmt = fmt.lower()
    x = np.asarray(x, dtype=np.float32)
    if fmt == "mxfp4":
        # Scalar proxy for MXFP4: apply amax mapping to each magnitude.
        ax = np.abs(x)
        q = np.array([simulate_mxfp4_amax(float(v)) for v in ax.reshape(-1)], dtype=np.float32).reshape(ax.shape)
        return np.sign(x) * q
    if fmt == "nvfp4":
        # Scalar proxy for NVFP4: apply amax mapping to each magnitude.
        ax = np.abs(x)
        q = np.array([simulate_nvfp4_amax(float(v)) for v in ax.reshape(-1)], dtype=np.float32).reshape(ax.shape)
        return np.sign(x) * q
    if fmt == "bf16":
        return quantize_dequantize_bf16(x)
    if fmt == "bfp8":
        return quantize_dequantize_bfp_ideal(x, mant_bits=7)
    if fmt == "bfp4":
        return quantize_dequantize_bfp_ideal(x, mant_bits=3)
    if fmt == "bfp2":
        return quantize_dequantize_bfp_ideal(x, mant_bits=1)
    if fmt == "fp0":
        return quantize_fp0(x)
    raise ValueError(f"Unsupported weight format: {fmt}")


def quantize_fp4_e2m1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    sign = np.sign(x)
    ax = np.abs(x)
    q = _nearest(ax, _FP4_E2M1_LEVELS_POS)
    return sign * q


def quantize_fp8_e4m3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    sign = np.sign(x)
    ax = np.abs(x)

    ebits, mbits = 4, 3
    bias = (2 ** (ebits - 1)) - 1
    out = np.zeros_like(ax, dtype=np.float32)
    nz = ax > 0
    ax_nz = ax[nz]
    if ax_nz.size == 0:
        return sign * out

    e = np.floor(np.log2(ax_nz)).astype(np.int32)
    e_min = 1 - bias
    e_max = (2 ** ebits - 2) - bias
    normal = (e >= e_min) & (e <= e_max)
    sub = e < e_min
    big = e > e_max

    nz_idx = nz.nonzero()[0]
    if np.any(normal):
        e_n = e[normal]
        m = ax_nz[normal] / (2.0 ** e_n)
        frac = m - 1.0
        frac_q = np.round(frac * (2 ** mbits)) / (2 ** mbits)
        bumped = frac_q >= 1.0
        if np.any(bumped):
            frac_q[bumped] = 0.0
            e_n = np.minimum(e_n + 1, e_max)
        out[nz_idx[normal]] = (1.0 + frac_q) * (2.0 ** e_n)

    if np.any(sub):
        step = (2.0 ** e_min) / (2 ** mbits)
        out[nz_idx[sub]] = np.round(ax_nz[sub] / step) * step

    if np.any(big):
        max_frac = (2 ** mbits - 1) / (2 ** mbits)
        max_val = (1.0 + max_frac) * (2.0 ** e_max)
        out[nz_idx[big]] = max_val

    return sign * out


def quantize_scale_e8m0_pow2_round_up(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float32)
    out = np.zeros_like(s, dtype=np.float32)
    nz = s > 0
    out[nz] = 2.0 ** np.ceil(np.log2(s[nz]))
    return out


def simulate_mxfp4_amax(am: float) -> float:
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
    block = np.full((16,), float(am), dtype=np.float32)
    amax = np.max(np.abs(block))
    s = amax / 6.0 if amax > 0 else 0.0
    s_q = float(quantize_fp8_e4m3(np.array([s], dtype=np.float32))[0])
    if s_q == 0:
        return 0.0
    xq = quantize_fp4_e2m1(block / s_q)
    xhat = xq * s_q
    return float(np.max(np.abs(xhat)))


def simulate_bfp_amax(am: float, mant_bits: int, mode: str, rand_samples: int = 100, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)

    def _reconstruct_with_amax(amax: float) -> float:
        if amax == 0:
            return 0.0
        max_norm = 2.0 - 2.0 ** (-mant_bits)
        e = int(np.ceil(np.log2(amax / max_norm)))
        scale = float(2.0 ** e)
        ax = abs(float(am)) / scale
        ax = np.clip(ax, 0.0, 2.0 - 2.0 ** (-mant_bits))
        step = 2.0 ** (-mant_bits)
        ax_q = np.round(ax / step) * step
        return float(abs(np.sign(float(am)) * ax_q * scale))

    if mode == "ideal":
        return _reconstruct_with_amax(float(abs(am)))
    if mode == "rand":
        total = 0.0
        for _ in range(rand_samples):
            block = rng.normal(loc=0.0, scale=1.0, size=(16,)).astype(np.float32)
            amax = float(np.max(np.abs(block)))
            total += _reconstruct_with_amax(amax)
        return total / float(rand_samples)
    raise ValueError("mode must be 'ideal' or 'rand'")


def make_synth_curves(xs: np.ndarray, formats: list[str], rand_samples: int = 100) -> dict[str, np.ndarray]:
    xs = np.asarray(xs, dtype=np.float32)
    out: dict[str, np.ndarray] = {"ideal": xs}
    if "mxfp4" in formats:
        out["mxfp4"] = np.array([simulate_mxfp4_amax(float(x)) for x in xs], dtype=np.float32)
    if "nvfp4" in formats:
        out["nvfp4"] = np.array([simulate_nvfp4_amax(float(x)) for x in xs], dtype=np.float32)
    if "bf16" in formats:
        out["bf16"] = quantize_dequantize_bf16(xs)
    if "bfp8" in formats:
        out["bfp8_ideal"] = np.array([simulate_bfp_amax(float(x), 7, "ideal", rand_samples=rand_samples) for x in xs], dtype=np.float32)
        out["bfp8_rand"] = np.array([simulate_bfp_amax(float(x), 7, "rand", rand_samples=rand_samples) for x in xs], dtype=np.float32)
    if "bfp4" in formats:
        out["bfp4_ideal"] = np.array([simulate_bfp_amax(float(x), 3, "ideal", rand_samples=rand_samples) for x in xs], dtype=np.float32)
        out["bfp4_rand"] = np.array([simulate_bfp_amax(float(x), 3, "rand", rand_samples=rand_samples) for x in xs], dtype=np.float32)
    if "bfp2" in formats:
        out["bfp2_ideal"] = np.array([simulate_bfp_amax(float(x), 1, "ideal", rand_samples=rand_samples) for x in xs], dtype=np.float32)
        out["bfp2_rand"] = np.array([simulate_bfp_amax(float(x), 1, "rand", rand_samples=rand_samples) for x in xs], dtype=np.float32)
    if "fp0" in formats:
        out["fp0"] = np.zeros_like(xs, dtype=np.float32)
    return out
