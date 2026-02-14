#!/usr/bin/env python3
"""
Plot per-tensor weight-value quantization curves over each tensor's full value range.

For every selected tensor, this script builds x from [min(weight), max(weight)] and
plots reconstructed value for:
  - Ideal (y=x)
  - BF16
  - BFP8 (0E7M + sign with per-value ideal shared exponent)
  - BFP4 (0E3M + sign with per-value ideal shared exponent)
  - BFP2 (0E1M + sign with per-value ideal shared exponent)
  - FP0 (all zeros)

It also reports MAE on the real tensor values for each format in the legend.

Dependencies:
  pip install numpy matplotlib safetensors torch huggingface_hub
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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
    """
    BFP ideal mode: pick the shared exponent from each value itself (scalar idealization).
    """
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
    from safetensors.torch import safe_open as safe_open_t
    import torch

    with safe_open_t(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        if tensor_name not in keys:
            raise KeyError(f"Tensor '{tensor_name}' not found in {path}")
        t = f.get_tensor(tensor_name)
        return t.to(dtype=torch.float32).cpu().numpy()


def _dequantize_tensor(tensor, inv_scale, block_shape: tuple[int, int]):
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


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def _mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(x, dtype=np.float32) - np.asarray(y, dtype=np.float32))))


def _plot_one_tensor(
    weights: np.ndarray,
    tensor_name: str,
    label: str,
    points: int,
    out_dir: Path,
    show: bool,
) -> Path:
    flat = np.asarray(weights, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise ValueError(f"Tensor '{tensor_name}' is empty.")

    w_min = float(np.min(flat))
    w_max = float(np.max(flat))
    if w_min == w_max:
        eps = max(abs(w_min) * 1e-6, 1e-6)
        xs = np.array([w_min - eps, w_max + eps], dtype=np.float32)
    else:
        xs = np.linspace(w_min, w_max, points, dtype=np.float32)

    y_ideal = xs
    y_bf16 = quantize_dequantize_bf16(xs)
    y_bfp8 = quantize_dequantize_bfp_ideal(xs, mant_bits=7)
    y_bfp4 = quantize_dequantize_bfp_ideal(xs, mant_bits=3)
    y_bfp2 = quantize_dequantize_bfp_ideal(xs, mant_bits=1)
    y_fp0 = quantize_fp0(xs)

    q_bf16 = quantize_dequantize_bf16(flat)
    q_bfp8 = quantize_dequantize_bfp_ideal(flat, mant_bits=7)
    q_bfp4 = quantize_dequantize_bfp_ideal(flat, mant_bits=3)
    q_bfp2 = quantize_dequantize_bfp_ideal(flat, mant_bits=1)
    q_fp0 = quantize_fp0(flat)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, y_ideal, label="Ideal", linewidth=2.0, color="black")
    ax.plot(xs, y_bf16, label=f"BF16 (MAE={_mae(flat, q_bf16):.3e})")
    ax.plot(xs, y_bfp8, label=f"BFP8 (MAE={_mae(flat, q_bfp8):.3e})")
    ax.plot(xs, y_bfp4, label=f"BFP4 (MAE={_mae(flat, q_bfp4):.3e})")
    ax.plot(xs, y_bfp2, label=f"BFP2 (MAE={_mae(flat, q_bfp2):.3e})")
    ax.plot(xs, y_fp0, label=f"FP0 (MAE={_mae(flat, q_fp0):.3e})")

    ax.set_title(f"{label}\n{tensor_name}")
    ax.set_xlabel("Original weight value")
    ax.set_ylabel("Reconstructed value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_slug(tensor_name)}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-tensor quantization curves over each tensor's min..max range.")
    parser.add_argument(
        "--ds-weights-path",
        type=str,
        default=None,
        help="Path to a DeepSeek safetensors shard/file, or cache directory.",
    )
    parser.add_argument(
        "--ds-tensors-file",
        type=str,
        default="ds_tensors.txt",
        help="Text file listing DeepSeek tensor names (one per line). Optional label via 'tensor_name | Label'.",
    )
    parser.add_argument(
        "--ds-tensor",
        type=str,
        default=None,
        help="Single tensor name (used if --ds-tensors-file is not provided).",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=1200,
        help="Number of x points for each curve over [min(weight), max(weight)].",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="plots/quantization_error_weights",
        help="Directory where per-tensor plot images are written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively (plots are always saved).",
    )
    args = parser.parse_args()

    if args.ds_tensors_file:
        try:
            tensors = _read_tensor_list(args.ds_tensors_file)
        except Exception as exc:
            raise SystemExit(f"Failed to read tensor list: {exc}") from exc
    elif args.ds_tensor:
        tensors = [(args.ds_tensor, None)]
    else:
        raise SystemExit("Provide --ds-tensors-file or --ds-tensor.")

    ds_cache = Path("data/deepseek-r1")
    ds_weights_file = None
    if args.ds_weights_path:
        p = Path(args.ds_weights_path).expanduser().resolve()
        if p.is_file():
            ds_weights_file = str(p)
        elif p.is_dir():
            ds_cache = p
        else:
            raise SystemExit(f"--ds-weights-path does not exist: {args.ds_weights_path}")

    try:
        index = _load_ds_index(ds_cache)
        weight_map = index.get("weight_map", {})
    except Exception as exc:
        print(f"Warning: failed to load DeepSeek index: {exc}", file=sys.stderr)
        weight_map = {}

    out_dir = Path(args.out_dir)
    shard_cache: dict[str, str] = {}
    produced: list[Path] = []

    for tensor_name, label_override in tensors:
        label = label_override or tensor_name
        file_exc = None
        try:
            if ds_weights_file is not None:
                try:
                    weights = _load_ds_weights_auto_from_file(ds_weights_file, tensor_name)
                except Exception as exc:
                    file_exc = exc
                    if not weight_map:
                        raise
                    weights = _load_ds_weights_auto_from_index(ds_cache, weight_map, tensor_name, shard_cache)
            else:
                weights = _load_ds_weights_auto_from_index(ds_cache, weight_map, tensor_name, shard_cache)

            out_path = _plot_one_tensor(
                weights=weights,
                tensor_name=tensor_name,
                label=label,
                points=args.points,
                out_dir=out_dir,
                show=args.show,
            )
            produced.append(out_path)
            print(f"Wrote {out_path}")
        except Exception as exc:
            if file_exc is not None:
                print(
                    f"Warning: local file fallback for {tensor_name} failed ({file_exc}); "
                    f"index/shard load failed: {exc}",
                    file=sys.stderr,
                )
            else:
                print(f"Warning: failed tensor {tensor_name}: {exc}", file=sys.stderr)

    if not produced:
        raise SystemExit("No plots were generated.")

    print(f"Generated {len(produced)} plot(s) in {out_dir}")


if __name__ == "__main__":
    main()
