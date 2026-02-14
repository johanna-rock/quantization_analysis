#!/usr/bin/env python3
"""
Per-tensor quantization transfer plots over each tensor's full weight range.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hf_model_utils import build_model_index, load_tensor_fp32, resolve_format_list, resolve_selected_tensors
from quantization_formats import SUPPORTED_FORMATS, quantize_weight_values


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def _mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(x, dtype=np.float32) - np.asarray(y, dtype=np.float32))))


def _quantize(x: np.ndarray, fmt: str) -> np.ndarray:
    return quantize_weight_values(x, fmt)


def _label(fmt: str) -> str:
    return fmt.upper()


def _plot_one_tensor(
    weights: np.ndarray,
    tensor_name: str,
    points: int,
    out_dir: Path,
    show: bool,
    formats: list[str],
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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, xs, label="Ideal", linewidth=2.0, color="black")

    for fmt in formats:
        y = _quantize(xs, fmt)
        q = _quantize(flat, fmt)
        ax.plot(xs, y, label=f"{_label(fmt)} (MAE={_mae(flat, q):.3e})")

    ax.set_title(tensor_name)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compare_reconstr_error_weights.py",
        description="Plot per-tensor quantization transfer curves from any Hugging Face model repo.",
    )
    parser.add_argument("repo_or_url", help="Hugging Face model repo id or URL.")
    parser.add_argument(
        "filter_query",
        nargs="*",
        help="Optional tensor filter (substring, or dotted torch-style path prefix).",
    )
    parser.add_argument("--revision", default="main", help="Hugging Face revision (default: main).")
    parser.add_argument(
        "--cache-dir",
        default="data/hf-cache",
        help="Shared local cache for downloaded and dequantized tensors (default: data/hf-cache).",
    )
    parser.add_argument(
        "-c",
        "--compress",
        action="append",
        metavar="FORMAT",
        help="Formats to plot (repeatable): mxfp4, nvfp4, bf16, bfp8, bfp4, bfp2, fp0, or all.",
    )
    parser.add_argument("--points", type=int, default=1200, help="Points per curve (default: 1200).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="plots/visualize_quantization_error",
        help="Output directory for per-tensor PNG plots.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filter_query = " ".join(args.filter_query).strip() or None
    formats = resolve_format_list(args.compress, SUPPORTED_FORMATS)

    index = build_model_index(repo_or_url=args.repo_or_url, revision=args.revision, cache_dir=args.cache_dir)
    tensor_names = resolve_selected_tensors(index, filter_query)

    out_dir = Path(args.out_dir)
    produced: list[Path] = []
    for tensor_name in tensor_names:
        weights = load_tensor_fp32(index, tensor_name)
        out = _plot_one_tensor(
            weights=weights,
            tensor_name=tensor_name,
            points=args.points,
            out_dir=out_dir,
            show=args.show,
            formats=formats,
        )
        produced.append(out)
        print(f"Wrote {out}")

    print(f"Generated {len(produced)} plot(s) in {out_dir}")


if __name__ == "__main__":
    main()
