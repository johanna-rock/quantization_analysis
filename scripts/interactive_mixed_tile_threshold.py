#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, Path.cwd()):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from compression_algorithms.metrics import metric_is_good
from compression_algorithms.quantizer import Quantizer
from compression_algorithms.tile_utils import (
    MIXED_TILE_BYTES_PER_ELEM,
    MIXED_TILE_FORMATS,
    reshape_to_2d_with_padding,
    tile_metrics,
)
from hf_model_utils import build_model_index, load_tensor_fp32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive mixed-tile threshold visualization.",
    )
    parser.add_argument("repo_or_url", help="Hugging Face model repo/URL.")
    parser.add_argument("tensor_name", help="Tensor name to analyze.")
    parser.add_argument("--revision", default="main", help="Hugging Face revision (default: main).")
    parser.add_argument(
        "--cache-dir",
        default="data/hf-cache",
        help="Shared local cache for downloaded/dequantized tensors (default: data/hf-cache).",
    )
    parser.add_argument(
        "--backend",
        choices=["emulation", "ttnn"],
        default="emulation",
        help="Quantization backend for BFP formats (default: emulation).",
    )
    parser.add_argument(
        "--formats",
        default="bf16,bfp8,bfp4,bfp2",
        help="Comma-separated mixed-tile formats (default: bf16,bfp8,bfp4,bfp2).",
    )
    return parser.parse_args()


def _import_ttnn_quiet():
    import os

    os.environ.setdefault("LOGURU_LEVEL", "ERROR")
    os.environ.setdefault("TTNN_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "ERROR")
    try:
        import ttnn  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        raise RuntimeError("TTNN backend requires `ttnn` in the active Python environment.") from exc
    return ttnn


def _parse_formats(value: str) -> List[str]:
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    formats = []
    seen = set()
    for part in parts:
        if part not in MIXED_TILE_FORMATS:
            raise ValueError(f"Unsupported mixed-tile format: {part}")
        if part in seen:
            continue
        seen.add(part)
        formats.append(part)
    if not formats:
        raise ValueError("No valid mixed-tile formats selected.")
    return formats


def _compute_assignment(
    scores_by_fmt: Dict[str, np.ndarray],
    formats_by_precision: List[str],
    fmt_to_idx: Dict[str, int],
    metric: str,
    threshold: float,
) -> np.ndarray:
    n_tiles = scores_by_fmt[formats_by_precision[0]].shape[0]
    best_precision = max(formats_by_precision, key=lambda f: MIXED_TILE_BYTES_PER_ELEM.get(f, 0.0))
    assignments = np.full((n_tiles,), fmt_to_idx[best_precision], dtype=np.int8)
    for tile_idx in range(n_tiles):
        for fmt in formats_by_precision:
            score = scores_by_fmt[fmt][tile_idx]
            if metric_is_good(score, metric, threshold):
                assignments[tile_idx] = fmt_to_idx[fmt]
                break
    return assignments


def main() -> int:
    args = parse_args()
    formats = _parse_formats(args.formats)

    index = build_model_index(repo_or_url=args.repo_or_url, revision=args.revision, cache_dir=args.cache_dir)
    x = load_tensor_fp32(index, args.tensor_name)
    xf = np.asarray(x, dtype=np.float32)

    padded_ref, _shape_info, pad_info = reshape_to_2d_with_padding(xf)
    tile_hw = 32
    tiles_h = pad_info[2] // tile_hw
    tiles_w = pad_info[3] // tile_hw
    tiles_ref = (
        padded_ref.reshape(tiles_h, tile_hw, tiles_w, tile_hw)
        .transpose(0, 2, 1, 3)
        .reshape(-1, tile_hw, tile_hw)
    )

    ttnn = None
    if args.backend == "ttnn":
        ttnn = _import_ttnn_quiet()
    quantizer = Quantizer(backend=args.backend, ttnn=ttnn)

    tiles_by_fmt: Dict[str, np.ndarray] = {}
    scores_by_metric: Dict[str, Dict[str, np.ndarray]] = {"pcc": {}, "mae": {}, "atol": {}}

    for fmt in formats:
        y_fmt = quantizer.quantize(xf, fmt)
        padded_q, _shape_q, pad_info_q = reshape_to_2d_with_padding(y_fmt)
        if pad_info_q != pad_info:
            raise ValueError("Quantized tensor padding mismatch.")
        tiles_q = (
            padded_q.reshape(tiles_h, tile_hw, tiles_w, tile_hw)
            .transpose(0, 2, 1, 3)
            .reshape(-1, tile_hw, tile_hw)
        )
        tiles_by_fmt[fmt] = tiles_q
        for metric in scores_by_metric:
            scores_by_metric[metric][fmt] = tile_metrics(tiles_ref, tiles_q, metric)

    fmt_to_idx = {fmt: idx for idx, fmt in enumerate(MIXED_TILE_FORMATS)}
    formats_by_precision = sorted(formats, key=lambda f: MIXED_TILE_BYTES_PER_ELEM.get(f, 0.0))

    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons, Slider
    except Exception:
        raise RuntimeError("matplotlib is required for interactive mode.")

    fmt_bytes = {fmt: MIXED_TILE_BYTES_PER_ELEM[fmt] for fmt in MIXED_TILE_FORMATS}
    sorted_fmts = sorted(MIXED_TILE_FORMATS, key=lambda f: fmt_bytes[f], reverse=True)

    h, w = tiles_h, tiles_w
    cell_size = 0.4
    fig_w = max(6.0, min(18.0, w * cell_size))
    fig_h = max(6.0, min(18.0, h * cell_size))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    active_metric = "pcc"
    threshold_values = {"pcc": 0.999, "mae": 1e-3, "atol": 1e-2}

    def _render(metric: str, threshold: float):
        assignments = _compute_assignment(
            scores_by_metric[metric],
            formats_by_precision,
            fmt_to_idx,
            metric,
            threshold,
        )
        assignment_2d = assignments.reshape(tiles_h, tiles_w)
        cmap = plt.get_cmap("Blues")
        color_steps = np.linspace(0.95, 0.15, num=len(sorted_fmts))
        fmt_to_color = {fmt: cmap(step) for fmt, step in zip(sorted_fmts, color_steps)}
        idx_to_color = [fmt_to_color[fmt] for fmt in MIXED_TILE_FORMATS]
        cmap_listed = plt.matplotlib.colors.ListedColormap(idx_to_color)
        cmap_listed.set_bad("gray")
        ax.clear()
        ax.imshow(
            assignment_2d,
            cmap=cmap_listed,
            vmin=-0.5,
            vmax=len(MIXED_TILE_FORMATS) - 0.5,
            interpolation="nearest",
        )
        x_step = 1 if tiles_w <= 64 else max(1, tiles_w // 32)
        y_step = 1 if tiles_h <= 64 else max(1, tiles_h // 32)
        ax.set_xticks(np.arange(0, tiles_w, x_step))
        ax.set_yticks(np.arange(0, tiles_h, y_step))
        ax.set_xticklabels([str(i) for i in range(0, tiles_w, x_step)], fontsize=7)
        ax.set_yticklabels([str(i) for i in range(0, tiles_h, y_step)], fontsize=7)
        ax.set_xlabel("Tile X")
        ax.set_ylabel("Tile Y")
        ax.set_xticks(np.arange(-0.5, tiles_w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, tiles_h, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title(f"{args.tensor_name}\\nmetric={metric} threshold={threshold:.4g}")
        from matplotlib.patches import Patch
        legend_handles = [Patch(color=fmt_to_color[fmt], label=fmt.upper()) for fmt in sorted_fmts]
        ax.legend(handles=legend_handles, title="Data format", loc="upper right", fontsize=8)
        fig.canvas.draw_idle()

    axcolor = "lightgoldenrodyellow"
    ax_pcc = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_mae = plt.axes([0.25, 0.11, 0.65, 0.03], facecolor=axcolor)
    ax_atol = plt.axes([0.25, 0.07, 0.65, 0.03], facecolor=axcolor)
    ax_radio = plt.axes([0.02, 0.4, 0.18, 0.15], facecolor=axcolor)

    slider_pcc = Slider(ax_pcc, "PCC", 0.9, 1.0, valinit=threshold_values["pcc"])
    slider_mae = Slider(ax_mae, "MAE", 1e-6, 1e-1, valinit=threshold_values["mae"])
    slider_atol = Slider(ax_atol, "ATOL", 1e-5, 1e-1, valinit=threshold_values["atol"])
    radio = RadioButtons(ax_radio, ("pcc", "mae", "atol"), active=0)

    def on_radio(label):
        nonlocal active_metric
        active_metric = label
        _render(active_metric, threshold_values[active_metric])

    def on_slider(metric: str):
        def _handler(val):
            threshold_values[metric] = val
            if active_metric != metric:
                return
            _render(metric, val)
        return _handler

    radio.on_clicked(on_radio)
    slider_pcc.on_changed(on_slider("pcc"))
    slider_mae.on_changed(on_slider("mae"))
    slider_atol.on_changed(on_slider("atol"))

    _render(active_metric, threshold_values[active_metric])
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
