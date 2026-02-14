#!/usr/bin/env python3
"""
Synthetic amax -> reconstructed-value staircase plots for quantization formats.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons

from hf_model_utils import resolve_format_list
from quantization_formats import (
    SUPPORTED_FORMATS,
    make_synth_curves,
    quantize_dequantize_bfp_ideal,
    simulate_bfp_ttnn_rand_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compare_reconstr_error_synth_data.py",
        description="Plot synthetic-only amax reconstruction curves for quantization formats.",
    )
    parser.add_argument(
        "-c",
        "--compress",
        action="append",
        metavar="FORMAT",
        help="Formats to include (repeatable): bf16, bfp8, bfp4, bfp2, fp0, or all.",
    )
    parser.add_argument(
        "--rand-samples",
        type=int,
        default=100,
        help="Number of random blocks to average for BFP rand curves (default: 100).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path to save the plot as a PNG.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (useful for headless runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = resolve_format_list(args.compress, SUPPORTED_FORMATS)

    xs = np.linspace(0.0, 1.0, 400, dtype=np.float32)
    curves = make_synth_curves(xs=xs, formats=formats, rand_samples=args.rand_samples)
    if "bfp8" in formats:
        curves["bfp8_ideal"] = quantize_dequantize_bfp_ideal(xs, mant_bits=7)
        rng = np.random.default_rng(0)
        curves["bfp8_rand"] = np.array(
            [simulate_bfp_ttnn_rand_row(float(x), 7, rand_samples=args.rand_samples, rng=rng) for x in xs],
            dtype=np.float32,
        )
    if "bfp4" in formats:
        curves["bfp4_ideal"] = quantize_dequantize_bfp_ideal(xs, mant_bits=3)
        rng = np.random.default_rng(0)
        curves["bfp4_rand"] = np.array(
            [simulate_bfp_ttnn_rand_row(float(x), 3, rand_samples=args.rand_samples, rng=rng) for x in xs],
            dtype=np.float32,
        )
    if "bfp2" in formats:
        curves["bfp2_ideal"] = quantize_dequantize_bfp_ideal(xs, mant_bits=1)
        rng = np.random.default_rng(0)
        curves["bfp2_rand"] = np.array(
            [simulate_bfp_ttnn_rand_row(float(x), 1, rand_samples=args.rand_samples, rng=rng) for x in xs],
            dtype=np.float32,
        )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    lines = []
    labels = []

    def _add_line(y, label, **kwargs):
        line = ax.plot(xs, y, label=label, **kwargs)[0]
        lines.append(line)
        labels.append(label)

    if "bf16" in curves:
        _add_line(curves["bf16"], "BF16")
    if "mxfp4" in curves:
        _add_line(curves["mxfp4"], "MXFP4")
    if "nvfp4" in curves:
        _add_line(curves["nvfp4"], "NVFP4")
    if "bfp8_ideal" in curves and "bfp8_rand" in curves:
        _add_line(curves["bfp8_ideal"], "BFP8 (ideal exp)")
        _add_line(curves["bfp8_rand"], "BFP8 (rand16 exp)")
    if "bfp4_ideal" in curves and "bfp4_rand" in curves:
        _add_line(curves["bfp4_ideal"], "BFP4 (ideal exp)")
        _add_line(curves["bfp4_rand"], "BFP4 (rand16 exp)")
    if "bfp2_ideal" in curves and "bfp2_rand" in curves:
        _add_line(curves["bfp2_ideal"], "BFP2 (ideal exp)")
        _add_line(curves["bfp2_rand"], "BFP2 (rand16 exp)")
    if "fp0" in curves:
        _add_line(curves["fp0"], "FP0")

    _add_line(curves["ideal"], "IDEAL", linewidth=2)

    ax.set_xlabel("FP amax value")
    ax.set_ylabel("Reconstructed FP value")
    ax.set_title("amax reconstruction under low-precision formats")
    ax.grid(True, alpha=0.3)

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

    headless = not os.environ.get("DISPLAY")
    out_path = args.out
    if out_path is None and (headless or args.no_show):
        out_path = os.path.join("plots", "compare_reconstr_error_synth_data.png")

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")

    if not headless and not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
