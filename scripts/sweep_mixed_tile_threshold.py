#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, Path.cwd()):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from tqdm import tqdm

from compression_algorithms.metrics import pearson_corr
from compression_algorithms.quantizer import Quantizer
from compression_algorithms.tile_utils import (
    MIXED_TILE_BYTES_PER_ELEM,
    MIXED_TILE_FORMATS,
    mixed_tile_total_bytes,
    reconstruct_from_tiles,
    reshape_to_2d_with_padding,
    tile_metrics,
)
from hf_model_utils import build_model_index, filter_tensor_names, load_tensor_fp32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep mixed-tile-threshold over a range of metric thresholds.",
    )
    parser.add_argument("repo_or_url", help="Hugging Face model repo/URL.")
    parser.add_argument(
        "tensor_name",
        help="Tensor name or filter (supports substrings and fnmatch patterns like 'model.layers.*.weight').",
    )
    parser.add_argument(
        "--regex",
        action="store_true",
        default=True,
        help="Interpret tensor_name as a regular expression (default: true).",
    )
    parser.add_argument(
        "--no-regex",
        dest="regex",
        action="store_false",
        help="Disable regex matching for tensor_name.",
    )
    parser.add_argument(
        "--list-matches",
        action="store_true",
        help="List matched tensor names and exit.",
    )
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
    parser.add_argument(
        "--metric",
        choices=["pcc", "mae", "atol"],
        default="pcc",
        help="Metric used to select tiles (default: pcc).",
    )
    parser.add_argument(
        "--lowest-metric-val",
        type=float,
        required=True,
        help="Lowest metric value to sweep to (pcc: lower bound, mae/atol: upper bound).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of threshold steps including endpoints (default: 100).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to results/<model>/mixed_tile_threshold_sweep/<timestamp>/<tensor>.",
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


def _metric_value(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "pcc":
        return pearson_corr(a, b)
    diff = np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))
    if metric == "mae":
        return float(np.mean(diff))
    if metric == "atol":
        return float(np.max(diff))
    raise ValueError(f"Unsupported metric: {metric}")


def _compute_assignment(
    scores_stack: np.ndarray,
    metric: str,
    threshold: float,
) -> np.ndarray:
    if metric == "pcc":
        good = scores_stack >= threshold
    else:
        good = scores_stack <= threshold
    good[-1, :] = True
    return np.argmax(good, axis=0).astype(np.int32)


def _pareto_mask(points: list[dict], metric: str) -> list[bool]:
    is_pcc = metric == "pcc"
    mask = [True for _ in points]
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if i == j:
                continue
            if is_pcc:
                dominates = b["size"] <= a["size"] and b["metric"] >= a["metric"]
                strictly = b["size"] < a["size"] or b["metric"] > a["metric"]
            else:
                dominates = b["size"] <= a["size"] and b["metric"] <= a["metric"]
                strictly = b["size"] < a["size"] or b["metric"] < a["metric"]
            if dominates and strictly:
                mask[i] = False
                break
    return mask


def _pareto_frontier(points: list[dict], metric: str) -> list[dict]:
    pareto_mask = _pareto_mask(points, metric)
    pareto_points = [p for p, keep in zip(points, pareto_mask) if keep]
    return sorted(pareto_points, key=lambda p: p["size"])


def _rgb_from_point(point: dict) -> tuple[float, float, float]:
    total = sum(float(point.get(f"{fmt}_tiles", 0.0)) for fmt in MIXED_TILE_FORMATS)
    if total <= 0.0:
        return (0.2, 0.2, 0.8)
    r = float(point.get("bfp2_tiles", 0.0)) / total
    b = float(point.get("bfp4_tiles", 0.0)) / total
    g = (float(point.get("bfp8_tiles", 0.0)) + float(point.get("bf16_tiles", 0.0))) / total
    gamma = 0.5
    r, g, b = (r ** gamma, g ** gamma, b ** gamma)
    norm = max(1e-8, r + g + b)
    return (r / norm, g / norm, b / norm)


def _pad_limits(min_v: float, max_v: float, pad_frac: float = 0.03) -> tuple[float, float]:
    span = max(max_v - min_v, 1e-9)
    pad = span * pad_frac
    return min_v - pad, max_v + pad


def _lighten_color(color: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    amount = min(max(amount, 0.0), 1.0)
    return tuple(c + (1.0 - c) * amount for c in color)


def _write_plot(
    out_path: Path,
    metric: str,
    points: list[dict],
    formats: list[str],
    tensor_name: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
    except Exception:
        return

    max_bytes = max(p["size"] for p in points)
    if max_bytes >= 1e9:
        scale = 1e9
        unit = "GB"
    elif max_bytes >= 1e6:
        scale = 1e6
        unit = "MB"
    else:
        scale = 1e3
        unit = "KB"

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    pareto_points = _pareto_frontier(points, metric)
    if not pareto_points:
        return
    xs = [p["size"] / scale for p in pareto_points]
    ys = [p["metric"] for p in pareto_points]
    point_colors = [_rgb_from_point(p) for p in pareto_points]
    if len(xs) > 1:
        segments = [
            [(xs[i], ys[i]), (xs[i + 1], ys[i + 1])]
            for i in range(len(xs) - 1)
        ]
        seg_colors = [
            tuple(
                (point_colors[i][c] + point_colors[i + 1][c]) / 2.0
                for c in range(3)
            )
            for i in range(len(point_colors) - 1)
        ]
        lc = LineCollection(segments, colors=seg_colors, linewidths=1.5)
        ax.add_collection(lc)
    ax.scatter(xs, ys, color=point_colors, s=20)

    for x, y, p in zip(xs, ys, pareto_points):
        if p.get("kind") != "baseline":
            continue
        ax.annotate(
            f"{p['label']} ({x:.2f}{unit})",
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
        )

    ax.set_xlabel(f"Size ({unit})")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Size vs metric sweep — {tensor_name}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(*_pad_limits(min(xs), max(xs)))
    ax.set_ylim(*_pad_limits(min(ys), max(ys)))

    fmt_colors = {
        "bf16": (0.0, 1.0, 0.0),
        "bfp8": (0.0, 1.0, 0.0),
        "bfp4": (0.0, 0.0, 1.0),
        "bfp2": (1.0, 0.0, 0.0),
    }
    legend_elements = []
    for fmt in formats:
        color = fmt_colors.get(fmt, (0.2, 0.2, 0.8))
        legend_elements.append(
            Line2D([0], [0], marker="o", color=color, label=fmt.upper(), markerfacecolor=color, markersize=6)
        )
    ax.legend(handles=legend_elements, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


_LAYER_RE = re.compile(r"(?:^|.*\.)layers\.(\d+)\.(.+)$")
_EXPERT_RE = re.compile(r"^(.*\bexperts)\.(\d+)\.(.+)$")


def _split_layer_suffix(tensor_name: str) -> tuple[int | None, str]:
    match = _LAYER_RE.match(tensor_name)
    if not match:
        return None, tensor_name
    return int(match.group(1)), match.group(2)


def _split_expert_suffix(suffix: str) -> tuple[str, int | None]:
    match = _EXPERT_RE.match(suffix)
    if not match:
        return suffix, None
    base = f"{match.group(1)}.{match.group(3)}"
    return base, int(match.group(2))


def _select_tensors(index, query: str, use_regex: bool) -> list[str]:
    names = list(index.tensor_to_file.keys())
    weight_like = [
        n for n in names
        if "weight" in n.lower() and not n.lower().endswith("_scale_inv")
    ]
    candidates = weight_like if weight_like else names

    if use_regex:
        try:
            pattern = re.compile(query)
        except re.error as exc:
            raise RuntimeError(f"Invalid regex '{query}': {exc}") from exc
        matches = [n for n in candidates if pattern.search(n)]
        if matches:
            return sorted(matches)
        raise RuntimeError("No tensors matched the regex query.")

    if query in candidates:
        return [query]

    if any(ch in query for ch in "*?[]"):
        matches = [n for n in candidates if fnmatch.fnmatch(n, query)]
        if matches:
            return sorted(matches)

    needle = query.lower()
    matches = [n for n in candidates if needle in n.lower()]
    if matches:
        return sorted(matches)

    matches = filter_tensor_names(candidates, query)
    if matches:
        return sorted(matches)

    raise RuntimeError("No tensors matched the filter query.")


def _write_group_overlays(
    out_path: Path,
    metric: str,
    grouped: dict[str, list[dict]],
    baselines: dict[str, list[dict]],
) -> None:
    if not grouped:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except Exception:
        return

    groups = sorted(grouped.items(), key=lambda kv: kv[0])
    ncols = len(groups)
    fig, axes = plt.subplots(1, ncols, figsize=(max(6.0, 4.0 * ncols), 4.5), squeeze=False)

    cmap = plt.get_cmap("Blues")

    all_metric_points = []
    for entries in grouped.values():
        for entry in entries:
            all_metric_points.extend(entry["points"])
    for entries in baselines.values():
        for entry in entries:
            all_metric_points.extend(entry["points"])
    global_min = min(p["metric"] for p in all_metric_points)
    global_max = max(p["metric"] for p in all_metric_points)
    for ax, (group_name, lines) in zip(axes[0], groups):
        baseline_lines = baselines.get(group_name, [])
        all_points = [p for line in lines for p in line["points"]]
        all_points += [p for line in baseline_lines for p in line["points"]]
        if not all_points:
            ax.set_axis_off()
            continue

        max_bytes = max(p["size"] for p in all_points)
        if max_bytes >= 1e9:
            scale = 1e9
            unit = "GB"
        elif max_bytes >= 1e6:
            scale = 1e6
            unit = "MB"
        else:
            scale = 1e3
            unit = "KB"

        layer_ids = [line["layer_id"] for line in lines if line["layer_id"] is not None]
        min_id = min(layer_ids) if layer_ids else 0
        max_id = max(layer_ids) if layer_ids else 0
        denom = max(1, max_id - min_id)

        for line in sorted(lines, key=lambda l: (l["layer_id"] is None, l["layer_id"])):
            xs = [p["size"] / scale for p in line["points"]]
            ys = [p["metric"] for p in line["points"]]
            point_colors = [_rgb_from_point(p) for p in line["points"]]
            if len(xs) > 1:
                layer_id = line["layer_id"]
                if layer_id is None:
                    t = 0.5
                else:
                    t = 0.9 - 0.8 * ((layer_id - min_id) / denom)
                color = cmap(t)
                ax.plot(xs, ys, color=color, linewidth=1.5)

        baseline_points = [p for line in baseline_lines for p in line["points"]]
        for p in baseline_points:
            x = p["size"] / scale
            y = p["metric"]
            color = _rgb_from_point(p)
            ax.scatter([x], [y], color=color, marker="o", s=30, edgecolors="black", linewidths=0.4)

        if baseline_lines:
            first_points = baseline_lines[0]["points"]
            for p in first_points:
                x = p["size"] / scale
                y = p["metric"]
                ax.annotate(
                    f"{p['label']} ({x:.2f}{unit})",
                    (x, y),
                    textcoords="offset points",
                    xytext=(6, 0),
                    ha="left",
                    va="center",
                    fontsize=6,
                )

        ax.set_title(group_name)
        ax.set_xlabel(f"Size ({unit})")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(*_pad_limits(global_min, global_max))
        x_vals = [p["size"] / scale for p in all_points]
        ax.set_xlim(*_pad_limits(min(x_vals), max(x_vals)))

    axes[0][0].set_ylabel(metric.upper())
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_layer_overlays(
    out_path: Path,
    metric: str,
    grouped: dict[int, list[dict]],
    baselines: dict[int, list[dict]],
) -> None:
    if not grouped:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return

    layers = sorted(grouped.items(), key=lambda kv: kv[0])
    ncols = len(layers)
    fig, axes = plt.subplots(1, ncols, figsize=(max(6.0, 4.0 * ncols), 4.5), squeeze=False)

    weight_names = sorted({line["weight_name"] for lines in grouped.values() for line in lines})
    if not weight_names:
        return
    if len(weight_names) <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(len(weight_names))]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / max(1, len(weight_names) - 1)) for i in range(len(weight_names))]
    weight_color = {name: colors[idx] for idx, name in enumerate(weight_names)}

    all_metric_points = []
    for entries in grouped.values():
        for entry in entries:
            all_metric_points.extend(entry["points"])
    for entries in baselines.values():
        for entry in entries:
            all_metric_points.extend(entry["points"])
    global_min = min(p["metric"] for p in all_metric_points)
    global_max = max(p["metric"] for p in all_metric_points)
    for ax, (layer_id, lines) in zip(axes[0], layers):
        baseline_lines = baselines.get(layer_id, [])
        all_points = [p for line in lines for p in line["points"]]
        all_points += [p for line in baseline_lines for p in line["points"]]
        if not all_points:
            ax.set_axis_off()
            continue

        max_bytes = max(p["size"] for p in all_points)
        if max_bytes >= 1e9:
            scale = 1e9
            unit = "GB"
        elif max_bytes >= 1e6:
            scale = 1e6
            unit = "MB"
        else:
            scale = 1e3
            unit = "KB"

        expert_ids = [line["expert_id"] for line in lines if line.get("expert_id") is not None]
        min_expert = min(expert_ids) if expert_ids else 0
        max_expert = max(expert_ids) if expert_ids else 0
        denom_expert = max(1, max_expert - min_expert)

        for line in sorted(lines, key=lambda l: l["weight_name"]):
            xs = [p["size"] / scale for p in line["points"]]
            ys = [p["metric"] for p in line["points"]]
            color = weight_color.get(line["weight_name"], (0.2, 0.2, 0.8))
            if line.get("expert_id") is not None:
                t = (line["expert_id"] - min_expert) / denom_expert if denom_expert else 0.0
                color = _lighten_color(color, 0.6 * t)
            ax.plot(xs, ys, color=color, linewidth=1.5)

        baseline_points = [p for line in baseline_lines for p in line["points"]]
        for p in baseline_points:
            x = p["size"] / scale
            y = p["metric"]
            color = _rgb_from_point(p)
            ax.scatter([x], [y], color=color, marker="o", s=30, edgecolors="black", linewidths=0.4)

        if baseline_lines:
            first_points = baseline_lines[0]["points"]
            for p in first_points:
                x = p["size"] / scale
                y = p["metric"]
                ax.annotate(
                    f"{p['label']} ({x:.2f}{unit})",
                    (x, y),
                    textcoords="offset points",
                    xytext=(6, 0),
                    ha="left",
                    va="center",
                    fontsize=6,
                )

        ax.set_title(f"Layer {layer_id}")
        ax.set_xlabel(f"Size ({unit})")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(*_pad_limits(global_min, global_max))
        x_vals = [p["size"] / scale for p in all_points]
        ax.set_xlim(*_pad_limits(min(x_vals), max(x_vals)))

    axes[0][0].set_ylabel(metric.upper())

    legend_handles = [
        Line2D([0], [0], color=weight_color[name], lw=2, label=name) for name in weight_names
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(4, len(legend_handles)),
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    formats = _parse_formats(args.formats)

    index = build_model_index(repo_or_url=args.repo_or_url, revision=args.revision, cache_dir=args.cache_dir)
    selected_tensors = _select_tensors(index, args.tensor_name, args.regex)
    if not selected_tensors:
        print("error: no tensors matched the filter query")
        return 1
    if args.list_matches:
        print(f"Matched {len(selected_tensors)} tensor(s):")
        for name in selected_tensors:
            print(f"  {name}")
        return 0

    multiple = len(selected_tensors) > 1
    base_out_dir = args.out_dir
    if base_out_dir is None:
        safe_model = index.repo_id.replace("/", "__")
        base_out_dir = (
            Path("results")
            / safe_model
            / "mixed_tile_threshold_sweep"
            / time.strftime("%Y%m%d-%H%M%S")
        )
    base_out = Path(base_out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    detail_out = base_out / "details"
    detail_out.mkdir(parents=True, exist_ok=True)

    ttnn = None
    if args.backend == "ttnn":
        ttnn = _import_ttnn_quiet()
    quantizer = Quantizer(backend=args.backend, ttnn=ttnn)

    grouped_lines: dict[str, list[dict]] = {}
    grouped_by_layer: dict[int, list[dict]] = {}
    grouped_baselines: dict[str, list[dict]] = {}
    grouped_baselines_by_layer: dict[int, list[dict]] = {}

    tensor_iter = tqdm(selected_tensors, desc="Tensors", unit="tensor")
    for tensor_name in tensor_iter:
        xf = np.asarray(load_tensor_fp32(index, tensor_name), dtype=np.float32)

        padded_ref, shape_info, pad_info = reshape_to_2d_with_padding(xf)
        tile_hw = 32
        tiles_h = pad_info[2] // tile_hw
        tiles_w = pad_info[3] // tile_hw
        tiles_ref = (
            padded_ref.reshape(tiles_h, tile_hw, tiles_w, tile_hw)
            .transpose(0, 2, 1, 3)
            .reshape(-1, tile_hw, tile_hw)
        )

        tiles_by_fmt: Dict[str, np.ndarray] = {}
        scores_by_fmt: Dict[str, np.ndarray] = {}

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
            scores_by_fmt[fmt] = tile_metrics(tiles_ref, tiles_q, args.metric)

        formats_by_precision = sorted(formats, key=lambda f: MIXED_TILE_BYTES_PER_ELEM.get(f, 0.0))
        highest_precision = max(formats_by_precision, key=lambda f: MIXED_TILE_BYTES_PER_ELEM.get(f, 0.0))
        fmt_order = {fmt: i for i, fmt in enumerate(formats_by_precision)}

        scores_stack = np.stack([scores_by_fmt[fmt] for fmt in formats_by_precision], axis=0)
        tiles_stack = np.stack([tiles_by_fmt[fmt] for fmt in formats_by_precision], axis=0)

        if args.metric == "pcc":
            start_metric = float(np.max(scores_by_fmt[highest_precision]))
            if args.lowest_metric_val > start_metric:
                print("error: lowest-metric-val must be <= start metric for pcc")
                return 1
            thresholds = np.linspace(start_metric, args.lowest_metric_val, max(1, args.steps))
        else:
            start_metric = float(np.min(scores_by_fmt[highest_precision]))
            if args.lowest_metric_val < start_metric:
                print("error: lowest-metric-val must be >= start metric for mae/atol")
                return 1
            thresholds = np.linspace(start_metric, args.lowest_metric_val, max(1, args.steps))

        slug = tensor_name.replace("/", "_").replace(".", "_")
        out_path = detail_out / slug
        out_path.mkdir(parents=True, exist_ok=True)

        config_payload = {
            "repo_or_url": args.repo_or_url,
            "tensor_name": tensor_name,
            "revision": args.revision,
            "backend": args.backend,
            "formats": formats,
            "metric": args.metric,
            "lowest_metric_val": args.lowest_metric_val,
            "steps": args.steps,
        }
        (out_path / "sweep_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

        baseline_points: list[dict] = []
        for fmt in formats:
            y_fmt = reconstruct_from_tiles(tiles_by_fmt[fmt], shape_info, pad_info, tile_hw=tile_hw)
            pcc = pearson_corr(xf, y_fmt)
            diff = np.abs(xf - y_fmt)
            mae = float(np.mean(diff))
            atol = float(np.max(diff))
            bytes_per_elem = MIXED_TILE_BYTES_PER_ELEM.get(fmt, 0.0)
            size_bytes = float(xf.size) * float(bytes_per_elem)
            metric_value = pcc if args.metric == "pcc" else (mae if args.metric == "mae" else atol)
            if args.metric == "pcc":
                if metric_value < args.lowest_metric_val:
                    continue
            else:
                if metric_value > args.lowest_metric_val:
                    continue
            baseline_points.append(
                {
                    "label": fmt.upper(),
                    "size": size_bytes,
                    "metric": metric_value,
                    "kind": "baseline",
                    "pcc": pcc,
                    "mae": mae,
                    "atol": atol,
                    f"{fmt}_tiles": int(tiles_ref.shape[0]),
                }
            )

        rows: list[dict] = []
        mixed_points: list[dict] = []
        last_assignments: np.ndarray | None = None
        last_metrics: dict | None = None

        thresh_iter = tqdm(
            enumerate(thresholds),
            total=len(thresholds),
            desc=f"Sweep {tensor_name}",
            unit="step",
            leave=False,
        )
        for step_idx, threshold in thresh_iter:
            assignments_idx = _compute_assignment(
                scores_stack,
                args.metric,
                float(threshold),
            )

            reuse = last_assignments is not None and np.array_equal(assignments_idx, last_assignments)
            if reuse and last_metrics is not None:
                pcc = last_metrics["pcc"]
                mae = last_metrics["mae"]
                atol = last_metrics["atol"]
                size_bytes = last_metrics["size_bytes"]
                counts = last_metrics["counts"]
            else:
                tiles_out = tiles_stack[assignments_idx, np.arange(assignments_idx.size)]
                y = reconstruct_from_tiles(tiles_out, shape_info, pad_info, tile_hw=tile_hw)
                pcc = pearson_corr(xf, y)
                diff = np.abs(xf - y)
                mae = float(np.mean(diff))
                atol = float(np.max(diff))

                counts_raw = np.bincount(assignments_idx, minlength=len(formats_by_precision))
                counts = {fmt: 0 for fmt in MIXED_TILE_FORMATS}
                for fmt, idx in fmt_order.items():
                    counts[fmt] = int(counts_raw[idx])
                size_bytes = mixed_tile_total_bytes(counts)

                last_assignments = assignments_idx
                last_metrics = {
                    "pcc": pcc,
                    "mae": mae,
                    "atol": atol,
                    "size_bytes": size_bytes,
                    "counts": counts,
                }

            metric_value = pcc if args.metric == "pcc" else (mae if args.metric == "mae" else atol)

            rows.append(
                {
                    "step": step_idx,
                    "threshold": float(threshold),
                    "size_bytes": size_bytes,
                    "pcc": pcc,
                    "mae": mae,
                    "atol": atol,
                    **{f"{fmt}_tiles": counts.get(fmt, 0) for fmt in formats},
                }
            )
            mixed_points.append(
                {
                    "label": f"t{step_idx}",
                    "size": size_bytes,
                    "metric": metric_value,
                    "kind": "mixed",
                    "pcc": pcc,
                    "mae": mae,
                    "atol": atol,
                    **{f"{fmt}_tiles": counts.get(fmt, 0) for fmt in formats},
                }
            )

        csv_path = out_path / "sweep_results.csv"
        headers = ["step", "threshold", "size_bytes", "pcc", "mae", "atol", *[f"{fmt}_tiles" for fmt in formats]]
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in rows:
                f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")

        plot_points = baseline_points + mixed_points
        _write_plot(out_path / "size_vs_metric.png", args.metric, plot_points, formats, tensor_name)

        layer_id, group_name = _split_layer_suffix(tensor_name)
        group_base, expert_id = _split_expert_suffix(group_name)
        group_key = group_base if expert_id is not None else group_name
        pareto_points = _pareto_frontier(plot_points, args.metric)
        if pareto_points:
            grouped_lines.setdefault(group_key, []).append(
                {"layer_id": layer_id, "points": pareto_points, "expert_id": expert_id}
            )
            if layer_id is not None:
                grouped_by_layer.setdefault(layer_id, []).append(
                    {"weight_name": group_key, "points": pareto_points, "expert_id": expert_id}
                )
        if baseline_points:
            grouped_baselines.setdefault(group_key, []).append(
                {"layer_id": layer_id, "points": baseline_points, "expert_id": expert_id}
            )
            if layer_id is not None:
                grouped_baselines_by_layer.setdefault(layer_id, []).append(
                    {"weight_name": group_key, "points": baseline_points, "expert_id": expert_id}
                )

    if grouped_lines:
        _write_group_overlays(
            base_out / "weight_overlays.png",
            args.metric,
            grouped_lines,
            grouped_baselines,
        )
    if grouped_by_layer:
        _write_layer_overlays(
            base_out / "layer_overlays.png",
            args.metric,
            grouped_by_layer,
            grouped_baselines_by_layer,
        )

    print(f"Wrote sweep results to {base_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
