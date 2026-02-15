#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compression_algorithms.quantizer import Quantizer
from compression_algorithms.tile_utils import (
    MIXED_TILE_FORMATS,
    reconstruct_from_tiles,
    reshape_to_2d_with_padding,
)
from hf_model_utils import build_model_index, load_tensor_fp32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a quantized tensor using a mixed-tile assignment map.",
    )
    parser.add_argument("repo_or_url", help="Hugging Face model repo/URL.")
    parser.add_argument("tensor_name", help="Tensor name to reconstruct.")
    parser.add_argument("assignment", help="Path to assignment .npy file (ints per tile).")
    parser.add_argument(
        "--assignment-mapping",
        default=None,
        help="Optional JSON mapping file (defaults to bf16,bfp8,bfp4,bfp2).",
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
        "--out",
        default=None,
        help="Optional output path (.npy). Defaults to <assignment>_recon.npy",
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


def _load_mapping(path: str | None) -> list[str]:
    if path is None:
        return MIXED_TILE_FORMATS
    mapping_path = Path(path)
    with mapping_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    formats = data.get("int_to_format")
    if not isinstance(formats, list) or not formats:
        raise ValueError("assignment mapping must contain int_to_format list")
    return [str(x).strip().lower() for x in formats]


def _quantize_tiles_by_assignment(
    tiles_ref: np.ndarray,
    assignments: np.ndarray,
    formats: list[str],
    quantizer: Quantizer,
) -> np.ndarray:
    tiles_out = tiles_ref.copy()
    for fmt_idx, fmt in enumerate(formats):
        tile_ids = np.where(assignments == fmt_idx)[0]
        if tile_ids.size == 0:
            continue
        tiles_out[tile_ids] = quantizer.quantize(tiles_ref[tile_ids], fmt)
    return tiles_out


def main() -> int:
    args = parse_args()

    index = build_model_index(repo_or_url=args.repo_or_url, revision=args.revision, cache_dir=args.cache_dir)
    x = load_tensor_fp32(index, args.tensor_name)
    xf = np.asarray(x, dtype=np.float32)

    assignment = np.load(args.assignment)
    assignment = np.asarray(assignment, dtype=np.int8)

    padded, shape_info, pad_info = reshape_to_2d_with_padding(xf)
    tile_hw = 32
    tiles_h = pad_info[2] // tile_hw
    tiles_w = pad_info[3] // tile_hw
    expected_shape = (tiles_h, tiles_w)
    if assignment.shape != expected_shape:
        raise ValueError(f"Assignment shape {assignment.shape} does not match expected {expected_shape}")

    tiles_ref = (
        padded.reshape(tiles_h, tile_hw, tiles_w, tile_hw)
        .transpose(0, 2, 1, 3)
        .reshape(-1, tile_hw, tile_hw)
    )

    formats = _load_mapping(args.assignment_mapping)
    assignments_flat = assignment.reshape(-1)

    ttnn = None
    if args.backend == "ttnn":
        ttnn = _import_ttnn_quiet()

    quantizer = Quantizer(backend=args.backend, ttnn=ttnn)
    tiles_q = _quantize_tiles_by_assignment(tiles_ref, assignments_flat, formats, quantizer)
    y = reconstruct_from_tiles(tiles_q, shape_info, pad_info, tile_hw=tile_hw)

    out_path = args.out
    if out_path is None:
        out_path = str(Path(args.assignment).with_suffix("")) + "_recon.npy"
    np.save(out_path, y)
    print(f"Wrote reconstructed tensor to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
