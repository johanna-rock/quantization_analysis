from __future__ import annotations

import numpy as np

from .base import CompressionAlgorithm, CompressionResult
from .cache import CacheContext
from .metrics import metric_better, metric_is_good, metric_value
from .quantizer import Quantizer
from .tile_utils import (
    MIXED_TILE_BYTES_PER_ELEM,
    MIXED_TILE_FORMATS,
    assignment_to_array,
    mixed_tile_total_bytes,
    reconstruct_from_tiles,
    reshape_to_2d_with_padding,
)


class MixedTileRandomCompression(CompressionAlgorithm):
    name = "mixed-tile-random"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params=params)
        self.metric = self.params.get("metric", "pcc")
        self.threshold = float(self.params.get("threshold", 0.999))
        self.iters = int(self.params.get("iters", 50))
        self.seed = int(self.params.get("seed", 0))
        self.formats = self._parse_formats(self.params.get("formats"))

        if self.metric not in {"pcc", "mae", "atol"}:
            raise ValueError(f"Unsupported metric: {self.metric}")
        if self.iters < 1:
            raise ValueError("iters must be >= 1")

    @classmethod
    def from_params(cls, params: dict | None = None) -> "MixedTileRandomCompression":
        return cls(params=params or {})

    def expected_evals(self, formats: list[str]) -> int:
        return 1

    @staticmethod
    def _parse_formats(value) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            parts = [p.strip().lower() for p in value.split(",") if p.strip()]
        elif isinstance(value, list):
            parts = [str(p).strip().lower() for p in value if str(p).strip()]
        else:
            raise ValueError("formats must be a comma-separated string or a list of strings")
        if not parts:
            return []
        formats: list[str] = []
        seen = set()
        for part in parts:
            if part not in MIXED_TILE_FORMATS:
                raise ValueError(f"Unsupported mixed-tile format: {part}")
            if part in seen:
                continue
            seen.add(part)
            formats.append(part)
        return formats

    @staticmethod
    def _filter_from_formats(formats: list[str]) -> list[str]:
        allowed = [fmt for fmt in formats if fmt in MIXED_TILE_FORMATS]
        if not allowed:
            raise ValueError(
                "mixed-tile-random requires at least one of "
                f"{', '.join(MIXED_TILE_FORMATS)} in quantization_formats"
            )
        return allowed

    def _quantize_tiles_by_assignment(
        self,
        tiles_ref: np.ndarray,
        assignments: np.ndarray,
        quantizer: Quantizer,
    ) -> np.ndarray:
        tiles_out = tiles_ref.copy()
        for fmt_idx, fmt in enumerate(MIXED_TILE_FORMATS):
            tile_ids = np.where(assignments == fmt_idx)[0]
            if tile_ids.size == 0:
                continue
            tiles_out[tile_ids] = quantizer.quantize(tiles_ref[tile_ids], fmt)
        return tiles_out

    def _compress(
        self,
        xf: np.ndarray,
        quantizer: Quantizer,
        tile_formats: list[str],
    ) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
        if xf.size == 0:
            return (
                np.asarray(xf, dtype=np.float32),
                {fmt: 0 for fmt in MIXED_TILE_FORMATS},
                np.zeros((1, 1), dtype=np.int8),
            )

        padded, shape_info, pad_info = reshape_to_2d_with_padding(xf)
        tile_hw = 32
        tiles_h = pad_info[2] // tile_hw
        tiles_w = pad_info[3] // tile_hw
        tiles_ref = (
            padded.reshape(tiles_h, tile_hw, tiles_w, tile_hw)
            .transpose(0, 2, 1, 3)
            .reshape(-1, tile_hw, tile_hw)
        )

        fmt_indices = [MIXED_TILE_FORMATS.index(fmt) for fmt in tile_formats]
        if not fmt_indices:
            fmt_indices = list(range(len(MIXED_TILE_FORMATS)))
        fmt_indices = np.asarray(fmt_indices, dtype=np.int8)
        rng = np.random.default_rng(self.seed)
        bytes_per_elem = np.asarray(
            [
                MIXED_TILE_BYTES_PER_ELEM["bf16"],
                MIXED_TILE_BYTES_PER_ELEM["bfp8"],
                MIXED_TILE_BYTES_PER_ELEM["bfp4"],
                MIXED_TILE_BYTES_PER_ELEM["bfp2"],
            ],
            dtype=np.float32,
        )
        best_metric = None
        best_tiles = None
        best_assignments = None
        best_bytes = None

        for _ in range(max(1, self.iters)):
            choice_idx = rng.integers(0, len(fmt_indices), size=tiles_ref.shape[0], dtype=np.int64)
            assignments = fmt_indices[choice_idx].astype(np.int8)
            tiles_q = self._quantize_tiles_by_assignment(tiles_ref, assignments, quantizer)
            y = reconstruct_from_tiles(tiles_q, shape_info, pad_info, tile_hw=tile_hw)
            score = metric_value(xf, y, self.metric)
            meets = metric_is_good(score, self.metric, self.threshold)
            if meets:
                counts = np.bincount(assignments.astype(np.int64), minlength=len(MIXED_TILE_FORMATS))
                total_bytes = float(np.sum(counts * bytes_per_elem) * (tile_hw * tile_hw))
                if best_bytes is None or total_bytes < best_bytes:
                    best_bytes = total_bytes
                    best_metric = score
                    best_tiles = tiles_q
                    best_assignments = assignments.copy()
            elif best_bytes is None:
                if best_metric is None or metric_better(score, best_metric, self.metric):
                    best_metric = score
                    best_tiles = tiles_q
                    best_assignments = assignments.copy()

        if best_tiles is None or best_assignments is None:
            best_tiles = tiles_ref
            best_assignments = np.full((tiles_ref.shape[0],), -1, dtype=np.int8)

        counts = {fmt: 0 for fmt in MIXED_TILE_FORMATS}
        for fmt_idx, fmt in enumerate(MIXED_TILE_FORMATS):
            counts[fmt] = int(np.sum(best_assignments == fmt_idx))

        return (
            reconstruct_from_tiles(best_tiles, shape_info, pad_info, tile_hw=tile_hw),
            counts,
            best_assignments.reshape(tiles_h, tiles_w),
        )

    def run(
        self,
        xf: np.ndarray,
        formats: list[str],
        quantizer: Quantizer,
        cache: CacheContext,
    ) -> list[CompressionResult]:
        tile_formats = self.formats or self._filter_from_formats(formats)
        cache_path = cache.mixed_path(
            compression=self.name,
            metric=self.metric,
            threshold=self.threshold,
            cluster=None,
            k=None,
            iters=self.iters,
            random_formats=tile_formats,
        )
        cached = cache.load_mixed(cache_path)
        y = None
        counts = None
        assignment = None
        if cached is not None:
            y, counts, assignment = cached
            if y.shape != xf.shape:
                y = None
                counts = None
                assignment = None

        if y is None or counts is None or assignment is None:
            y, counts, assignment = self._compress(
                xf=xf,
                quantizer=quantizer,
                tile_formats=tile_formats,
            )
            cache.save_mixed(cache_path, y, counts, assignment_to_array(assignment))

        total_bytes = mixed_tile_total_bytes(counts)
        return [
            CompressionResult(
                fmt="MIXED",
                compression=self.name,
                y=y,
                tile_counts=counts,
                tile_bytes=total_bytes,
            )
        ]
