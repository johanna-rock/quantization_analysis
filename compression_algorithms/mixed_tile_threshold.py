from __future__ import annotations

import numpy as np

from .base import CompressionAlgorithm, CompressionResult
from .cache import CacheContext
from .metrics import metric_is_good
from .quantizer import Quantizer
from .tile_utils import (
    MIXED_TILE_BYTES_PER_ELEM,
    MIXED_TILE_FORMATS,
    mixed_tile_total_bytes,
    reconstruct_from_tiles,
    reshape_to_2d_with_padding,
    tile_metrics,
)


class MixedTileThresholdCompression(CompressionAlgorithm):
    name = "mixed-tile-threshold"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params=params)
        self.metric = self.params.get("metric", "pcc")
        self.threshold = float(self.params.get("threshold", 0.999))
        raw_formats = self.params.get("formats", self.params.get("tile_formats"))
        self.tile_formats = self._parse_formats(raw_formats) if raw_formats is not None else None

        if self.metric not in {"pcc", "mae", "atol"}:
            raise ValueError(f"Unsupported metric: {self.metric}")

    @classmethod
    def from_params(cls, params: dict | None = None) -> "MixedTileThresholdCompression":
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
                "mixed-tile-threshold requires at least one of "
                f"{', '.join(MIXED_TILE_FORMATS)} in quantization_formats"
            )
        return allowed

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

        padded_ref, shape_info, pad_info = reshape_to_2d_with_padding(xf)
        tile_hw = 32
        tiles_h = pad_info[2] // tile_hw
        tiles_w = pad_info[3] // tile_hw
        tiles_ref = (
            padded_ref.reshape(tiles_h, tile_hw, tiles_w, tile_hw)
            .transpose(0, 2, 1, 3)
            .reshape(-1, tile_hw, tile_hw)
        )

        y_by_fmt: dict[str, np.ndarray] = {}
        tiles_by_fmt: dict[str, np.ndarray] = {}
        scores_by_fmt: dict[str, np.ndarray] = {}

        for fmt in tile_formats:
            y_fmt = quantizer.quantize(xf, fmt)
            y_by_fmt[fmt] = y_fmt
            padded_q, _shape_q, pad_info_q = reshape_to_2d_with_padding(y_fmt)
            if pad_info_q != pad_info:
                raise ValueError("Quantized tensor padding mismatch.")
            tiles_q = (
                padded_q.reshape(tiles_h, tile_hw, tiles_w, tile_hw)
                .transpose(0, 2, 1, 3)
                .reshape(-1, tile_hw, tile_hw)
            )
            tiles_by_fmt[fmt] = tiles_q
            scores_by_fmt[fmt] = tile_metrics(tiles_ref, tiles_q, self.metric)

        fmt_to_idx = {fmt: idx for idx, fmt in enumerate(MIXED_TILE_FORMATS)}
        formats_by_precision = sorted(
            tile_formats, key=lambda f: MIXED_TILE_BYTES_PER_ELEM.get(f, 0.0)
        )
        best_precision = max(formats_by_precision, key=lambda f: MIXED_TILE_BYTES_PER_ELEM.get(f, 0.0))

        assignments = np.full((tiles_ref.shape[0],), fmt_to_idx[best_precision], dtype=np.int8)
        for tile_idx in range(tiles_ref.shape[0]):
            for fmt in formats_by_precision:
                score = scores_by_fmt[fmt][tile_idx]
                if metric_is_good(score, self.metric, self.threshold):
                    assignments[tile_idx] = fmt_to_idx[fmt]
                    break

        tiles_out = tiles_ref.copy()
        for fmt in tile_formats:
            tile_ids = np.where(assignments == fmt_to_idx[fmt])[0]
            if tile_ids.size == 0:
                continue
            tiles_out[tile_ids] = tiles_by_fmt[fmt][tile_ids]

        y = reconstruct_from_tiles(tiles_out, shape_info, pad_info, tile_hw=tile_hw)
        counts = {fmt: 0 for fmt in MIXED_TILE_FORMATS}
        for fmt in tile_formats:
            counts[fmt] = int(np.sum(assignments == fmt_to_idx[fmt]))

        return y, counts, assignments.reshape(tiles_h, tiles_w)

    def run(
        self,
        xf: np.ndarray,
        formats: list[str],
        quantizer: Quantizer,
        cache: CacheContext,
    ) -> list[CompressionResult]:
        tile_formats = self.tile_formats or self._filter_from_formats(formats)
        y, counts, assignment = self._compress(
            xf=xf,
            quantizer=quantizer,
            tile_formats=tile_formats,
        )
        total_bytes = mixed_tile_total_bytes(counts)
        return [
            CompressionResult(
                fmt="MIXED",
                compression=self.name,
                y=y,
                tile_counts=counts,
                tile_bytes=total_bytes,
                meta={"assignment": assignment, "tile_formats": tile_formats},
            )
        ]
