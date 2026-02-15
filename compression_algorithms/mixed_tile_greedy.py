from __future__ import annotations

import numpy as np

from .base import CompressionAlgorithm, CompressionResult
from .cache import CacheContext
from .metrics import metric_better, metric_is_good, metric_value
from .quantizer import Quantizer
from .tile_utils import (
    MIXED_TILE_FORMATS,
    global_metric,
    kmeans_1d,
    mixed_tile_total_bytes,
    reconstruct_from_tiles,
    reshape_to_2d_with_padding,
    tile_metrics,
)


class MixedTileGreedyCompression(CompressionAlgorithm):
    name = "mixed-tile-greedy"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params=params)
        raw_formats = self.params.get("formats", self.params.get("tile_formats"))
        self.metric = self.params.get("metric", "pcc")
        self.threshold = float(self.params.get("threshold", 0.999))
        self.cluster = self.params.get("cluster", "kmeans")
        self.k = int(self.params.get("k", 10))
        self.tile_formats = self._parse_formats(raw_formats) if raw_formats is not None else None

        if self.metric not in {"pcc", "mae", "atol"}:
            raise ValueError(f"Unsupported metric: {self.metric}")
        if self.cluster not in {"kmeans", "single"}:
            raise ValueError(f"Unsupported cluster mode: {self.cluster}")
        if self.k < 1:
            raise ValueError("k must be >= 1")

    @classmethod
    def from_params(cls, params: dict | None = None) -> "MixedTileGreedyCompression":
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
                "mixed-tile-greedy requires at least one of "
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

        padded, shape_info, pad_info = reshape_to_2d_with_padding(xf)
        tile_hw = 32
        tiles_h = pad_info[2] // tile_hw
        tiles_w = pad_info[3] // tile_hw
        tiles_ref = (
            padded.reshape(tiles_h, tile_hw, tiles_w, tile_hw)
            .transpose(0, 2, 1, 3)
            .reshape(-1, tile_hw, tile_hw)
        )
        tiles_cur = tiles_ref.copy()
        unassigned = np.ones((tiles_ref.shape[0],), dtype=bool)
        assignments = np.full((tiles_ref.shape[0],), -1, dtype=np.int8)

        counts: dict[str, int] = {fmt: 0 for fmt in MIXED_TILE_FORMATS}

        if not metric_is_good(global_metric(xf, tiles_cur, shape_info, pad_info, self.metric), self.metric, self.threshold):
            return (
                reconstruct_from_tiles(tiles_cur, shape_info, pad_info, tile_hw=tile_hw),
                counts,
                assignments.reshape(tiles_h, tiles_w),
            )

        for fmt in tile_formats:
            any_accepted = False
            while True:
                candidates = np.where(unassigned)[0]
                if candidates.size == 0:
                    return (
                        reconstruct_from_tiles(tiles_cur, shape_info, pad_info, tile_hw=tile_hw),
                        counts,
                        assignments.reshape(tiles_h, tiles_w),
                    )

                cand_tiles = tiles_ref[candidates]
                cand_q = quantizer.quantize(cand_tiles, fmt)
                tile_scores = tile_metrics(cand_tiles, cand_q, self.metric)

                if self.cluster == "single" or candidates.size == 1:
                    if self.metric == "pcc":
                        best_idx = int(np.argmax(tile_scores))
                    else:
                        best_idx = int(np.argmin(tile_scores))
                    cluster_sel = np.array([best_idx], dtype=np.int64)
                else:
                    labels, _centroids = kmeans_1d(tile_scores, self.k)
                    best_cluster = None
                    best_score = None
                    for cluster_id in range(labels.max() + 1):
                        mask = labels == cluster_id
                        if not np.any(mask):
                            continue
                        avg_score = float(np.mean(tile_scores[mask]))
                        if best_score is None or metric_better(avg_score, best_score, self.metric):
                            best_score = avg_score
                            best_cluster = cluster_id
                    if best_cluster is None:
                        return (
                            reconstruct_from_tiles(tiles_cur, shape_info, pad_info, tile_hw=tile_hw),
                            counts,
                            assignments.reshape(tiles_h, tiles_w),
                        )
                    cluster_sel = np.where(labels == best_cluster)[0]

                cluster_avg = float(np.mean(tile_scores[cluster_sel]))
                if not metric_is_good(cluster_avg, self.metric, self.threshold):
                    if not any_accepted:
                        return (
                            reconstruct_from_tiles(tiles_cur, shape_info, pad_info, tile_hw=tile_hw),
                            counts,
                            assignments.reshape(tiles_h, tiles_w),
                        )
                    break

                tile_ids = candidates[cluster_sel]
                backup_tiles = tiles_cur[tile_ids].copy()
                tiles_cur[tile_ids] = cand_q[cluster_sel]
                global_score = metric_value(
                    xf,
                    reconstruct_from_tiles(tiles_cur, shape_info, pad_info, tile_hw=tile_hw),
                    self.metric,
                )
                if not metric_is_good(global_score, self.metric, self.threshold):
                    tiles_cur[tile_ids] = backup_tiles
                    return (
                        reconstruct_from_tiles(tiles_cur, shape_info, pad_info, tile_hw=tile_hw),
                        counts,
                        assignments.reshape(tiles_h, tiles_w),
                    )

                unassigned[tile_ids] = False
                counts[fmt] += int(tile_ids.size)
                assignments[tile_ids] = MIXED_TILE_FORMATS.index(fmt)
                any_accepted = True

        return (
            reconstruct_from_tiles(tiles_cur, shape_info, pad_info, tile_hw=tile_hw),
            counts,
            assignments.reshape(tiles_h, tiles_w),
        )

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
            )
        ]
