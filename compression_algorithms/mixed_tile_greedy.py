from __future__ import annotations

import math
import secrets

import numpy as np

from .base import CompressionAlgorithm, CompressionResult
from .cache import CacheContext
from .metrics import metric_is_good
from .quantizer import Quantizer
from .tile_utils import (
    MIXED_TILE_FORMATS,
    mixed_tile_total_bytes,
    reconstruct_from_tiles,
    reshape_to_2d_with_padding,
)


class MixedTileGreedyCompression(CompressionAlgorithm):
    name = "mixed-tile-greedy"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params=params)
        raw_formats = self.params.get("formats", self.params.get("tile_formats"))
        self.metric = self.params.get("metric", "pcc")
        self.threshold = float(self.params.get("threshold", 0.999))
        self.seed = int(self.params.get("seed", 0))
        self.tile_formats = self._parse_formats(raw_formats) if raw_formats is not None else None

        if self.metric not in {"pcc", "mae", "atol"}:
            raise ValueError(f"Unsupported metric: {self.metric}")

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
        num_tiles = tiles_ref.shape[0]
        fmt_to_idx = {fmt: idx for idx, fmt in enumerate(MIXED_TILE_FORMATS)}
        base_fmt = tile_formats[0]
        base_idx = fmt_to_idx[base_fmt]
        tiles_cur = quantizer.quantize(tiles_ref, base_fmt)
        assignments = np.full((num_tiles,), base_idx, dtype=np.int8)
        fixed = np.zeros((num_tiles,), dtype=bool)

        counts: dict[str, int] = {fmt: 0 for fmt in MIXED_TILE_FORMATS}
        counts[base_fmt] = int(num_tiles)

        h, w, _h_pad, _w_pad = pad_info
        row_end_by_tr = np.clip(h - (np.arange(tiles_h) * tile_hw), 0, tile_hw).astype(np.int32)
        col_end_by_tc = np.clip(w - (np.arange(tiles_w) * tile_hw), 0, tile_hw).astype(np.int32)
        vector_partial = False
        vector_partial_tr = -1
        vector_partial_cols = tile_hw
        if shape_info[0] == "vector":
            n = int(shape_info[1])
            last_valid_cols = n % tile_hw
            if last_valid_cols == 0:
                last_valid_cols = tile_hw
            if last_valid_cols != tile_hw:
                vector_partial = True
                last_row = h - 1
                vector_partial_tr = last_row // tile_hw
                vector_partial_cols = last_valid_cols

        def _iter_tile_views(x_tile: np.ndarray, y_tile: np.ndarray, tr: int, tc: int):
            row_end = int(row_end_by_tr[tr])
            col_end = int(col_end_by_tc[tc])
            if vector_partial and tr == vector_partial_tr:
                full_rows = row_end - 1
                if full_rows > 0:
                    yield x_tile[:full_rows, :col_end], y_tile[:full_rows, :col_end]
                yield x_tile[full_rows, :vector_partial_cols], y_tile[full_rows, :vector_partial_cols]
            else:
                yield x_tile[:row_end, :col_end], y_tile[:row_end, :col_end]

        metric = self.metric
        elem_count = float(xf.size)
        if metric == "pcc":
            per_tile_sum_y = np.zeros((num_tiles,), dtype=np.float64)
            per_tile_sum_y2 = np.zeros((num_tiles,), dtype=np.float64)
            per_tile_sum_xy = np.zeros((num_tiles,), dtype=np.float64)
            per_tile_sum_abs = np.zeros((num_tiles,), dtype=np.float64)
            sum_x = 0.0
            sum_x2 = 0.0
            sum_y = 0.0
            sum_y2 = 0.0
            sum_xy = 0.0
            sum_abs = 0.0

            for tile_id in range(num_tiles):
                tr, tc = divmod(tile_id, tiles_w)
                x_tile = tiles_ref[tile_id]
                y_tile = tiles_cur[tile_id]
                sx = 0.0
                sx2 = 0.0
                sy = 0.0
                sy2 = 0.0
                sxy = 0.0
                sab = 0.0
                for x_view, y_view in _iter_tile_views(x_tile, y_tile, tr, tc):
                    sx += float(np.sum(x_view, dtype=np.float64))
                    sx2 += float(np.sum(x_view * x_view, dtype=np.float64))
                    sy += float(np.sum(y_view, dtype=np.float64))
                    sy2 += float(np.sum(y_view * y_view, dtype=np.float64))
                    sxy += float(np.sum(x_view * y_view, dtype=np.float64))
                    diff = x_view - y_view
                    sab += float(np.sum(np.abs(diff), dtype=np.float64))
                sum_x += sx
                sum_x2 += sx2
                sum_y += sy
                sum_y2 += sy2
                sum_xy += sxy
                sum_abs += sab
                per_tile_sum_y[tile_id] = sy
                per_tile_sum_y2[tile_id] = sy2
                per_tile_sum_xy[tile_id] = sxy
                per_tile_sum_abs[tile_id] = sab

            def pcc_value(sum_y_val: float, sum_y2_val: float, sum_xy_val: float, sum_abs_val: float) -> float:
                if elem_count == 0.0:
                    return 1.0
                mean_x = sum_x / elem_count
                mean_y = sum_y_val / elem_count
                am2 = sum_x2 - elem_count * mean_x * mean_x
                bm2 = sum_y2_val - elem_count * mean_y * mean_y
                if am2 < 0.0:
                    am2 = 0.0
                if bm2 < 0.0:
                    bm2 = 0.0
                denom = math.sqrt(am2 * bm2)
                if denom == 0.0:
                    return 1.0 if sum_abs_val == 0.0 else 0.0
                return (sum_xy_val - elem_count * mean_x * mean_y) / denom

        elif metric == "mae":
            per_tile_sum_abs = np.zeros((num_tiles,), dtype=np.float64)
            sum_abs = 0.0
            for tile_id in range(num_tiles):
                tr, tc = divmod(tile_id, tiles_w)
                x_tile = tiles_ref[tile_id]
                y_tile = tiles_cur[tile_id]
                sab = 0.0
                for x_view, y_view in _iter_tile_views(x_tile, y_tile, tr, tc):
                    diff = x_view - y_view
                    sab += float(np.sum(np.abs(diff), dtype=np.float64))
                per_tile_sum_abs[tile_id] = sab
                sum_abs += sab

        else:
            per_tile_max = np.zeros((num_tiles,), dtype=np.float64)
            for tile_id in range(num_tiles):
                tr, tc = divmod(tile_id, tiles_w)
                x_tile = tiles_ref[tile_id]
                y_tile = tiles_cur[tile_id]
                tile_max = 0.0
                for x_view, y_view in _iter_tile_views(x_tile, y_tile, tr, tc):
                    diff = np.abs(x_view - y_view)
                    local_max = float(np.max(diff)) if diff.size else 0.0
                    if local_max > tile_max:
                        tile_max = local_max
                per_tile_max[tile_id] = tile_max
            max_abs = float(np.max(per_tile_max))
            max_abs_count = int(np.sum(per_tile_max == max_abs))

        seed = self.seed
        if seed == 0:
            seed = secrets.randbits(31)
        rng = np.random.default_rng(seed)

        for fmt in tile_formats:
            candidates = np.where(~fixed)[0]
            if candidates.size == 0:
                break
            order = rng.permutation(candidates)
            tiles_q = quantizer.quantize(tiles_ref, fmt)
            fmt_idx = fmt_to_idx[fmt]
            for tile_id in order:
                prev_idx = int(assignments[tile_id])
                if metric == "pcc":
                    current_value = pcc_value(sum_y, sum_y2, sum_xy, sum_abs)
                    if prev_idx == fmt_idx:
                        if not metric_is_good(current_value, metric, self.threshold):
                            fixed[tile_id] = True
                        continue
                    tr, tc = divmod(tile_id, tiles_w)
                    x_tile = tiles_ref[tile_id]
                    y_new = tiles_q[tile_id]
                    sy = 0.0
                    sy2 = 0.0
                    sxy = 0.0
                    sab = 0.0
                    for x_view, y_view in _iter_tile_views(x_tile, y_new, tr, tc):
                        sy += float(np.sum(y_view, dtype=np.float64))
                        sy2 += float(np.sum(y_view * y_view, dtype=np.float64))
                        sxy += float(np.sum(x_view * y_view, dtype=np.float64))
                        diff = x_view - y_view
                        sab += float(np.sum(np.abs(diff), dtype=np.float64))
                    old_sy = per_tile_sum_y[tile_id]
                    old_sy2 = per_tile_sum_y2[tile_id]
                    old_sxy = per_tile_sum_xy[tile_id]
                    old_sab = per_tile_sum_abs[tile_id]
                    cand_sum_y = sum_y + (sy - old_sy)
                    cand_sum_y2 = sum_y2 + (sy2 - old_sy2)
                    cand_sum_xy = sum_xy + (sxy - old_sxy)
                    cand_sum_abs = sum_abs + (sab - old_sab)
                    cand_value = pcc_value(cand_sum_y, cand_sum_y2, cand_sum_xy, cand_sum_abs)
                    if metric_is_good(cand_value, metric, self.threshold):
                        sum_y = cand_sum_y
                        sum_y2 = cand_sum_y2
                        sum_xy = cand_sum_xy
                        sum_abs = cand_sum_abs
                        per_tile_sum_y[tile_id] = sy
                        per_tile_sum_y2[tile_id] = sy2
                        per_tile_sum_xy[tile_id] = sxy
                        per_tile_sum_abs[tile_id] = sab
                        tiles_cur[tile_id] = y_new
                        counts[MIXED_TILE_FORMATS[prev_idx]] -= 1
                        counts[fmt] += 1
                        assignments[tile_id] = fmt_idx
                    else:
                        fixed[tile_id] = True
                elif metric == "mae":
                    current_value = sum_abs / elem_count if elem_count else 0.0
                    if prev_idx == fmt_idx:
                        if not metric_is_good(current_value, metric, self.threshold):
                            fixed[tile_id] = True
                        continue
                    tr, tc = divmod(tile_id, tiles_w)
                    x_tile = tiles_ref[tile_id]
                    y_new = tiles_q[tile_id]
                    sab = 0.0
                    for x_view, y_view in _iter_tile_views(x_tile, y_new, tr, tc):
                        diff = x_view - y_view
                        sab += float(np.sum(np.abs(diff), dtype=np.float64))
                    old_sab = per_tile_sum_abs[tile_id]
                    cand_sum_abs = sum_abs + (sab - old_sab)
                    cand_value = cand_sum_abs / elem_count if elem_count else 0.0
                    if metric_is_good(cand_value, metric, self.threshold):
                        sum_abs = cand_sum_abs
                        per_tile_sum_abs[tile_id] = sab
                        tiles_cur[tile_id] = y_new
                        counts[MIXED_TILE_FORMATS[prev_idx]] -= 1
                        counts[fmt] += 1
                        assignments[tile_id] = fmt_idx
                    else:
                        fixed[tile_id] = True
                else:
                    current_value = max_abs
                    if prev_idx == fmt_idx:
                        if not metric_is_good(current_value, metric, self.threshold):
                            fixed[tile_id] = True
                        continue
                    tr, tc = divmod(tile_id, tiles_w)
                    x_tile = tiles_ref[tile_id]
                    y_new = tiles_q[tile_id]
                    new_max = 0.0
                    for x_view, y_view in _iter_tile_views(x_tile, y_new, tr, tc):
                        diff = np.abs(x_view - y_view)
                        local_max = float(np.max(diff)) if diff.size else 0.0
                        if local_max > new_max:
                            new_max = local_max
                    old_max = per_tile_max[tile_id]
                    cand_max = max_abs
                    cand_count = max_abs_count
                    if new_max > max_abs:
                        cand_max = new_max
                        cand_count = 1
                    elif new_max == max_abs:
                        if old_max != max_abs:
                            cand_count = max_abs_count + 1
                    else:
                        if old_max == max_abs:
                            if max_abs_count > 1:
                                cand_count = max_abs_count - 1
                            else:
                                updated = per_tile_max.copy()
                                updated[tile_id] = new_max
                                cand_max = float(np.max(updated))
                                cand_count = int(np.sum(updated == cand_max))
                    if metric_is_good(cand_max, metric, self.threshold):
                        per_tile_max[tile_id] = new_max
                        max_abs = cand_max
                        max_abs_count = cand_count
                        tiles_cur[tile_id] = y_new
                        counts[MIXED_TILE_FORMATS[prev_idx]] -= 1
                        counts[fmt] += 1
                        assignments[tile_id] = fmt_idx
                    else:
                        fixed[tile_id] = True

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
                meta={"assignment": assignment, "tile_formats": tile_formats},
            )
        ]
