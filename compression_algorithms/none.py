from __future__ import annotations

import numpy as np

from .base import CompressionAlgorithm, CompressionResult
from .cache import CacheContext
from .quantizer import Quantizer


class NoneCompression(CompressionAlgorithm):
    name = "none"

    def run(
        self,
        xf: np.ndarray,
        formats: list[str],
        quantizer: Quantizer,
        cache: CacheContext,
    ) -> list[CompressionResult]:
        results: list[CompressionResult] = []
        for fmt in formats:
            y = cache.load_array(self.name, fmt)
            if y is not None and y.shape != xf.shape:
                y = None
            if y is None:
                y = quantizer.quantize(xf, fmt)
                cache.save_array(self.name, fmt, y)
            results.append(
                CompressionResult(fmt=fmt.upper(), compression=self.name, y=y)
            )
        return results
