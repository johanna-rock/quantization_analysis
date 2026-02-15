from __future__ import annotations

import numpy as np

from .base import CompressionAlgorithm, CompressionResult
from .cache import CacheContext
from .quantizer import Quantizer


class TransposeCompression(CompressionAlgorithm):
    name = "transpose"

    def run(
        self,
        xf: np.ndarray,
        formats: list[str],
        quantizer: Quantizer,
        cache: CacheContext,
    ) -> list[CompressionResult]:
        results: list[CompressionResult] = []
        xf_t = np.transpose(np.asarray(xf, dtype=np.float32))
        for fmt in formats:
            y = cache.load_array(self.name, fmt)
            if y is not None and y.shape != xf.shape:
                y = None
            if y is None:
                y_comp = quantizer.quantize(xf_t, fmt)
                y = np.transpose(y_comp)
                cache.save_array(self.name, fmt, y)
            results.append(
                CompressionResult(fmt=fmt.upper(), compression=self.name, y=y)
            )
        return results
