from __future__ import annotations

import numpy as np

from quantization_formats import quantize_weight_values


class Quantizer:
    def __init__(self, backend: str, ttnn=None) -> None:
        self.backend = backend
        self.ttnn = ttnn

    def quantize(self, xf: np.ndarray, fmt: str) -> np.ndarray:
        fmt_l = fmt.lower()
        if self.backend == "ttnn" and fmt_l in {"bfp8", "bfp4"}:
            import torch

            dtype_attr_map = {
                "bfp8": "bfloat8_b",
                "bfp4": "bfloat4_b",
                "bfp2": "bfloat2_b",
            }
            attr = dtype_attr_map[fmt_l]
            if self.ttnn is None:
                raise RuntimeError("Internal error: TTNN backend selected but ttnn is not initialized.")
            if not hasattr(self.ttnn, attr):
                raise RuntimeError(f"Active ttnn does not support format '{fmt_l}' (missing {attr}).")
            tt_dtype = getattr(self.ttnn, attr)
            x_t = torch.from_numpy(np.asarray(xf, dtype=np.float32))
            tt_tensor = self.ttnn.from_torch(x_t, dtype=tt_dtype, layout=self.ttnn.TILE_LAYOUT)
            y_t = self.ttnn.to_torch(tt_tensor).to(dtype=torch.float32).cpu()
            return y_t.numpy()

        return quantize_weight_values(xf, fmt_l)
