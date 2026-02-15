from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hf_model_utils import _safe_tensor_key
from .tile_utils import counts_from_array, counts_to_array, format_tag


def _safe_float_tag(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace("-", "m").replace(".", "p")


@dataclass
class CacheContext:
    root: Path
    tensor_name: str
    backend: str
    recompute: bool
    run_tag: str

    @property
    def safe_tensor(self) -> str:
        return _safe_tensor_key(self.tensor_name)

    def quant_path(self, compression: str, fmt: str) -> Path:
        return self.root / compression / self.backend / fmt / f"{self.safe_tensor}.npy"

    def mixed_path(
        self,
        compression: str,
        metric: str,
        threshold: float,
        cluster: str | None,
        k: int | None,
        iters: int | None,
        random_formats: list[str] | None,
    ) -> Path:
        thr_tag = _safe_float_tag(threshold)
        if compression == "mixed-tile-random":
            fmt_tag = format_tag(random_formats or [])
            return (
                self.root
                / compression
                / f"run-{self.run_tag}"
                / f"metric-{metric}"
                / f"iters-{iters}"
                / f"thr-{thr_tag}"
                / f"formats-{fmt_tag}"
                / self.backend
                / f"{self.safe_tensor}.npz"
            )
        return (
            self.root
            / compression
            / f"run-{self.run_tag}"
            / f"metric-{metric}"
            / f"thr-{thr_tag}"
            / f"cluster-{cluster}"
            / f"k-{k}"
            / self.backend
            / f"{self.safe_tensor}.npz"
        )

    def load_array(self, compression: str, fmt: str) -> np.ndarray | None:
        if self.recompute:
            return None
        path = self.quant_path(compression, fmt)
        if not path.exists():
            return None
        return np.load(path)

    def save_array(self, compression: str, fmt: str, y: np.ndarray) -> None:
        path = self.quant_path(compression, fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, y)

    def load_mixed(self, path: Path) -> tuple[np.ndarray, dict[str, int], np.ndarray] | None:
        if self.recompute or not path.exists():
            return None
        try:
            with np.load(path) as data:
                if not isinstance(data, np.lib.npyio.NpzFile):
                    return None
                if "y" not in data or "counts" not in data or "assignment" not in data:
                    return None
                y = np.asarray(data["y"], dtype=np.float32)
                counts = counts_from_array(data["counts"])
                assignment = np.asarray(data["assignment"], dtype=np.int8)
                return y, counts, assignment
        except Exception:
            return None

    def save_mixed(self, path: Path, y: np.ndarray, counts: dict[str, int], assignment: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            y=y,
            counts=counts_to_array(counts),
            assignment=assignment.astype(np.int8),
        )
