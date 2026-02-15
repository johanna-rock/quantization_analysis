from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .cache import CacheContext
from .quantizer import Quantizer


@dataclass
class CompressionResult:
    fmt: str
    compression: str
    y: np.ndarray
    tile_counts: dict[str, int] | None = None
    tile_bytes: float | None = None
    meta: dict | None = None


class CompressionAlgorithm(ABC):
    name: str

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}

    @classmethod
    def from_params(cls, params: dict | None = None) -> "CompressionAlgorithm":
        return cls(params=params or {})

    def expected_evals(self, formats: Iterable[str]) -> int:
        return len(list(formats))

    @abstractmethod
    def run(
        self,
        xf: np.ndarray,
        formats: list[str],
        quantizer: Quantizer,
        cache: CacheContext,
    ) -> list[CompressionResult]:
        raise NotImplementedError
