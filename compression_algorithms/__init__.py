from __future__ import annotations

from .base import CompressionAlgorithm, CompressionResult
from .config import CompressionConfig, load_compression_config
from .mixed_tile_greedy import MixedTileGreedyCompression
from .mixed_tile_random import MixedTileRandomCompression
from .mixed_tile_threshold import MixedTileThresholdCompression
from .none import NoneCompression
from .transpose import TransposeCompression

ALGORITHM_REGISTRY: dict[str, type[CompressionAlgorithm]] = {
    "none": NoneCompression,
    "transpose": TransposeCompression,
    "mixed-tile-greedy": MixedTileGreedyCompression,
    "mixed-tile-threshold": MixedTileThresholdCompression,
    "mixed-tile-random": MixedTileRandomCompression,
    "mixed-tile": MixedTileGreedyCompression,
}


def create_algorithm(name: str, params: dict | None = None) -> CompressionAlgorithm:
    key = name.strip().lower()
    cls = ALGORITHM_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unsupported compression algorithm '{name}'. "
            f"Supported: {', '.join(sorted(ALGORITHM_REGISTRY))}"
        )
    return cls.from_params(params or {})
