from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompressionConfig:
    algorithm: str
    params: dict
    quantization_formats: list[str] | None


def load_compression_config(path: str | None) -> CompressionConfig:
    if path is None:
        return CompressionConfig(algorithm="none", params={}, quantization_formats=None)

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Compression config not found: {path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Compression config must be a JSON object")

    algorithm = str(data.get("algorithm", "none")).strip().lower()
    params = data.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("Compression config 'params' must be an object")

    qformats = data.get("quantization_formats")
    if qformats is None:
        quantization_formats = None
    else:
        if not isinstance(qformats, list):
            raise ValueError("Compression config 'quantization_formats' must be a list of strings")
        quantization_formats = [str(item).strip().lower() for item in qformats if str(item).strip()]
        if not quantization_formats:
            quantization_formats = None

    return CompressionConfig(algorithm=algorithm, params=params, quantization_formats=quantization_formats)
