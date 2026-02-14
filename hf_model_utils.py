#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import struct
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download


def resolve_hf_token() -> Optional[str]:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = __import__("os").getenv(env_name)
        if value and value.strip():
            return value.strip()
    return None


def normalize_repo_id(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        raise ValueError("Empty repo value.")

    if "://" not in value:
        return value.strip("/")

    parsed = urlparse(value)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in {"huggingface.co", "hf.co"}:
        raise ValueError(f"Unsupported host: {parsed.netloc}")

    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        raise ValueError("URL path does not contain a repo id.")

    if parts[0] in {"models", "model"}:
        parts = parts[1:]
    elif parts[0] in {"datasets", "spaces"}:
        raise ValueError("Only model repos are supported.")

    stop_tokens = {"tree", "blob", "resolve", "commit", "discussions"}
    for idx, part in enumerate(parts):
        if part in stop_tokens:
            parts = parts[:idx]
            break

    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def filter_tensor_names(names: list[str], query: Optional[str]) -> list[str]:
    if not query:
        return sorted(names)
    trimmed = query.strip()
    if not trimmed:
        return sorted(names)

    if "." in trimmed:
        qparts = [p.lower() for p in trimmed.split(".") if p]
        out = []
        for name in names:
            parts = name.lower().split(".")
            if len(parts) >= len(qparts) and parts[: len(qparts)] == qparts:
                out.append(name)
        return sorted(out)

    needle = trimmed.lower()
    return sorted([n for n in names if needle in n.lower()])


def _parse_safetensors_header_bytes(data: bytes, filename: str) -> dict:
    if len(data) < 8:
        raise RuntimeError(f"{filename}: invalid safetensors header.")
    header_len = struct.unpack("<Q", data[:8])[0]
    header_end = 8 + header_len
    if len(data) < header_end:
        raise RuntimeError(f"{filename}: truncated safetensors header.")
    try:
        return json.loads(data[8:header_end].decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{filename}: malformed safetensors header JSON.") from exc


def _read_safetensors_header_remote(fs: HfFileSystem, repo_id: str, filename: str, revision: str) -> dict:
    with fs.open(f"{repo_id}/{filename}", "rb", revision=revision) as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"{filename}: invalid safetensors header prefix.")
        header_len = struct.unpack("<Q", prefix)[0]
        rest = f.read(header_len)
    return _parse_safetensors_header_bytes(prefix + rest, filename=filename)


@dataclass
class ModelIndex:
    repo_id: str
    revision: str
    cache_dir: Path
    hf_token: Optional[str]
    safetensor_files: list[str]
    tensor_to_file: dict[str, str]
    weight_map: Optional[dict[str, str]]


def _safe_repo_revision_key(repo_id: str, revision: str) -> str:
    digest = hashlib.sha1(f"{repo_id}@{revision}".encode("utf-8")).hexdigest()[:12]
    safe_repo = repo_id.replace("/", "__")
    safe_rev = re.sub(r"[^A-Za-z0-9._-]+", "_", revision)
    return f"{safe_repo}--{safe_rev}--{digest}"


def _safe_tensor_key(tensor_name: str) -> str:
    digest = hashlib.sha1(tensor_name.encode("utf-8")).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", tensor_name).strip("_")
    if not safe:
        safe = "tensor"
    return f"{safe}--{digest}"


def fp32_tensor_cache_dir(index: ModelIndex) -> Path:
    path = index.cache_dir / "tensor-fp32" / _safe_repo_revision_key(index.repo_id, index.revision)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_model_index(
    repo_or_url: str,
    revision: str = "main",
    cache_dir: str = "data/hf-cache",
) -> ModelIndex:
    repo_id = normalize_repo_id(repo_or_url)
    token = resolve_hf_token()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)

    weight_map: Optional[dict[str, str]] = None
    if "model.safetensors.index.json" in files:
        idx_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors.index.json",
            revision=revision,
            repo_type="model",
            cache_dir=str(cache_path),
            token=token,
        )
        with open(idx_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        raw_weight_map = index_data.get("weight_map", {})
        if isinstance(raw_weight_map, dict):
            weight_map = {str(k): str(v) for k, v in raw_weight_map.items()}

    if weight_map:
        safetensor_files = sorted(set(weight_map.values()))
    else:
        safetensor_files = sorted(
            [name for name in files if name.endswith(".safetensors") and not name.endswith(".safetensors.index.json")]
        )
    if not safetensor_files:
        raise RuntimeError(f"No .safetensors files found for repo '{repo_id}'.")

    tensor_to_file: dict[str, str] = {}
    if weight_map:
        tensor_to_file.update(weight_map)
    else:
        fs = HfFileSystem(token=token)
        for filename in safetensor_files:
            header = _read_safetensors_header_remote(fs=fs, repo_id=repo_id, filename=filename, revision=revision)
            for tensor_name, meta in header.items():
                if tensor_name == "__metadata__":
                    continue
                if not isinstance(meta, dict):
                    continue
                if tensor_name not in tensor_to_file:
                    tensor_to_file[tensor_name] = filename

    return ModelIndex(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_path,
        hf_token=token,
        safetensor_files=safetensor_files,
        tensor_to_file=tensor_to_file,
        weight_map=weight_map,
    )


def _infer_block_shape(tensor_shape: tuple[int, ...], scale_shape: tuple[int, ...]) -> tuple[int, ...]:
    out = []
    for ts, ss in zip(tensor_shape, scale_shape):
        if ss <= 0:
            out.append(1)
            continue
        out.append(max(1, int(np.ceil(float(ts) / float(ss)))))
    return tuple(out)


def _dequantize_tensor_with_scale_inv(tensor, inv_scale):
    assert tensor.ndim == inv_scale.ndim
    block_shape = _infer_block_shape(tuple(tensor.shape), tuple(inv_scale.shape))
    for i, block_dim in enumerate(block_shape):
        inv_scale = inv_scale.repeat_interleave(block_dim, dim=i)
    slices = tuple(slice(0, int(s)) for s in tensor.shape)
    return tensor.float() * inv_scale[slices].float()


def _load_raw_tensor(index: ModelIndex, tensor_name: str):
    filename = index.tensor_to_file.get(tensor_name)
    if filename is None:
        raise KeyError(f"Tensor '{tensor_name}' not found in repo '{index.repo_id}'.")

    local_file = hf_hub_download(
        repo_id=index.repo_id,
        filename=filename,
        revision=index.revision,
        repo_type="model",
        cache_dir=str(index.cache_dir),
        token=index.hf_token,
    )

    from safetensors.torch import safe_open as safe_open_t

    with safe_open_t(local_file, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        if tensor_name not in keys:
            raise KeyError(f"Tensor '{tensor_name}' missing in file '{filename}'.")
        return f.get_tensor(tensor_name)


def load_tensor_fp32(index: ModelIndex, tensor_name: str) -> np.ndarray:
    return load_tensor_fp32_cached(index=index, tensor_name=tensor_name, use_cache=True)


def load_tensor_fp32_cached(index: ModelIndex, tensor_name: str, use_cache: bool = True) -> np.ndarray:
    import torch

    cache_file = fp32_tensor_cache_dir(index) / f"{_safe_tensor_key(tensor_name)}.npy"
    if use_cache and cache_file.exists():
        return np.load(cache_file)

    out: np.ndarray
    if tensor_name.endswith("_fp32"):
        try:
            t = _load_raw_tensor(index, tensor_name)
            out = t.to(dtype=torch.float32).cpu().numpy()
            if use_cache:
                np.save(cache_file, out)
            return out
        except Exception:
            base = tensor_name[:-5]
            scale_name = f"{base}_scale_inv"
            w = _load_raw_tensor(index, base)
            s = _load_raw_tensor(index, scale_name)
            w = _dequantize_tensor_with_scale_inv(w, s)
            out = w.to(dtype=torch.float32).cpu().numpy()
            if use_cache:
                np.save(cache_file, out)
            return out

    # For quantized checkpoints that use separate inverse scale tensors, automatically
    # reconstruct fp32 from {name, name_scale_inv} when both are present.
    scale_name = f"{tensor_name}_scale_inv"
    if scale_name in index.tensor_to_file and not tensor_name.endswith("_scale_inv"):
        w = _load_raw_tensor(index, tensor_name)
        s = _load_raw_tensor(index, scale_name)
        w = _dequantize_tensor_with_scale_inv(w, s)
        out = w.to(dtype=torch.float32).cpu().numpy()
        if use_cache:
            np.save(cache_file, out)
        return out

    t = _load_raw_tensor(index, tensor_name)
    out = t.to(dtype=torch.float32).cpu().numpy()
    if use_cache:
        np.save(cache_file, out)
    return out


def resolve_selected_tensors(index: ModelIndex, filter_query: Optional[str]) -> list[str]:
    all_names = list(index.tensor_to_file.keys())
    weight_like = [
        n for n in all_names
        if "weight" in n.lower() and not n.lower().endswith("_scale_inv")
    ]
    selected = filter_tensor_names(weight_like if weight_like else all_names, filter_query)
    if not selected:
        selected = filter_tensor_names(all_names, filter_query)
    if not selected:
        raise RuntimeError("No tensors matched the filter query.")
    return selected


def warmup_tensor_cache(index: ModelIndex, tensor_names: list[str], use_cache: bool = True) -> list[Path]:
    outputs: list[Path] = []
    for name in tensor_names:
        arr = load_tensor_fp32_cached(index=index, tensor_name=name, use_cache=use_cache)
        if use_cache:
            cache_file = fp32_tensor_cache_dir(index) / f"{_safe_tensor_key(name)}.npy"
            if cache_file.exists():
                outputs.append(cache_file)
        else:
            _ = arr
    return outputs


def resolve_format_list(values: Optional[list[str]], supported: list[str]) -> list[str]:
    if not values:
        return supported
    seen = set()
    out: list[str] = []
    for raw in values:
        v = raw.strip().lower()
        if v == "all":
            for s in supported:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            continue
        if v not in supported:
            raise ValueError(f"Unsupported format '{raw}'. Supported: {', '.join(supported)}, all")
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out
