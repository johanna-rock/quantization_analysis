#!/usr/bin/env python3
"""
Download DeepSeek-R1 shards needed for selected tensors and optionally build
a local dequantized fp32 safetensors file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import safe_open, save_file


def read_tensor_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Tensor list file not found: {path}")
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "|" in s:
            s = s.split("|", 1)[0].strip()
        out.append(s)
    return out


def dequantize_tensor(tensor: torch.Tensor, inv_scale: torch.Tensor, block_shape: tuple[int, int]) -> torch.Tensor:
    assert tensor.ndim == inv_scale.ndim
    assert len(block_shape) == tensor.ndim and all(
        inv_scale.shape[i] * block_shape[i] >= tensor.shape[i] for i in range(tensor.ndim)
    )
    for i, block_dim in enumerate(block_shape):
        inv_scale = inv_scale.repeat_interleave(block_dim, dim=i)
    return tensor.float() * inv_scale[tuple(slice(0, s) for s in tensor.shape)].float()


def load_tensor_from_weight_map(
    tensor_name: str,
    weight_map: dict[str, str],
    repo_id: str,
    cache_dir: Path,
    shard_cache: dict[str, Path],
) -> torch.Tensor:
    shard = weight_map.get(tensor_name)
    if shard is None:
        raise KeyError(f"Tensor '{tensor_name}' not found in index")
    if shard not in shard_cache:
        path = hf_hub_download(repo_id=repo_id, filename=shard, cache_dir=str(cache_dir))
        shard_cache[shard] = Path(path)
    shard_path = shard_cache[shard]
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        if tensor_name not in set(f.keys()):
            raise KeyError(f"Tensor '{tensor_name}' missing in shard '{shard}'")
        return f.get_tensor(tensor_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DeepSeek-R1 tensors used by this repo.")
    parser.add_argument("--repo-id", type=str, default="deepseek-ai/DeepSeek-R1")
    parser.add_argument("--cache-dir", type=str, default="data/deepseek-r1")
    parser.add_argument("--tensors-file", type=str, default="ds_tensors.txt")
    parser.add_argument(
        "--output-fp32-file",
        type=str,
        default="data/deepseek-r1/deepseek-r1-layer0-fp32.safetensors",
        help="Write dequantized tensors here. Use --no-build-fp32 to skip.",
    )
    parser.add_argument("--no-build-fp32", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensors = read_tensor_list(Path(args.tensors_file))
    if not tensors:
        raise SystemExit("No active tensors found in tensors file.")

    index_path = hf_hub_download(
        repo_id=args.repo_id,
        filename="model.safetensors.index.json",
        cache_dir=str(cache_dir),
    )
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map: dict[str, str] = index.get("weight_map", {})

    needed_keys: set[str] = set()
    for name in tensors:
        if name.endswith("_fp32"):
            base = name[:-5]
            needed_keys.add(base)
            needed_keys.add(f"{base}_scale_inv")
            if name in weight_map:
                needed_keys.add(name)
        else:
            needed_keys.add(name)

    needed_shards: set[str] = set()
    missing: list[str] = []
    for key in sorted(needed_keys):
        shard = weight_map.get(key)
        if shard is None:
            missing.append(key)
        else:
            needed_shards.add(shard)

    if missing:
        print("Warning: some requested keys are not in DeepSeek index:")
        for m in missing:
            print(f"  - {m}")

    shard_cache: dict[str, Path] = {}
    for shard in sorted(needed_shards):
        path = hf_hub_download(repo_id=args.repo_id, filename=shard, cache_dir=str(cache_dir))
        shard_cache[shard] = Path(path)
        print(f"Downloaded {shard}")

    if args.no_build_fp32:
        print("Done. Shards downloaded.")
        return

    out_tensors: dict[str, torch.Tensor] = {}
    for name in tensors:
        if name.endswith("_fp32"):
            if name in weight_map:
                t = load_tensor_from_weight_map(name, weight_map, args.repo_id, cache_dir, shard_cache)
                out_tensors[name] = t.float().cpu()
                print(f"Loaded fp32 tensor {name}")
            else:
                base = name[:-5]
                scale_name = f"{base}_scale_inv"
                w = load_tensor_from_weight_map(base, weight_map, args.repo_id, cache_dir, shard_cache)
                s = load_tensor_from_weight_map(scale_name, weight_map, args.repo_id, cache_dir, shard_cache)
                out_tensors[name] = dequantize_tensor(w, s, block_shape=(128, 128)).float().cpu()
                print(f"Dequantized {name} from {base} + {scale_name}")
        else:
            t = load_tensor_from_weight_map(name, weight_map, args.repo_id, cache_dir, shard_cache)
            out_tensors[name] = t.float().cpu()
            print(f"Loaded tensor {name}")

    out_path = Path(args.output_fp32_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, str(out_path))
    print(f"Wrote {out_path} with {len(out_tensors)} tensor(s)")


if __name__ == "__main__":
    main()
