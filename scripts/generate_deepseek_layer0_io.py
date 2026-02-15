#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm


class StopForward(Exception):
    pass


@dataclass
class SampleMeta:
    idx: int
    split: str
    prompt: str
    input_ids: list[int]
    attention_mask: list[int] | None


def _add_tt_metal_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _ensure_no_init_weights() -> None:
    try:
        import transformers.modeling_utils as modeling_utils  # type: ignore
    except Exception:
        return
    if hasattr(modeling_utils, "no_init_weights"):
        return

    from contextlib import contextmanager
    import torch.nn.init as init

    @contextmanager
    def no_init_weights():
        def _no_init(*_args, **_kwargs):
            return None

        patched = {}
        for name in (
            "uniform_",
            "normal_",
            "constant_",
            "xavier_uniform_",
            "xavier_normal_",
            "kaiming_uniform_",
            "kaiming_normal_",
            "trunc_normal_",
        ):
            if hasattr(init, name):
                patched[name] = getattr(init, name)
                setattr(init, name, _no_init)
        try:
            yield
        finally:
            for name, fn in patched.items():
                setattr(init, name, fn)

    setattr(modeling_utils, "no_init_weights", no_init_weights)


def _ensure_transformers_import_utils() -> None:
    try:
        from transformers.utils import import_utils
    except Exception:
        return
    if not hasattr(import_utils, "is_torch_fx_available"):
        def is_torch_fx_available() -> bool:
            return False
        setattr(import_utils, "is_torch_fx_available", is_torch_fx_available)


def _load_prompts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], dict) and "prompt" in data[0]:
            return [str(item["prompt"]) for item in data]
        if isinstance(data[0], str):
            return [str(item) for item in data]
    raise ValueError(f"Unsupported prompts file format at {path}")


def _select_prompts(prompts: list[str], num_samples: int, seed: int) -> list[str]:
    if not prompts:
        raise ValueError("No prompts available.")
    rng = random.Random(seed)
    if num_samples <= len(prompts):
        indices = rng.sample(range(len(prompts)), num_samples)
    else:
        indices = [rng.randrange(len(prompts)) for _ in range(num_samples)]
    return [prompts[i] for i in indices]


def _has_weight_param(module: torch.nn.Module) -> bool:
    for name, _ in module.named_parameters(recurse=False):
        if name == "weight":
            return True
    return False


def _sanitize(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_sanitize(item) for item in obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _save_sample(
    out_dir: Path, op_name: str, split: str, sample_idx: int, entry: tuple[tuple, dict[str, Any], Any]
) -> None:
    args, kwargs, output = entry
    op_dir = out_dir / op_name.replace(".", "/") / split
    op_dir.mkdir(parents=True, exist_ok=True)
    path = op_dir / f"sample_{sample_idx:04d}.pt"
    payload = {
        "args": _sanitize(args),
        "kwargs": _sanitize(kwargs),
        "output": _sanitize(output),
        "sample_idx": sample_idx,
        "split": split,
    }
    torch.save(payload, path)


def _sample_exists(out_dir: Path, op_names: Iterable[str], split: str, sample_idx: int) -> bool:
    for op_name in op_names:
        path = out_dir / op_name.replace(".", "/") / split / f"sample_{sample_idx:04d}.pt"
        if not path.exists():
            return False
    return True


def _tokenize(tokenizer, prompt: str, max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)


def _run_model(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> None:
    kwargs = dict(inputs)
    kwargs["use_cache"] = False
    try:
        model(**kwargs)
    except StopForward:
        return
    except TypeError:
        kwargs.pop("use_cache", None)
        try:
            model(**kwargs)
        except StopForward:
            return


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate DeepSeek layer0 op IO pairs using the reference model.",
    )
    parser.add_argument(
        "--tt-metal-path",
        type=str,
        default="/home/jrock/wa/tt-metal",
        help="Path to tt-metal repo (default: /home/jrock/wa/tt-metal).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Local path to DeepSeek HF model weights and tokenizer files.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Output model name (default: basename of --model-path).",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="/home/jrock/wa/tt-metal/models/demos/deepseek_v3/demo/test_prompts.json",
        help="Path to a JSON file containing prompts.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Total number of samples to generate (default: 100).",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=None,
        help="Number of calibration samples (default: 70%% of num-samples).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max prompt length for tokenization (default: 128).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for prompt selection (default: 0).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/io_data",
        help="Output root directory (default: data/io_data).",
    )
    parser.add_argument(
        "--eager-weights",
        action="store_true",
        help="Load all weights eagerly instead of using LazyStateDict.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples that already exist on disk.",
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        default="model.layers.0",
        help="Module name to stop after (default: model.layers.0). Use empty string to disable.",
    )
    return parser.parse_args()


def main() -> int:
    args = create_parser()

    tt_metal_path = Path(args.tt_metal_path)
    _add_tt_metal_path(tt_metal_path)
    _ensure_no_init_weights()
    _ensure_transformers_import_utils()

    from models.demos.deepseek_v3.utils.hf_model_utils import (
        add_dynamic_weight_loading_hooks,
        load_model_uninitialized,
        load_model_weights,
        load_tokenizer,
    )
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict

    model_path = Path(args.model_path)
    model_name = args.model_name or model_path.name

    prompts = _load_prompts(Path(args.prompts_file))
    selected_prompts = _select_prompts(prompts, args.num_samples, args.seed)

    calib_size = args.calib_size
    if calib_size is None:
        calib_size = int(round(args.num_samples * 0.7))
    calib_size = max(0, min(calib_size, args.num_samples))

    out_root = Path(args.out_dir) / model_name
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(str(model_path))

    print("Loading model...")
    model = load_model_uninitialized(str(model_path))
    model.eval()

    if args.eager_weights:
        print("Loading weights eagerly (this may be very large)...")
        weights_dict = load_model_weights(str(model_path))
    else:
        print("Using LazyStateDict for weights...")
        weights_dict = LazyStateDict(model_path)

    add_dynamic_weight_loading_hooks(model, weights_dict)

    op_modules: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        if not name.startswith("model.layers.0"):
            continue
        if _has_weight_param(module):
            op_modules[name] = module

    if not op_modules:
        raise RuntimeError("No weighted modules found under model.layers.0")

    print(f"Found {len(op_modules)} weighted ops under model.layers.0")

    current_logs: dict[str, tuple[tuple, dict[str, Any], Any] | None] | None = None

    def make_hook(op_name: str):
        def hook(_module, args, kwargs, output):
            nonlocal current_logs
            if current_logs is None:
                return
            if current_logs[op_name] is None:
                current_logs[op_name] = (args, kwargs, output)
        return hook

    for op_name, module in op_modules.items():
        module.register_forward_hook(make_hook(op_name), with_kwargs=True)

    if args.stop_after:
        stop_module = dict(model.named_modules()).get(args.stop_after)
        if stop_module is None:
            raise ValueError(f"Unable to find module '{args.stop_after}' to stop after.")

        def _stop_hook(_module, _args, _kwargs, _output):
            raise StopForward()

        stop_module.register_forward_hook(_stop_hook, with_kwargs=True)

    samples_meta: list[SampleMeta] = []

    with torch.no_grad():
        progress = tqdm(selected_prompts, desc="Generating samples", unit="sample")
        start_time = time.perf_counter()
        for idx, prompt in enumerate(progress):
            split = "calibration" if idx < calib_size else "test"
            if args.skip_existing and _sample_exists(out_root, op_modules.keys(), split, idx):
                inputs = _tokenize(tokenizer, prompt, args.max_length)
                input_ids = inputs["input_ids"][0].tolist()
                attention_mask = inputs.get("attention_mask")
                attention_mask_list = attention_mask[0].tolist() if attention_mask is not None else None
                progress.set_postfix({"skipped": True})
                samples_meta.append(
                    SampleMeta(
                        idx=idx,
                        split=split,
                        prompt=prompt,
                        input_ids=input_ids,
                        attention_mask=attention_mask_list,
                    )
                )
                continue
            inputs = _tokenize(tokenizer, prompt, args.max_length)
            input_ids = inputs["input_ids"][0].tolist()
            attention_mask = inputs.get("attention_mask")
            attention_mask_list = attention_mask[0].tolist() if attention_mask is not None else None

            iter_start = time.perf_counter()
            current_logs = {name: None for name in op_modules}
            _run_model(model, inputs)
            iter_end = time.perf_counter()

            missing = [name for name, entry in current_logs.items() if entry is None]
            if missing:
                print(f"Warning: missing logs for {len(missing)} ops on sample {idx}")

            for op_name, entry in current_logs.items():
                if entry is None:
                    continue
                _save_sample(out_root, op_name, split, idx, entry)

            samples_meta.append(
                SampleMeta(
                    idx=idx,
                    split=split,
                    prompt=prompt,
                    input_ids=input_ids,
                    attention_mask=attention_mask_list,
                )
            )
            current_logs = None

            elapsed = iter_end - iter_start
            total_elapsed = iter_end - start_time
            avg = total_elapsed / (idx + 1)
            remaining = avg * (args.num_samples - idx - 1)
            progress.set_postfix(
                {
                    "sec/sample": f"{elapsed:.1f}",
                    "avg": f"{avg:.1f}",
                    "eta_min": f"{remaining / 60:.1f}",
                }
            )

    manifest = {
        "model_name": model_name,
        "model_path": str(model_path),
        "num_samples": args.num_samples,
        "calib_size": calib_size,
        "test_size": args.num_samples - calib_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "ops": sorted(op_modules.keys()),
        "samples": [sample.__dict__ for sample in samples_meta],
    }
    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {out_root / 'manifest.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
