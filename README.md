# quantization-plots

Quantization visualization tools for any Hugging Face model repo with `.safetensors` weights.

## Scripts

- `compare_reconstr_error_synth_data.py`
  - Plots `amax -> reconstructed value` staircase curves.
  - Includes synthetic curves only (`BF16`, `BFP8`, `BFP4`, `BFP2`, `FP0`).
  - BFP “ideal” curves use per-element exponents; BFP “rand” curves use TTNN-style shared-exponent rows.
- `compare_reconstr_error_weights.py`
  - Plots one figure per matched tensor over that tensor's real min/max range.
  - Compares `Ideal` vs quantized reconstructions (`BF16`, `BFP8`, `BFP4`, `BFP2`, `FP0`).
- `wq`
  - wa-like quantization analyzer for matched tensors.
  - Reports per-format `pcc`, `mae`, and `atol` using the same emulation as the plotting scripts.
- `quantization_formats.py`
  - Shared quantization format definitions/emulation used by all quantization scripts.
  - BFP emulation matches TTNN packing (shared exponent per 16-element row in 32x32 tiles).
- `wa`
  - Tensor explorer used as CLI style inspiration.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional:
- `wq --backend ttnn` requires a separate `ttnn` install from Tenstorrent tooling; it is intentionally not in `requirements.txt`.

Optional for higher HF rate limits:

```bash
export HF_TOKEN=your_token_here
```

## CLI Pattern

`compare_reconstr_error_weights.py`:

```bash
[repo_or_url] [filter_query] [--revision REVISION]
```

- `repo_or_url`: HF repo id or HF model URL.
- `filter_query`: optional tensor filter.
  - Dotted query (example: `model.layers.0`) = torch-style prefix match.
  - Non-dotted query (example: `q_proj`) = substring match.
- `--revision`: HF revision (default `main`).
- `-c FORMAT`: repeatable format selector; use `-c all` for all supported formats.

`compare_reconstr_error_synth_data.py`:

```bash
[-c FORMAT] [--rand-samples N] [--out PATH] [--no-show]
```

`wq`:

```bash
[repo_or_url] [filter_query] [--revision REVISION] [-c FORMAT]
```

Optional flags:
- `--compression-config PATH`: JSON config describing the compression algorithm and its params.
- `--recompute`: recompute and overwrite cached quantized tensors.
- `--summary`: print the aggregate summary (default: off).

Example compression config (`compression_configs/compression_config.mixed_tile_greedy.example.json`):

```json
{
  "algorithm": "mixed-tile-greedy",
  "quantization_formats": [
    "mxfp4",
    "nvfp4",
    "bf16",
    "bfp8",
    "bfp4",
    "bfp2",
    "fp0"
  ],
  "params": {
    "metric": "pcc",
    "threshold": 0.999,
    "cluster": "kmeans",
    "k": 10
  }
}
```

Supported algorithms: `none`, `transpose`, `mixed-tile-greedy`, `mixed-tile-random`.
If `quantization_formats` is omitted, all formats are used by default.
For mixed-tile algorithms, `quantization_formats` is intersected with `bf16,bfp8,bfp4,bfp2`.
When a config is provided, `wq` runs the selected algorithm alongside the `none` baseline.
Optional `seed` can be an integer or the string `random`; the used seed is recorded in `compression_config.used.json`.

Other example configs:
- `compression_configs/compression_config.transpose.example.json`
- `compression_configs/compression_config.mixed_tile_random.example.json`

## Usage

### 1) Amax reconstruction plot

```bash
python compare_reconstr_error_synth_data.py -c all --rand-samples 100
```

Headless (save PNG instead of showing a window):

```bash
python compare_reconstr_error_synth_data.py -c all --rand-samples 100 --out plots/compare_reconstr_error_synth_data.png --no-show
```

### 2) Per-weight-range plots

```bash
python compare_reconstr_error_weights.py deepseek-ai/DeepSeek-R1 model.layers.0.self_attn.kv_a_layernorm.weight -c all
```

Write PNGs to a custom folder:

```bash
python compare_reconstr_error_weights.py deepseek-ai/DeepSeek-R1 model.layers.0.self_attn.kv_a_layernorm.weight \
  --revision main -c all --out-dir plots/visualize_quantization_error
```

Show interactively:

```bash
python compare_reconstr_error_weights.py deepseek-ai/DeepSeek-R1 model.layers.0.self_attn \
  -c all --show
```

### 3) CLI Quantization report

```bash
python ./wq deepseek-ai/DeepSeek-R1 model.layers.0.self_attn --limit 10
```

Use emulation backend (default):

```bash
python ./wq deepseek-ai/DeepSeek-R1 model.layers.0.self_attn --backend emulation
```

Compare compression modes:

```bash
python ./wq deepseek-ai/DeepSeek-R1 model.layers.0.self_attn --compression-config compression_configs/compression_config.mixed_tile_greedy.example.json
```

Use TTNN roundtrip backend for BFP formats:

```bash
python ./wq deepseek-ai/DeepSeek-R1 model.layers.0.self_attn --backend ttnn
```

Notes:
- `--backend ttnn` requires `ttnn` in the active Python environment.
- With `--backend ttnn`, only `bfp8` and `bfp4` use TTNN conversion. `bfp2` and other formats still use emulation.
- Each run writes a `results/<model>/<algorithm>/<timestamp>/` folder with `table.txt` and a copy of the config used. Mixed-tile-random also emits per-tensor CSV/PNG under `results/.../mixed_tile_random`.
- Mixed-tile-random also writes per-tensor assignment maps (`*_assignment.npy`) plus a JSON mapping file, which can be reconstructed with `scripts/reconstruct_mixed_tile_assignment.py`.

Open all generated PNGs on macOS:

```bash
open plots/visualize_quantization_error/*.png
```

## Notes

- The loader supports indexed and non-indexed safetensors repos.
- If a tensor has a matching `*_scale_inv`, the loader dequantizes to fp32 automatically.
- All scripts share the same default cache root: `data/hf-cache`.
- Downloaded model shards and dequantized fp32 tensors are reused across scripts.
