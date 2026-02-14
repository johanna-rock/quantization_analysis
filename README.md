# quantization-plots

Quantization visualization tools for any Hugging Face model repo with `.safetensors` weights.

## Scripts

- `compare_reconstr_error_synth_data.py`
  - Plots `amax -> reconstructed value` staircase curves.
  - Includes synthetic curves only (`BF16`, `BFP8`, `BFP4`, `BFP2`, `FP0`).
- `compare_reconstr_error_weights.py`
  - Plots one figure per matched tensor over that tensor's real min/max range.
  - Compares `Ideal` vs quantized reconstructions (`BF16`, `BFP8`, `BFP4`, `BFP2`, `FP0`).
- `wq`
  - wa-like quantization analyzer for matched tensors.
  - Reports per-format `pcc`, `mae`, and `atol` using the same emulation as the plotting scripts.
- `quantization_formats.py`
  - Shared quantization format definitions/emulation used by all quantization scripts.
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
[repo_or_url] [filter_query] [--revision REVISION] [-c FORMAT]
```

- `repo_or_url`: HF repo id or HF model URL.
- `filter_query`: optional tensor filter.
  - Dotted query (example: `model.layers.0`) = torch-style prefix match.
  - Non-dotted query (example: `q_proj`) = substring match.
- `--revision`: HF revision (default `main`).
- `-c FORMAT`: repeatable format selector; use `-c all` for all supported formats.

`compare_reconstr_error_synth_data.py`:

```bash
[-c FORMAT] [--rand-samples N]
```

`wq`:

```bash
[repo_or_url] [filter_query] [--revision REVISION] [-c FORMAT]
```

## Usage

### 1) Amax reconstruction plot

```bash
python compare_reconstr_error_synth_data.py -c all --rand-samples 100
```

### 2) Per-weight-range plots

```bash
python compare_reconstr_error_weights.py deepseek-ai/DeepSeek-R1 model.layers.0.self_attn --revision main -c all
```

Write PNGs to a custom folder:

```bash
python compare_reconstr_error_weights.py deepseek-ai/DeepSeek-R1 model.layers.0.self_attn \
  --revision main -c all --out-dir plots/visualize_quantization_error
```

Show interactively:

```bash
python compare_reconstr_error_weights.py deepseek-ai/DeepSeek-R1 model.layers.0.self_attn \
  --revision main -c all --show
```

### 3) wa-like quantization report

```bash
python ./wq deepseek-ai/DeepSeek-R1 model.layers.0.self_attn --revision main -c all --limit 5
```

Use emulation backend (default):

```bash
python ./wq deepseek-ai/DeepSeek-R1 model.layers.0.self_attn --revision main -c bfp8 -c bfp4 -c bfp2 --backend emulation
```

Use TTNN roundtrip backend for BFP formats:

```bash
python ./wq deepseek-ai/DeepSeek-R1 model.layers.0.self_attn --revision main -c bfp8 -c bfp4 -c bfp2 --backend ttnn
```

Notes:
- `--backend ttnn` requires `ttnn` in the active Python environment.
- With `--backend ttnn`, only `bfp8`, `bfp4`, and `bfp2` use TTNN conversion; other formats still use emulation.

Open all generated PNGs on macOS:

```bash
open plots/visualize_quantization_error/*.png
```

## Notes

- The loader supports indexed and non-indexed safetensors repos.
- If a tensor has a matching `*_scale_inv`, the loader dequantizes to fp32 automatically.
- All scripts share the same default cache root: `data/hf-cache`.
- Downloaded model shards and dequantized fp32 tensors are reused across scripts.
