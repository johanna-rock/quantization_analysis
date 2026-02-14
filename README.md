# quantization-plots

Small research scripts for visualizing low-precision quantization behavior, with optional DeepSeek-R1 weight-driven curves.

## What this repo contains

- `quantization_plot.py`: plots `amax -> reconstructed value` staircase curves for:
  - `MXFP4`, `NVFP4`, `BF16`
  - `BFP8`, `BFP4`, `BFP2` (ideal exponent and rand16 exponent modes)
  - optional DeepSeek-based BFP4 curves from selected tensors
- `quantization_error_weights.py`: plots per-tensor quantization transfer curves over each tensor’s real value range (`min(weight)` to `max(weight)`) for:
  - `Ideal`, `BF16`, `BFP8`, `BFP4`, `BFP2`, `FP0`
  - writes one plot image per tensor
- `download_deepseek_weights.py`: downloads required DeepSeek shards for tensors in `ds_tensors.txt` and can build a local dequantized fp32 safetensors file
- `ds_tensors.txt`: list of DeepSeek tensor names to include (`tensor_name | Label` format)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 0) Reproducible data bootstrap (if `data/` is missing)

Set your HF token if needed (recommended for higher rate limits):

```bash
export HF_TOKEN=your_token_here
```

Download the required DeepSeek shards and build
`data/deepseek-r1/deepseek-r1-layer0-fp32.safetensors`

```bash
python download_deepseek_weights.py \
  --cache-dir data/deepseek-r1 \
  --tensors-file ds_tensors.txt \
  --output-fp32-file data/deepseek-r1/deepseek-r1-layer0-fp32.safetensors
```

If you only want to download shards (skip local fp32 file creation):

```bash
python download_deepseek_weights.py \
  --cache-dir data/deepseek-r1 \
  --tensors-file ds_tensors.txt \
  --no-build-fp32
```

### 1) Amax reconstruction plot

```bash
python quantization_plot.py \
  --ds-weights-path data/deepseek-r1/deepseek-r1-layer0-fp32.safetensors \
  --ds-tensors-file ds_tensors.txt
```

Notes:
- If a requested `*_fp32` tensor is not present in the local file, the script falls back to loading base tensor + `*_scale_inv` from DeepSeek shards and dequantizes automatically.
- You can toggle lines in the plot via the checkbox panel.

### 2) Per-weight-range error plots (one output file per tensor)

```bash
python quantization_error_weights.py \
  --ds-weights-path data/deepseek-r1/deepseek-r1-layer0-fp32.safetensors \
  --ds-tensors-file ds_tensors.txt \
  --out-dir plots/quantization_error_weights
```

Optional interactive display:

```bash
python quantization_error_weights.py \
  --ds-weights-path data/deepseek-r1/deepseek-r1-layer0-fp32.safetensors \
  --ds-tensors-file ds_tensors.txt \
  --out-dir plots/quantization_error_weights \
  --show
```

To open all generated plots on macOS:

```bash
open plots/quantization_error_weights/*.png
```

## Data assumptions

- Local cache/data is expected under `data/deepseek-r1/` (or set your own path with `--ds-weights-path`).
- `ds_tensors.txt` should include tensor names that exist in the selected model files/index.
