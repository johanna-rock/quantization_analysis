# Compression Configs

This folder contains example JSON configs for `wq --compression-config`.

Implemented algorithms are registered in `compression_algorithms/__init__.py`:
- `none`
- `transpose`
- `mixed-tile-threshold`
- `mixed-tile-greedy`
- `mixed-tile-random`
- `mixed-tile` (alias for `mixed-tile-greedy`)

## Config Schema

Minimal config:

```json
{
  "algorithm": "none",
  "params": {}
}
```

Full schema used by `compression_algorithms/config.py`:

```json
{
  "algorithm": "mixed-tile-threshold",
  "quantization_formats": ["bf16", "bfp8", "bfp4", "bfp2", "fp0"],
  "seed": 123,
  "random_seed": false,
  "params": {
    "metric": "pcc",
    "threshold": 0.94
  }
}
```

Field semantics:
- `algorithm`: string, default `none`.
- `params`: object, default `{}`.
- `quantization_formats`: optional list of format names. If omitted, `wq` uses all supported formats from `quantization_formats.py`:
  - `mxfp4`, `nvfp4`, `bf16`, `bfp8`, `bfp4`, `bfp2`, `fp0`
- `seed`: optional int, `0`, or `"random"`.
- `random_seed`: optional bool.

Seed behavior in `wq`:
- If `seed` is a non-zero int in config, that value is used.
- If `seed` is `0` or `"random"`, a fresh random seed is generated.
- Else if `random_seed: true`, a fresh random seed is generated.
- Else if `params.seed` exists, it is used (and `0` in `params.seed` is also treated as random).
- The resolved seed is written to `results/.../compression_config.used.json` as:
  - `seed`
  - `seed_source` (`config`, `params`, or `random`)

Notes:
- `wq` always runs `none` baseline plus your selected algorithm (except when selected algorithm is `none`).
- `quantization_formats` controls baseline rows and default candidate formats for mixed-tile algorithms.
- For mixed-tile algorithms, `params.formats` (or `params.tile_formats`) overrides mixed-tile candidate formats if provided.

## Algorithm Reference

### `none`

What it does:
- Quantizes the original tensor directly for each format in `quantization_formats`.

Params:
- None required.
- Extra params are ignored.

When to use:
- Baseline error/size comparison without tensor transform or tile assignment.

---

### `transpose`

What it does:
- Applies `np.transpose(x)` first, quantizes the transposed tensor, then transposes back.
- For 2D tensors this is a standard matrix transpose.
- For N-D tensors this reverses axis order (`np.transpose` default behavior).

Params:
- None required.
- Extra params are ignored.

When to use:
- Quick check whether transposed memory/layout orientation affects quantization error.

---

### `mixed-tile-threshold`

What it does:
- Reshapes input to 2D and pads to multiples of `32x32`.
- Quantizes tensor once per candidate tile format.
- For each tile, chooses the lowest-byte format that satisfies threshold:
  - `pcc`: score must be `>= threshold`
  - `mae`/`atol`: score must be `<= threshold`
- If no format passes, falls back to highest-precision candidate format for that tile.
- Reconstructs tensor from mixed assignments and reports one `MIXED` result row.

Supported mixed-tile formats:
- `bf16`, `bfp8`, `bfp4`, `bfp2`

Params:
- `metric`: `pcc` | `mae` | `atol` (default: `pcc`)
- `threshold`: float (default: `0.999`)
- `formats` or `tile_formats`: optional
  - comma-separated string or list
  - must be subset of `bf16,bfp8,bfp4,bfp2`

Output artifacts (under `results/<model>/mixed-tile-threshold/<timestamp>/`):
- `table.txt`
- `compression_config.used.json`
- `mixed_tile_threshold/<tensor>/assignment.npy`
- `mixed_tile_threshold/<tensor>/assignment_mapping.json`
- `mixed_tile_threshold/<tensor>/<tensor>_assignment.png`
- `mixed_tile_threshold/<tensor>/size_vs_accuracy.png`

---

### `mixed-tile-greedy`

What it does:
- Uses `32x32` tiles on padded 2D representation, same as threshold mode.
- Starts all tiles in a base format:
  - first entry of tile format list (order matters)
- Iterates candidate formats and attempts per-tile replacements in random order.
- Accepts a replacement only if global metric remains within threshold.
- Produces one final `MIXED` assignment.

Supported mixed-tile formats:
- `bf16`, `bfp8`, `bfp4`, `bfp2`

Params:
- `metric`: `pcc` | `mae` | `atol` (default: `pcc`)
- `threshold`: float (default: `0.999`)
- `seed`: int (default: `0`)
  - `0` means random seed at runtime
- `formats` or `tile_formats`: optional
  - comma-separated string or list
  - must be subset of `bf16,bfp8,bfp4,bfp2`

Important detail:
- Because greedy starts from `tile_formats[0]`, format order changes behavior and final mix.

Output artifacts (under `results/<model>/mixed-tile-greedy/<timestamp>/`):
- `table.txt`
- `compression_config.used.json`
- `mixed_tile_greedy/<tensor>/assignment.npy`
- `mixed_tile_greedy/<tensor>/assignment_mapping.json`
- `mixed_tile_greedy/<tensor>/<tensor>_assignment.png`
- `mixed_tile_greedy/<tensor>/size_vs_accuracy.png`

---

### `mixed-tile-random`

What it does:
- Uses `32x32` tiles on padded 2D representation.
- Randomly samples tile-format assignments for `iters` trials.
- For each sample, computes metric and size.
- Selection rule:
  - Prefer smallest-size sample that satisfies threshold.
  - If none satisfy threshold, choose best-metric sample.
- Returns one final `MIXED` result row plus sample diagnostics.

Supported mixed-tile formats:
- `bf16`, `bfp8`, `bfp4`, `bfp2`

Params:
- `metric`: `pcc` | `mae` | `atol` (default: `pcc`)
- `threshold`: float (default: `0.999`)
- `iters`: integer >= 1 (default: `50`)
- `seed`: int (default: `0`)
  - deterministic unless overridden via config-level random seed handling
- `formats`: optional
  - comma-separated string or list
  - subset of `bf16,bfp8,bfp4,bfp2`

Output artifacts (under `results/<model>/mixed-tile-random/<timestamp>/`):
- `table.txt`
- `compression_config.used.json`
- `mixed_tile_random/<tensor>.csv` (all sampled points)
- `mixed_tile_random/<tensor>.png` (PCC vs size scatter)
- `mixed_tile_random/<tensor>_assignment.npy`
- `mixed_tile_random/<tensor>_assignment_mapping.json`

## Metric and Size Conventions

Metric direction:
- `pcc`: higher is better, threshold is minimum acceptable value.
- `mae`: lower is better, threshold is maximum acceptable value.
- `atol`: lower is better, threshold is maximum acceptable value.

Mixed-tile size model (`compression_algorithms/tile_utils.py`):
- Tile size: `32 x 32 = 1024` elements.
- Bytes per element:
  - `bf16`: `2.0`
  - `bfp8`: `1.088`
  - `bfp4`: `0.50097`
  - `bfp2`: `0.25097`

## Example Files in This Folder

- `compression_config.transpose.example.json`
- `compression_config.mixed_tile_threshold.example.json`
- `compression_config.mixed_tile_greedy.example.json`
- `compression_config.mixed_tile_random.example.json`
