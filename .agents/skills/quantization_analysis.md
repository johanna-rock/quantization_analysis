# Quantization Analysis

## Environment
- Activate the project venv before running scripts:
  - `source .venv/bin/activate`
  - Use `python` from the venv (not system python).

## sweep_mixed_tile_threshold.py usage notes
- Regex arguments must be shell-quoted to avoid `(`, `|`, `*`, `?` parsing by the shell.
- Hugging Face repo IDs use a slash, e.g. `deepseek-ai/DeepSeek-R1` (not `deepseek-ai__DeepSeek-R1`).

## DeepSeek-R1 common regex
- Attention weights (broad): `^model\.layers\.\d+\.self_attn\..*weight$`
- Attention weights (q/k/v/o only): `^model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$`
- Embedding + lm_head: `^(model\.embed_tokens\.weight|lm_head\.weight)$`

## Example (list matches)
```bash
source .venv/bin/activate
python scripts/sweep_mixed_tile_threshold.py deepseek-ai/DeepSeek-R1 '^(model\\.embed_tokens\\.weight|lm_head\\.weight)$' --list-matches
```
