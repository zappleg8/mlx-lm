# CLAUDE.md — mlx-lm fork (zaya1 branch)

This is a fork of [`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm) for the ZAYA1-8B port.

## Where the actual context lives

This repo holds **only** the model code. Everything else — design doc, reference validation harness, HF upload scripts, status tracking — lives in `~/code/personal/zaya1-mlx`.

**For context, design, status, or workflow questions, read `~/code/personal/zaya1-mlx/CLAUDE.md` first.**

## What's added here

- `mlx_lm/models/zaya.py` (forthcoming) — the full ZAYA1 model port.
- `mlx_lm/models/__init__.py` — registers `"zaya"` → `zaya.Model`.

That's it. No tests, no docs, no harness in this repo. The intent is for this branch to become a clean upstream PR to `ml-explore/mlx-lm`.

## Working norms

- Match upstream conventions exactly (`ModelArgs` dataclass, `Model` class structure, `sanitize(weights)` function naming).
- Study `mlx_lm/models/jamba.py` and `mlx_lm/models/mamba2.py` for hybrid Mamba+Attention patterns and SSM scan style.
- Numerical parity gates are enforced in the sibling repo's `validation/` harness.
- Git authorship uses the GitHub noreply email — this branch will eventually be pushed and PR'd publicly.

## Branches

- `main` — tracks `upstream/main`. Do not commit here.
- `zaya1` — active work for the port.
