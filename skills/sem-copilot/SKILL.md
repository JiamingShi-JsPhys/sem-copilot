---
name: sem-copilot
description: "Single-skill SEM Copilot. One loop: fit a structural-equation model (semopy + independent numpy cross-check), draw a publication-ready vector PDF path diagram, and export a compilable LaTeX manuscript. Standalone or inside Chelate Shell."
metadata:
  author: Jiaming Shi & Keqia Ying
  license: MIT
  version: 1.0.0
  entry_point: python scripts/run.py
  python_requires: '>=3.10'
version: 1.0.0
---

# sem-copilot — one skill, the whole SEM loop

> **No need to install the full Chelate Shell.** This single skill does the
> entire *fit → draw → export* loop, standalone, from one sentence. Say
> goodbye to dragging arrows in AMOS: describe your model, get a
> publication-ready PDF path diagram and a compilable `.tex`.

## Quick execution

Run the entry point with a `--action` that selects one of the three loop
stages:

```bash
# 1) FIT + GATE the model
python scripts/run.py --action gate \
    --root <WS> --model "f1 =~ x1 + x2 + x3
f2 =~ y1 + y2 + y3
f2 ~ f1" --data data/dataset.csv

# 2) RENDER: fit + draw the vector PDF/PNG path diagram (one command)
python scripts/run.py --action render --root <WS> \
    --input spec.json --title "Measurement + structural model"

# 3) EXPORT: merge Markdown reports + figures into one .tex
python scripts/run.py --action export --root <WS> --name report.tex
```

- `--action` ∈ `gate` | `render` | `export` (default `gate`). `process` is an
  alias for `export`.
- `--root <WS>` = the workspace. **It is optional in standalone mode** — when
  omitted (or when the directory it points at lacks the `material/` +
  `process/` zone pair), all outputs land in `<cwd>/sem-copilot-output/`
  instead. See *Output location* below.
- Input via `--input spec.json` (`{"model": "<lavaan string>",
  "data": "<csv path>"}`) **or** directly with `--model` + `--data`.
- **Sandboxed** (`sandbox_required: true`); a 120 s circuit-breaker forces a
  `UNVERIFIED` verdict on timeout — never a crash.

## What each action does

| Action | Does | Writes |
|---|---|---|
| `gate` | Fit the SEM with `semopy`, cross-check the fit indices with the independent `sem_fit` numpy engine, run the AMOS-style extension battery (`robust_ml`, `categorical_wlsmv`, `mediation_effects`, `invariance_test`, `modification_index`, `bootstrap_ci`), and gate on CFI/TLI/RMSEA. | `<base>/process/gates/sem-inspection.json`, `sem-fit-indices.json`, + the gate JSON |
| `render` | Fit the model, emit a figure `sem_path` JSON spec, **and immediately draw** the vector PDF + raster PNG path diagram. One command = fit → draw. | `<base>/process/figures/sem_path_input.json`, `sem_path.pdf`, `sem_path.png` |
| `export` | Merge every Markdown report under `<base>/process/` and every graphic under `<base>/process/figures/` into a single compilable `.tex`. Pure stdlib, no pandoc. | `<base>/process/latex/<name>` |

`<base>` is the detected output root (see *Output location*).

## Independent re-derivation (AMOS replacement, no graphviz)

`gate` does not trust a single package. Alongside `semopy` it loads
`scripts/sem_fit.py`, a **pure-numpy** implementation of the non-recursive
continuous-SEM likelihood chi-square (factor-analytic / reduced-rank form,
L-BFGS-B on the free loadings, factor covariances and residual variances). It
deliberately depends on neither `semopy` nor `graphviz`, so the fit indices are
re-derivable from the raw covariance matrix by an independent implementation —
exactly the assurance commercial AMOS promises, in permissively-licensed,
readable code.

Cross-check logic: `semopy` PASS + `sem_fit` agrees → **PASS** (cross-
verified); `semopy` FAIL + `sem_fit` agrees → **FAIL**; a genuine divergence
downgrades to **UNVERIFIED** so a reviewer is alerted rather than trusting one
engine. If `sem_fit` is unavailable or the spec is outside its non-recursive
subset, the verdict falls back to `semopy`-only and says so.

## Output location (context-aware)

The skill detects its environment automatically — **no extra configuration
is needed**. It checks whether the current directory (or the one given by
`--root`) belongs to the Chelate Shell ecosystem by looking for the two
load-bearing zones of the four-zone layout: `material/` **and** `process/`.

- **Chelate mode** — `--root` points at a directory that contains **both
  `material/` and `process/` zones** (the Chelate four-zone layout). Outputs
  land under `<root>/process/{gates,figures,latex}/`, exactly as in the full
  Chelate Shell repo.
- **Standalone mode** — no `--root`, or `--root` lacks the `material/` +
  `process/` zone pair. A local
  `<cwd>/sem-copilot-output/{gates,figures,latex}/` directory is created and
  every JSON, PDF, and `.tex` file is written there. No Chelate install
  required.

## When to use
- You have a labelled CSV and a latent-variable model and want a verdict, a
  figure, and a manuscript-ready `.tex` — in one tool, no commercial software.
- An AMOS / lavaan output must be re-derived and audited with an open-source,
  cross-checked engine.

## When NOT to use
- No latent-variable structure present (use a tabular auditor for raw
  regression).
- A near-singular covariance matrix (the sandbox times out; re-specify the
  model first).

## Dependencies
- `semopy` — SEM fitting (open-source, replaces AMOS)
- `pandas`, `numpy` — data + matrix math
- `matplotlib`, `seaborn` — path-diagram rendering (vector PDF)
- `scipy` (optional) — the independent `sem_fit` cross-check
- Pure-stdlib LaTeX export (no pandoc)

A missing heavy dependency degrades the verdict to **UNVERIFIED** with a
precise reason — the pipeline never breaks.
