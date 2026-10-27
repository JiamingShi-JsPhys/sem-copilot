# SEM Copilot

> **A lightweight, transparent, and auditable Structural Equation Modeling (SEM) engine, acting as a sub-module of the Chelate Shell ecosystem.**
> Designed for social science researchers and anyone requiring reproducible, highly transparent modeling workflows.

**English** · [简体中文](README_zh.md)

**Version:** 1.0.0 | **Standalone** Sub-module | **Architecture:** Standalone / Adaptive Main-repo Compatibility

---

## Safety and Open Source Disclaimer

> **Disclaimer**: This repository contains auxiliary tools developed by the author for automated research workflows and is not a peer-reviewed academic project. Some boilerplate code may be AI-assisted, reviewed, and integrated at a high level by the author. The author assumes no responsibility for third-party code issues but is committed to promptly addressing any reasonable requests.
> 
> Generated reports and figures are "data analysis drafts" only; ultimate academic responsibility lies with the user and their affiliated institution.

---

## Why SEM Copilot?

Structural Equation Modeling (SEM) analysis in social science research often faces the following engineering and transparency pain points:

- **Closed-source black boxes are hard to verify.** Traditional graphical commercial software provides fit indices, but the underlying algorithms are closed. Reviewers can hardly independently reproduce and verify them from raw data.
- **Manual modeling is hard to automate.** Dragging and dropping, drawing lines in a GUI is time-consuming and error-prone, and cannot be seamlessly integrated into modern automated data pipelines.
- **Cross-platform and collaboration limitations.** Some software relies on specific operating systems or expensive commercial licenses, restricting team collaboration and open-source sharing.

### Minimalist Experience
1. Describe your model using natural language or standard `lavaan` syntax.
2. SEM Copilot automatically fits the model and outputs detailed statistical metrics.
3. Directly export vector PDF path diagrams ready for top-tier journal submissions.

Entirely code-based, no mouse required, completely free, and cross-platform.

---

## Technical Credibility: Dual-Engine Cross-Validation

Results from a single estimator are often hard to verify. SEM Copilot runs two independent engines simultaneously and cross-validates them:

- **`semopy` engine**: A mature open-source SEM estimator supporting mediation effects, multi-group invariance, categorical WLSMV, robust ML standard errors, modification indices, and bootstrap confidence intervals.
- **Pure Numpy Independent Re-derivation**: A from-scratch implementation that does not depend on `semopy`, directly recalculating the likelihood and fit indices (CFI / TLI / RMSEA) for continuous SEM from the data covariance matrix.

Both engines are independently implemented, allowing reviewers to independently re-derive your fit indices from raw data. This guarantees computational transparency and auditability at the code level.

---

## Core Architecture

This repository is a fully self-contained standalone plugin. It can be used as an independent tool or seamlessly integrated into the main repository's ecosystem. Core logic is strictly consolidated into a **single skill** to ensure no redundant files are scattered.

```text
sem-copilot/
│
├── Base Infrastructure
│   ├── core/             # Standalone Microkernel (Router, Sandbox)
│   └── contracts/        # Contracts (gate_result)
│
├── Shared Utils
│   └── lib/chelate_core/ # Base adapters and path handling
│
└── Workflow Processor
    └── skills/sem-copilot/ # The single, all-in-one skill
        ├── SKILL.md
        └── scripts/      # engine (fitting) / plotter (drawing) / exporter (typesetting)
```

**Context-Aware Detection**: On every run, the tool automatically detects whether the current directory (or the directory specified by `--root`) belongs to the sem copilot ecosystem structure—i.e., whether it contains both `material/` and `process/` directories.
- **Ecosystem Compatibility Mode**: If true, all outputs are written to `<root>/process/{gates,figures,latex}/`.
- **Standalone Mode**: If false, it safely falls back to `./sem-copilot-output/` in the current directory, ready to use out-of-the-box with zero configuration.

---

## Installation & Download

This repository is a cross-platform Agent skill suite supporting Claude Code, WorkBuddy, OpenAI Codex, and DeepSeek Harness (DSH). Please choose the installation method that best suits you:

### Option 1: Install via Git URL (Recommended, Agent-native method)

1. Open the plugin management (or custom marketplace configuration) interface on your platform.
2. Select "Install from Git repository" or "Add custom marketplace".
3. Enter the repository URL and confirm: `https://github.com/JiamingShi-JsPhys/sem-copilot`

### Option 2: let your AI install it

Copy the following prompt to your AI agent. **Install it as a plugin, not as loose skills scattered into the skill directory:**

> "Please visit the GitHub repository `https://github.com/JiamingShi-JsPhys/sem-copilot`. Install it **as a single plugin** (do NOT scatter the individual skills under `skills/` into your global skill directory, and do NOT flatten the repository layout). Follow your agent platform's plugin installation conventions, and only fall back to skill registration if your platform has no plugin concept."

> **Warning**: when an AI performs the installation it may scatter the individual skills from `skills/` flat into the platform's global skill directory. That breaks sem copilot's microkernel auto-routing architecture and degrades it into loose skills. Use Option 1 if you need the full plugin orchestration.

**Claude Code (slash command)** — install it in one line, directly in the chat (Claude Code v2.1.275+):

```
/plugin install sem-copilot --marketplace JiamingShi-JsPhys/sem-copilot
```

This fetches the Git repository and registers sem copilot as a plugin in a single step — the `--marketplace` flag names the source so Claude Code adds it on the fly, no manual clone. On older Claude Code versions without `--marketplace`, run the two-step form instead:

```
/plugin marketplace add JiamingShi-JsPhys/sem-copilot
/plugin install sem-copilot@sem-copilot-marketplace
```

### Option 3: clone the repository (developers / source inspection)

If you are a developer, or simply want to read the underlying code, clone this repository in a terminal:

```bash
git clone https://github.com/JiamingShi-JsPhys/sem-copilot.git
```
---

## Quick Start

All commands are executed in the plugin root directory. Never hardcode absolute paths.

```bash
# 1. Model Fitting & Gate Check
python skills/sem-copilot/scripts/run.py \
    --root <workspace> --action gate \
    --model "f1 =~ x1 + x2 + x3
f2 =~ y1 + y2 + y3
f2 ~ f1" \
    --data dataset.csv

# 2. Path Diagram Rendering
python skills/sem-copilot/scripts/run.py \
    --root <workspace> --action render \
    --model "..." --data dataset.csv

# 3. LaTeX Report Export
python skills/sem-copilot/scripts/run.py \
    --root <workspace> --action export
```

---

## Skill Dependency Matrix

| Skill | Required Dependencies | Optional Dependencies (Degraded if Missing) |
| --- | --- | --- |
| `sem-copilot` | `semopy`, `pandas`, `numpy` | `matplotlib`, `seaborn` (plotting), `sem_fit` (cross-validation) |

All heavy dependencies run in isolated sandboxes. If dependencies are missing, the system gracefully degrades to `UNVERIFIED` with precise prompts, without crashing the entire pipeline.

---

## Known Limitations

Honest declaration of edge cases; please note the following before relying on these outputs:

- **Model fitting is not absolute proof**: The fit indices (CFI/RMSEA) output by the engines are based on ML estimation and specific optimization algorithms. Although dual-engine cross-validation greatly improves credibility, results still need to be interpreted in light of theoretical context and cannot completely replace human review of the model.
- **Dual-engine results are not byte-for-byte identical**: There are differences in the underlying optimizers between `semopy` and the pure Numpy re-derivation engine. Rare edge models (e.g., just-identified models) may not fully converge, in which case the system will flag `UNVERIFIED`.
- **Not a GUI replacement**: This tool does not provide a drag-and-drop modeling GUI similar to traditional commercial software. All model definitions must be described via code syntax or natural language assistance.

---

## Report Issues & Feedback

**Regular Bugs / Feature Requests**:
Please submit them on the repository's [Issues](https://github.com/JiamingShi-JsPhys/sem-copilot/issues) page. When submitting, please include: reproduction steps, expected behavior, actual behavior, and runtime environment (OS, Python version).

**Security Vulnerability Reports (This type only)**:
Please **DO NOT** submit security vulnerabilities on the public Issues page. Send an email to `jiamingshi_git@outlook.com`.
- **Subject**: Must include the prefix `report`, the specific skill name, and a brief description.
- **Body**: Must include detailed reproduction steps and runtime environment.

*Contributors who submit clear, reproducible issues or pull requests will be listed in the acknowledgments of the next release notes.*

---

## Open Source License & Author

This repository is released under the [MIT License](LICENSE), as detailed in the `LICENSE` file in the root directory.

**Author & Maintainer**: Jiaming Shi (Westlake) & Keqia Ying (ZUEL)

---

## Copyright & Disclaimer

SEM Copilot is an independent open-source project and is not affiliated with, endorsed by, or sponsored by Microsoft Corporation. "Copilot" is a trademark of Microsoft Corporation.
