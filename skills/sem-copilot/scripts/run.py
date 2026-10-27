"""sem-copilot -- unified CLI over the single consolidated skill.

This is the one entry point for the whole SEM loop:

    fit model -> draw path diagram -> export LaTeX

The three capabilities live in this skill's scripts/ directory:
  * engine.py   -- SEM compute + gate (semopy fit + numpy cross-check),
                   AMOS-style extensions (MLR / WLSMV / mediation /
                   invariance / MI / bootstrap)
  * plotter.py  -- publication-quality SEM path diagram (vector PDF + PNG)
  * exporter.py -- merge Markdown reports + figures into a compilable .tex

Usage:
    python scripts/run.py --action gate    --root <WS> --input spec.json
    python scripts/run.py --action gate    --root <WS> --model "y ~ x1+x2" --data data.csv
    python scripts/run.py --action render  --root <WS> --input spec.json
    python scripts/run.py --action export  --root <WS> [--name report.tex]

Context-aware output location:
  * If ``--root`` points at a Chelate 4-zone workspace (a directory that
    contains both ``material/`` and ``process/`` zones), all outputs land
    under ``<root>/process/`` (gates/, figures/, latex/), as in the full
    Chelate Shell repo.
  * Otherwise (standalone use, e.g. no --root, or --root without the
    material/ + process/ zone pair), outputs go to
    ``<cwd>/sem-copilot-output/`` so the skill works with no Chelate
    install at all. When ``--root`` is omitted, the current working
    directory is probed for the Chelate zone pair instead.

Graceful degradation: a missing heavy dependency (semopy / matplotlib /
pandas / numpy) degrades the verdict to UNVERIFIED instead of crashing.
"""
from __future__ import annotations

import json
import os
import sys

# Make both the repository root (for `lib.chelate_core` + `contracts`) and
# this skill's own scripts/ dir (for `import engine/plotter/exporter/sem_fit/
# sem_extended`) importable regardless of the caller's cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.chelate_core import adapter  # noqa: E402

SKILL = "sem-copilot"

_SPEC_HINT = ('expected a JSON spec of shape '
              '{"model": "<lavaan string>", "data": "<csv path>"}; '
              'or pass --model + --data directly')


# ---------------------------------------------------------------------------
# context-aware output location (Chelate 4-zone vs. standalone)
# ---------------------------------------------------------------------------

def _is_chelate_root(root) -> bool:
    """True when ``root`` looks like a Chelate Shell workspace.

    The Chelate layout is a four-zone directory (inbox/material +
    process/ + results/ + _broadcast/).  The canonical, load-bearing zones
    are ``material/`` (the curated-source inbox) and ``process/`` (where
    every gate/figure/latex artifact is written).  We treat a root as
    Chelate only when it exists and carries BOTH ``material/`` and
    ``process/`` subdirectories -- a plain user directory with only one of
    them is standalone.

    When ``root`` is omitted (None), the current working directory is
    probed instead, so a bare ``python run.py`` executed inside a Chelate
    workspace still auto-detects Chelate mode.
    """
    if not root:
        root = os.getcwd()
    if not os.path.isdir(root):
        return False
    return (os.path.isdir(os.path.join(root, "material"))
            and os.path.isdir(os.path.join(root, "process")))


def _output_base(root) -> tuple:
    """Return ``(base, mode)`` where base is the directory all outputs land
    under, and mode is ``"chelate"`` or ``"standalone"``.

    Chelate: base = root (existing 4-zone process/ tree).
    Standalone: base = <cwd>/sem-copilot-output (created on demand).
    """
    if _is_chelate_root(root):
        return root, "chelate"
    base = os.path.join(os.getcwd(), "sem-copilot-output")
    os.makedirs(base, exist_ok=True)
    return base, "standalone"


def _resolve(data_path, root):
    """Absolute path for ``data_path``.

    A relative path is resolved against ``--root`` first, then against the
    current working directory; an absolute path is used as-is.  This lets
    the caller run from inside the workspace or from anywhere else.
    """
    if not data_path:
        return ""
    if os.path.isabs(data_path):
        return data_path
    if root:
        candidate = os.path.join(root, data_path)
        if os.path.isfile(candidate):
            return candidate
    return data_path


def _load_spec(args):
    """Resolve (model_spec, data_path, error) from --input JSON or
    --model/--data.  ``error`` is a human-readable string when the spec
    could not be parsed (so the caller emits a self-explanatory verdict).
    """
    if args.input:
        try:
            with open(args.input, encoding="utf-8") as fh:
                spec = json.load(fh)
        except json.JSONDecodeError as exc:
            return "", "", (f"invalid JSON in --input {args.input!r} "
                            f"({exc}); " + _SPEC_HINT)
        if isinstance(spec, dict):
            return spec.get("model") or "", spec.get("data") or "", None
        return "", "", (f"--input {args.input!r} is not a JSON object; "
                        + _SPEC_HINT)
    return args.model or "", args.data or "", None


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------

def _gate_handler(args):
    """Fit + gate the SEM (audit).  Output lands in the detected base's
    process/gates/ tree."""
    model_spec, data_path, err = _load_spec(args)
    if err:
        return {"status": "UNVERIFIED", "evidence_level": "UNVERIFIED",
                "reason": err, "artifacts": []}
    data_path = _resolve(data_path, args.root)
    if not model_spec or not data_path:
        return {"status": "UNVERIFIED", "evidence_level": "UNVERIFIED",
                "reason": "missing model specification or dataset path "
                          + _SPEC_HINT, "artifacts": []}
    base, mode = _output_base(args.root)
    from engine import audit
    verdict = audit(model_spec, data_path, root=base)
    # Merge the engine's structured payload (n_obs / mediation /
    # n_size_warning) with the CLI-level output-location fields so the
    # gate JSON carries both.
    verdict["data"] = {**(verdict.get("data") or {}),
                       "mode": mode, "output_base": base}
    return verdict


def _render_handler(args):
    """Closed-loop: fit the model, emit a ``sem_path`` figure spec, and
    immediately draw the vector PDF + PNG path diagram -- one command
    completes fit -> draw.  Output lands in ``<base>/process/figures/``
    (both chelate and standalone modes) so the ``export`` action can pick
    it up later in the same workspace."""
    model_spec, data_path, err = _load_spec(args)
    if err:
        return {"status": "UNVERIFIED", "evidence_level": "UNVERIFIED",
                "reason": err, "artifacts": []}
    data_path = _resolve(data_path, args.root)
    if not model_spec or not data_path:
        return {"status": "UNVERIFIED", "evidence_level": "UNVERIFIED",
                "reason": "missing model specification or dataset path "
                          + _SPEC_HINT, "artifacts": []}
    base, mode = _output_base(args.root)
    fig_dir = os.path.join(base, "process", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    from engine import path_spec
    spec = path_spec(model_spec, data_path, root=base, title=args.title)
    if "error" in spec:
        return {"status": "UNVERIFIED", "evidence_level": "UNVERIFIED",
                "reason": spec["error"], "artifacts": []}
    json_path = os.path.join(fig_dir, "sem_path_input.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    artifacts = [json_path]

    # Sample-size warning on the render path: a tiny N makes the SEM
    # estimates behind the diagram unreliable, so surface it prominently
    # even though the figure itself still draws.
    n_size_warning = None
    try:
        import pandas as _pd
        n_obs = int(len(_pd.read_csv(data_path)))
        if n_obs < 100:
            n_size_warning = (
                f"[WARNING] Sample size is too small (N={n_obs}). "
                f"Fit indices and p-values may be unreliable.")
    except Exception:
        n_obs = None

    # Draw the diagram from the spec we just wrote (fit -> draw in one step).
    from plotter import generate
    fig = generate(json_path, chart=None, x=None, y=None, hue=None,
                   title=args.title, output_dir=fig_dir, root=None)
    if fig["status"] == "PASS":
        artifacts += fig.get("artifacts", [])
        reason = f"sem_path spec + path diagram written to {fig_dir}"
    else:
        reason = (f"sem_path spec written to {json_path}; figure draw "
                  f"degraded ({fig['reason']})")
    data = {"mode": mode, "output_base": base}
    if n_size_warning:
        data["n_size_warning"] = n_size_warning
    if n_obs is not None:
        data["n_obs"] = n_obs
    return {"status": "PASS" if fig["status"] == "PASS" else "UNVERIFIED",
            "evidence_level": "EMPIRICAL",
            "reason": (n_size_warning + "; " if n_size_warning else "") + reason,
            "artifacts": artifacts, "data": data}


def _export_handler(args):
    """Merge Markdown reports + figures under the detected base into a
    single compilable .tex.  Output lands in the base's process/latex/
    tree (chelate) or latex/ (standalone)."""
    base, mode = _output_base(args.root)
    # exporter.export(root=...) reads <root>/process/ and writes
    # <root>/process/latex/ in Chelate mode. In standalone mode there is no
    # process/ under base, so we point it at the standalone process/ subtree
    # we created for consistency.
    export_root = base if mode == "chelate" else base
    from exporter import export
    verdict = export(root=export_root,
                     out_name=args.name or "merged_report.tex")
    verdict["data"] = dict(verdict.get("data") or {},
                           mode=mode, output_base=base)
    return verdict


def _configure(ap):
    ap.add_argument("--input",
                   help="JSON spec file: {\"model\": ..., \"data\": "
                        "<csv path>}")
    ap.add_argument("--model", help="lavaan-style model string")
    ap.add_argument("--data",
                   help="path to the dataset CSV (relative paths resolve "
                        "against --root)")
    ap.add_argument("--title",
                   help="optional title for the auto-generated SEM path "
                        "diagram")
    ap.add_argument("--name", default="merged_report.tex",
                   help="output .tex filename for the export action "
                        "(default: merged_report.tex)")


if __name__ == "__main__":
    # NOTE: adapter.dispatch() makes --root a hard requirement (it bails with
    # exit 2 when absent), which conflicts with standalone mode.  So we parse
    # args ourselves here while still routing the *action* through the
    # standard gate-emission path (adapter._emit) so a gate JSON is always
    # written and the single stdout JSON line contract is preserved.
    import argparse

    handlers = {"gate": _gate_handler, "render": _render_handler,
                "export": _export_handler, "process": _export_handler}

    ap = argparse.ArgumentParser(prog=SKILL,
                                 description="SEM Copilot: fit -> draw -> "
                                             "export LaTeX (one skill).")
    # NOTE: we deliberately do NOT use adapter.add_common_args(), which would
    # make --root required. Standalone mode omits --root entirely.
    ap.add_argument("--root", required=False, default=None,
                    help="Chelate workspace root (optional). When given and "
                         "it contains a process/ zone, outputs land in "
                         "<root>/process/; otherwise they go to "
                         "<cwd>/sem-copilot-output/.")
    ap.add_argument("--action", default="gate",
                    help="action: gate | render | export (default: gate); "
                         "'process' is an alias for export")
    ap.add_argument("--debug", action="store_true",
                   help="verbose diagnostics on stderr")
    _configure(ap)

    argv = sys.argv[1:]
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        # --help exits 0 from argparse; anything else is a usage error.
        raise

    action = args.action
    if action in ("process",) and action not in handlers:
        action = "export"
    if action not in handlers:
        print(f"{SKILL}: unknown action {action!r}; expected one of "
              f"{sorted(set(handlers) - {'process'})}", file=sys.stderr)
        sys.exit(2)

    base, mode = _output_base(args.root)
    handler = handlers[action]
    try:
        result = handler(args) or {}
        status = result.get("status", "UNVERIFIED")
        evidence = result.get("evidence_level", "UNVERIFIED")
        reason = result.get("reason", "")
        artifacts = result.get("artifacts") or []
        data = result.get("data") or {}
        # Emit the gate so a JSON artifact is always written and the stdout
        # contract (one JSON line) holds.  In standalone mode the gate file
        # lands under <cwd>/sem-copilot-output/process/gates/; in chelate
        # mode under <root>/process/gates/.
        gate_root = base if mode == "standalone" else args.root
        adapter._emit(gate_root, SKILL, status, evidence, reason,
                      artifacts=artifacts, data=data)
    except adapter.Degraded as exc:
        adapter._emit(args.root or base, SKILL, "UNVERIFIED",
                      exc.evidence_level, str(exc), artifacts=exc.artifacts)
    except adapter.Failed as exc:
        adapter._emit(args.root or base, SKILL, "FAIL",
                      exc.evidence_level, str(exc), artifacts=exc.artifacts)
    except Exception as exc:  # fail closed, never pass a gate on a crash
        import traceback
        tail = " | ".join(traceback.format_exc().splitlines()[-3:])
        adapter._emit(args.root or base, SKILL, "FAIL", "UNVERIFIED",
                      f"adapter error: {type(exc).__name__}: {exc} [{tail}]")
    sys.exit(0)
