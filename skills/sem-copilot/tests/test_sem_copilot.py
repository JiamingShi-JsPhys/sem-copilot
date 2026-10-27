"""Standalone-repo test suite for the consolidated ``sem-copilot`` skill.

Covers the three guarantees the user asked for, plus the basics:

* **Fit / render / export independently work** (the skill is self-contained:
  no reference to the old multi-skill main repo).
* **Standalone mode** writes to ``<cwd>/sem-copilot-output/``.
* **Chelate mode** writes to ``<root>/process/`` when BOTH ``material/``
  and ``process/`` zones exist.
* **The cross-check warning mechanism** triggers when ``semopy`` and the
  independent ``sem_fit`` engine disagree (semopy-PASS + sem_fit-misfit =>
  the gate verdict carries a WARNING string but the primary PASS is kept).

Run from the repo root:

    python -m pytest skills/sem-copilot/tests -v
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# --- locate the consolidated skill -----------------------------------------
SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_DIR / "scripts"
FIXTURE = SKILL_DIR / "tests" / "fixtures" / "mock_survey.csv"

MODEL = (
    "Stress =~ Q1_1 + Q1_2 + Q1_3\n"
    "Burnout =~ Q2_1 + Q2_2 + Q2_3\n"
    "Burnout ~ Stress + Age + Tenure"
)


def _load(path: Path, name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sem_engine():
    return _load(SCRIPTS / "engine.py", "sc_sem_engine")


@pytest.fixture(scope="module")
def plotter():
    return _load(SCRIPTS / "plotter.py", "sc_plotter")


@pytest.fixture(scope="module")
def exporter():
    return _load(SCRIPTS / "exporter.py", "sc_exporter")


@pytest.fixture(scope="module")
def run_module():
    return _load(SCRIPTS / "run.py", "sc_run")


# ---------------------------------------------------------------------------
# 0. sanity
# ---------------------------------------------------------------------------
def test_fixture_exists():
    assert FIXTURE.is_file(), f"mock_survey.csv missing: {FIXTURE}"


# ---------------------------------------------------------------------------
# 1. independent fit / render / export
# ---------------------------------------------------------------------------
def test_engine_fit_passes_and_writes_evidence(sem_engine, tmp_path):
    """``engine.audit`` should PASS on the mock fixture and persist the
    evidence (parameter table + fit indices) under the root it's given."""
    root = tmp_path / "ws"
    (root / "process" / "gates").mkdir(parents=True)
    res = sem_engine.audit(MODEL, str(FIXTURE), root=str(root))
    assert res["status"] == "PASS", res
    assert res["artifacts"], "audit must emit evidence artifacts"
    # the two evidence files must actually exist under the root
    names = {Path(a).name for a in res["artifacts"]}
    assert "sem-inspection.json" in names
    assert "sem-fit-indices.json" in names
    idx = json.load(open(root / "process" / "gates" / "sem-fit-indices.json"))
    assert "CFI" in idx and "RMSEA" in idx


def test_engine_path_spec_drawable(sem_engine):
    spec = sem_engine.path_spec(MODEL, str(FIXTURE), title="mock")
    assert "error" not in spec, spec
    lats = {l["name"] for l in spec["latents"]}
    assert lats == {"Stress", "Burnout"}
    exo = {e["name"] for e in spec["exogenous"]}
    assert exo == {"Age", "Tenure"}
    paths = {(p["from"], p["to"]) for p in spec["structural_paths"]}
    assert ("Stress", "Burnout") in paths


def test_plotter_renders_sem_path(plotter, tmp_path):
    spec = _load_pathspec_json(tmp_path)
    res = plotter.generate(str(spec), output_dir=str(tmp_path))
    assert res["status"] == "PASS", res
    kinds = [a for a in res["artifacts"]]
    assert any(a.endswith(".pdf") for a in kinds)
    assert any(a.endswith(".png") for a in kinds)


def _load_pathspec_json(tmp_path):
    """Return a path to a minimal-but-drawable sem_path JSON."""
    spec = {
        "type": "sem_path",
        "title": "unit",
        "latents": [
            {"name": "F", "label": "F",
             "indicators": ["a", "b", "c"],
             "loadings": [1.0, 0.8, 0.7]},
        ],
        "exogenous": [{"name": "Age", "label": "Age"}],
        "structural_paths": [
            {"from": "Age", "to": "F", "estimate": 0.4, "p": 0.001},
        ],
        "fit_indices": {"CFI": 0.98, "TLI": 0.97, "RMSEA": 0.04},
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def test_exporter_merges(tmp_path):
    """Pure-stdlib LaTeX export must produce a compilable .tex even with
    no inputs (a compilable skeleton)."""
    root = tmp_path / "ws"
    (root / "process").mkdir(parents=True)
    exporter = _load(SCRIPTS / "exporter.py", "sc_exporter_2")
    res = exporter.export(root=str(root), out_name="x.tex")
    assert res["status"] == "PASS", res
    tex = root / "process" / "latex" / "x.tex"
    assert tex.is_file()
    body = tex.read_text(encoding="utf-8")
    assert body.startswith(r"\documentclass")
    assert r"\end{document}" in body


# ---------------------------------------------------------------------------
# 2. standalone vs. chelate output-location
# ---------------------------------------------------------------------------
def test_standalone_mode_writes_sem_copilot_output(
        run_module, tmp_path, monkeypatch):
    """When no --root is passed, run.py must write to
    <cwd>/sem-copilot-output/ rather than any --root location."""
    monkeypatch.chdir(tmp_path)
    # simulate: no --root at all
    base, mode = run_module._output_base(None)
    assert mode == "standalone"
    assert base == str(tmp_path / "sem-copilot-output"), base
    # the dir must have been created on demand
    assert Path(base).is_dir()


def test_chelate_mode_uses_root_when_zones_present(
        run_module, tmp_path):
    """When --root points at a Chelate 4-zone workspace (has BOTH material/
    and process/ zones), output must land under that root, NOT under
    ./sem-copilot-output/."""
    chelate_root = tmp_path / "chelate"
    (chelate_root / "material").mkdir(parents=True)
    (chelate_root / "process").mkdir(parents=True)
    base, mode = run_module._output_base(str(chelate_root))
    assert mode == "chelate"
    assert base == str(chelate_root)


def test_plain_dir_without_process_falls_back_to_standalone(
        run_module, tmp_path):
    """A plain directory that lacks the process/ zone is treated as
    standalone even when it was passed as --root."""
    plain = tmp_path / "plain"
    plain.mkdir()
    base, mode = run_module._output_base(str(plain))
    assert mode == "standalone"


def test_is_chelate_root_predicate(run_module, tmp_path, monkeypatch):
    # --root omitted falls back to cwd; from a clean (non-Chelate) cwd it
    # must be False.
    monkeypatch.chdir(tmp_path)
    assert run_module._is_chelate_root(None) is False
    # a plain directory with neither zone is standalone
    plain = tmp_path / "plain"
    plain.mkdir()
    assert run_module._is_chelate_root(str(plain)) is False
    # material/ alone is not enough (must be the material/ + process/ pair)
    mat_only = tmp_path / "mat"
    (mat_only / "material").mkdir(parents=True)
    assert run_module._is_chelate_root(str(mat_only)) is False
    # both material/ and process/ => Chelate workspace
    chelate = tmp_path / "chelate"
    (chelate / "material").mkdir(parents=True)
    (chelate / "process").mkdir(parents=True)
    assert run_module._is_chelate_root(str(chelate)) is True


# ---------------------------------------------------------------------------
# 3. cross-check WARNING mechanism
# ---------------------------------------------------------------------------
def test_cross_check_warning_fires_on_divergence(
        sem_engine, monkeypatch, tmp_path):
    """Force ``sem_fit.fit_indices`` to return a deliberately-bad result
    so that semopy-PASS + sem_fit-misfit -> the gate still returns
    PASS but the reason string carries the WARNING text the user
    wants to see.  This is the exact mechanism observed in the manual
    run of Phase 3: semopy CFI=0.986 while the independent numpy
    engine reported CFI=0.556."""
    original = sem_engine._sem_fit
    if original is None:
        pytest.skip("sem_fit cross-check engine unavailable in this "
                    "interpreter; nothing to test")
    fake = {
        "recursive": False,
        "cfi": 0.556,
        "tli": 0.55,
        "rmsea": 0.203,
        "chi2": 180.0,
        "df": 10,
    }
    monkeypatch.setattr(original, "fit_indices",
                        lambda spec, data: fake)

    root = tmp_path / "ws"
    (root / "process" / "gates").mkdir(parents=True)
    res = sem_engine.audit(MODEL, str(FIXTURE), root=str(root))
    # primary verdict is semopy-based (PASS because semopy says CFI >= 0.90)
    assert res["status"] == "PASS", res
    # but the reason string MUST carry the cross-check warning
    reason = res["reason"].lower()
    assert "inconsistent" in reason or "warning" in reason, \
        f"cross-check warning missing from reason: {res['reason']}"
    # and it must mention the divergence on the two engines
    assert "sem_fit" in reason or "independent" in reason


def test_cross_check_pass_agrees(
        sem_engine, monkeypatch, tmp_path):
    """When the two engines agree, the reason says 'cross-verified'."""
    original = sem_engine._sem_fit
    if original is None:
        pytest.skip("sem_fit unavailable")
    # a 'agree' answer: return something close to the semopy result
    fake = {"recursive": False, "cfi": 0.98, "tli": 0.97, "rmsea": 0.04,
            "chi2": 12.0, "df": 10}
    monkeypatch.setattr(original, "fit_indices", lambda s, d: fake)
    root = tmp_path / "ws"
    (root / "process" / "gates").mkdir(parents=True)
    res = sem_engine.audit(MODEL, str(FIXTURE), root=str(root))
    assert res["status"] == "PASS"
    reason = res["reason"].lower()
    assert "cross-verified" in reason, \
        f"expected 'cross-verified' wording, got: {res['reason']}"


def test_sem_fit_unavailable_degrades_gracefully(
        sem_engine, monkeypatch, tmp_path):
    """If the independent engine is missing entirely, the gate degrades
    to UNVERIFIED-style note on the reason string and still returns
    PASS if semopy is happy -- never crashes."""
    monkeypatch.setattr(sem_engine, "_sem_fit", None)
    root = tmp_path / "ws"
    (root / "process" / "gates").mkdir(parents=True)
    res = sem_engine.audit(MODEL, str(FIXTURE), root=str(root))
    # semopy-only verdict; no crash, PASS still allowed
    assert res["status"] in ("PASS", "UNVERIFIED"), res
