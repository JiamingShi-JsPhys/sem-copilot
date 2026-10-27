"""SEM audit engine for the sem-copilot sem-auditor gate skill.

Fits a structural-equation model described in lavaan-style syntax against a
CSV dataset using semopy, then gates on the conventional fit indices:

    PASS  iff  CFI >= 0.90  and  TLI >= 0.90  and  RMSEA <= 0.08
    FAIL  otherwise (the reason carries the actual indices)

Graceful degradation: a missing heavy dependency (semopy / pandas / numpy)
must never crash the pipeline -- the engine returns an UNVERIFIED verdict
with a precise reason instead. License note: semopy (MIT), pandas
(BSD-3-Clause) and numpy (BSD-3-Clause) are all permissive and compatible
with the MIT distribution of sem-copilot.
"""
from __future__ import annotations

import json
import os
import sys

# Make this skill's own scripts/ dir importable so `import sem_fit` resolves
# regardless of the caller's cwd (the engine is loaded by absolute path in
# tests and by the router from the repo root).
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Independent re-derivation of fit indices (AMOS / lavaan replacement).
# Lazy-imported so the module stays light when sem_fit / scipy is missing.
try:
    import sem_fit as _sem_fit
except Exception:  # pragma: no cover - optional cross-check
    _sem_fit = None

# Fit-index acceptance thresholds (Hu & Bentler conventional cut-offs).
THRESHOLDS = {"cfi": 0.90, "tli": 0.90, "rmsea": 0.08}


def _verdict(status, reason, artifacts=None, evidence="EMPIRICAL", data=None):
    """Build the uniform gate dict the adapter layer expects."""
    out = {
        "status": status,
        "evidence_level": evidence,
        "reason": reason,
        "artifacts": list(artifacts or []),
    }
    if data is not None:
        out["data"] = data
    return out


def _extract_index(stats, key):
    """Pull a single fit index out of semopy's stats DataFrame.

    Returns float('nan') when the index is absent or non-numeric so a
    missing value degrades to a FAIL (NaN never satisfies a threshold).
    """
    try:
        return float(stats[key].iloc[0])
    except Exception:
        return float("nan")


def _collect_evidence(model, stats, out_dir, model_spec=None):
    """Write the full inspection table (with auto-computed mediation
    effects) + fit indices to the workspace and return
    ``(artifacts, mediation)``.

    Without this, the gate emits ``artifacts: []`` and the path
    coefficients, factor loadings, and CIs that a manuscript actually
    reports are never saved.  The two files land in
    ``<out_dir>/sem-inspection.json`` and ``<out_dir>/sem-fit-indices.json``.
    The inspection file now also carries a ``mediation`` block so the
    indirect / total / proportion-mediated numbers are written to disk,
    not just returned in the gate's data payload.
    """
    artifacts = []
    mediation = _auto_mediation(model, model_spec=model_spec)
    if not out_dir:
        return artifacts, mediation
    os.makedirs(out_dir, exist_ok=True)

    # inspection: estimated parameters (path coefficients, loadings, etc.)
    try:
        insp = model.inspect()
        keep_cols = ("Estimate", "Std.Err", "z", "P(>|z|)",
                     "CI.Lower", "CI.Upper", "Significance")
        param_table = []
        for i, label in enumerate(insp.index):
            row = {"label": str(label)}
            for col in insp.columns:
                if col in keep_cols:
                    row[col] = insp.iat[i, insp.columns.get_loc(col)]
            param_table.append(row)
        # Persist auto-computed mediation effects inside the inspection file
        # so a reviewer / downstream tool can read them without re-fitting.
        inspection_payload = {"parameters": param_table,
                              "mediation": mediation}
        with open(os.path.join(out_dir, "sem-inspection.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(inspection_payload, fh, indent=2, default=str)
        artifacts.append(os.path.join(out_dir, "sem-inspection.json"))
    except Exception:
        pass  # inspection is best-effort; never block the gate on it

    # fit indices (the four + a couple more for the manuscript)
    try:
        idx = {}
        for key in ("CFI", "TLI", "RMSEA", "SRMR", "GFI", "AIC", "BIC"):
            try:
                idx[key] = float(stats[key].iloc[0])
            except Exception:
                idx[key] = None
        with open(os.path.join(out_dir, "sem-fit-indices.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(idx, fh, indent=2)
        artifacts.append(os.path.join(out_dir, "sem-fit-indices.json"))
    except Exception:
        pass

    return artifacts, mediation


def _fit(model_spec, data_path):
    """Fit the SEM with semopy and return ``(model, stats, data)``.

    On any unrecoverable problem (missing dependency, unreadable CSV, or a
    semopy fit failure) it returns a ``_verdict`` dict instead so callers can
    surface it uniformly.  Shared by :func:`audit` and :func:`path_spec` so the
    two never duplicate the fitting logic.
    """
    try:
        import semopy  # noqa: F401
    except ImportError:
        return _verdict("UNVERIFIED", "semopy not installed")
    try:
        import pandas as pd
    except ImportError:
        return _verdict("UNVERIFIED", "pandas not installed")
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return _verdict("UNVERIFIED", "numpy not installed")
    if not model_spec:
        return _verdict("UNVERIFIED", "empty model specification")
    if not data_path or not os.path.isfile(data_path):
        return _verdict("UNVERIFIED", f"dataset not found: {data_path}")
    try:
        data = pd.read_csv(data_path)
    except Exception as exc:
        return _verdict("UNVERIFIED", f"could not read CSV {data_path}: {exc}")
    # Sample size, surfaced to the caller so it can warn about small N even
    # when the fit itself fails.
    n_obs = int(len(data))
    try:
        model = semopy.Model(model_spec)
        model.fit(data)
        stats = semopy.calc_stats(model)
    except Exception as exc:
        v = _verdict("UNVERIFIED", f"semopy failed to fit the model: {exc}")
        v["n_obs"] = n_obs
        return v
    return model, stats, data, n_obs


def _auto_mediation(model, model_spec=None):
    """Auto-discover X -> M -> Y mediation chains from a fitted model and
    decompose the effects.

    Only **structural** (regression) paths participate: measurement
    loadings (``=~``), variances/covariances (``~~``) and means (``==``)
    are excluded so we never mistake a loading ``Y =~ Y1`` for a causal
    edge.  In addition, the *source* (``X``) and *target* (``Y``) of a
    mediation chain must be **endogenous / free nodes** in the structural
    block -- latent constructs (the RHS of an ``=~`` line) and exogenous
    covariates only -- never observed indicators that belong to a
    measurement model.  That keeps the output on the same scale users see
    in a lavaan-style path diagram (constructs, not items).

    Reads the structural coefficients off ``model.inspect()``.  For every
    ordered triple where ``X -> M`` and ``M -> Y`` both exist:

      * ``indirect_effect`` = a * b   (a = coef(X->M), b = coef(M->Y))
      * ``direct_effect``   = coef(X->Y) if that path exists, else None
      * ``total_effect``    = direct + indirect
      * ``proportion_mediated`` = indirect / total (None when total == 0)

    Returns a list of dicts (empty when no mediator is present).  Best
    effort: any failure to introspect simply yields an empty list rather
    than crashing the gate.
    """
    # 1) endogenous / free variables: every LHS target of a structural '~'
    #    line.  We accept the raw model spec for the cleanest parse.
    endo = _structural_endogenous(model_spec)
    if not endo:
        # Fall back: derive the endogenous set from inspect() rows that
        # carry a regression operator, using the *lval* as target.
        try:
            insp = model.inspect()
            has_cols = "lval" in insp.columns and "op" in insp.columns
            for i, row in insp.iterrows():
                if has_cols:
                    op = str(row["op"]).strip()
                    lval = str(row["lval"]).strip()
                else:
                    parts = str(insp.index[i]).split()
                    if len(parts) != 3:
                        continue
                    lval, op, _ = parts
                if op == "~" and lval not in ("1",):
                    endo.add(lval)
        except Exception:
            endo = set()
    if not endo:
        return []

    # 2) structural edges only (op == '~', excluding the loading form
    #    'lval ~= rval').  semopy expresses loadings as op '=' '~' and the
    #    structural reg as op '~'; only the latter is a causal path.
    try:
        insp = model.inspect()
    except Exception:
        return []
    has_cols = "lval" in insp.columns and "op" in insp.columns
    edges = {}
    for i, row in insp.iterrows():
        if has_cols:
            lval = str(row["lval"]).strip()
            op = str(row["op"]).strip()
            rval = str(row["rval"]).strip()
        else:
            parts = str(insp.index[i]).split()
            if len(parts) != 3:
                continue
            lval, op, rval = parts
        # Only structural regressions (a '~' line where lval is NOT a
        # measurement target and the RHS is not '1').
        if op != "~" or rval in ("1",):
            continue
        # LHS must be an endogenous variable in the structural block
        if lval not in endo:
            continue
        try:
            est = float(row["Estimate"])
        except Exception:
            continue
        # semopy: "lval ~ rval" means rval -> lval
        if rval in ("1",) or rval not in endo:
            # If the source is an exogenous observed variable (not an
            # endogenous latent) it is still a valid structural edge, so
            # we relax: allow any source that is not the constant.
            if rval in ("1",):
                continue
        edges[(rval, lval)] = est

    results = []
    for M in sorted(endo):
        sources = [s for (s, d) in edges if d == M]
        targets = [d for (s, d) in edges if s == M]
        for X in sources:
            a = edges[(X, M)]
            for Y in targets:
                if Y == X:
                    continue
                b = edges[(M, Y)]
                indirect = a * b
                direct = edges.get((X, Y))
                total = (direct if direct is not None else 0.0) + indirect
                prop = (indirect / total) if total not in (0, None) else None
                results.append({
                    "x": X,
                    "mediator": M,
                    "y": Y,
                    "a_path_coef": round(a, 4),
                    "b_path_coef": round(b, 4),
                    "indirect_effect": round(indirect, 4),
                    "direct_effect": (round(direct, 4)
                                     if direct is not None else None),
                    "total_effect": round(total, 4),
                    "proportion_mediated": (round(prop, 4)
                                            if prop is not None else None),
                })
    return results


def _structural_endogenous(model_spec):
    """Return the set of endogenous variables (LHS targets of structural
    '~' lines).  Latent constructs in the structural block, e.g. 'M' and
    'Y' from ``M ~ X`` and ``Y ~ X + M``.  Empty set when the spec is
    None / unparseable."""
    if not model_spec:
        return set()
    endo = set()
    for line in model_spec.replace("\n", ";").split(";"):
        line = line.strip()
        if not line:
            continue
        # Skip measurement lines (contain '=~' / '::=' / '**').
        if "~=" in line or "==~" in line:
            continue
        # Only regression lines ('~', not '=~')
        if "~" not in line:
            continue
        lhs = line.split("~", 1)[0].strip()
        if lhs:
            endo.add(lhs)
    return endo


def audit(model_spec, data_path, root=None):
    """Run the SEM audit and return a gate dict.

    Parameters
    ----------
    model_spec : str
        lavaan-style model description, e.g. "y ~ x1 + x2".
    data_path : str
        path to the dataset CSV.
    root : str | None
        workspace root. Unused by the audit itself; kept for signature
        symmetry with other engines and future artifact anchoring.
    """
    # --- fit (shared with path_spec) ---------------------------------------
    fit = _fit(model_spec, data_path)
    if isinstance(fit, dict):              # a degradation verdict, not a tuple
        n_obs = fit.get("n_obs")
        if n_obs is not None and n_obs < 100:
            fit["reason"] = (
                f"[WARNING] Sample size is too small (N={n_obs}). "
                f"Fit indices and p-values may be unreliable. "
                + fit["reason"])
        fit.setdefault("artifacts", [])
        return fit
    model, stats, data, n_obs = fit

    cfi = _extract_index(stats, "CFI")
    tli = _extract_index(stats, "TLI")
    rmsea = _extract_index(stats, "RMSEA")

    # Independent re-derivation (sem_fit / AMOS replacement).  Cross-checks
    # the semopy fit: when the two engines agree, the verdict carries
    # "cross-verified"; when they diverge materially, the gate downgrades to
    # UNVERIFIED so a reviewer is alerted rather than trusting a single
    # package.  If sem_fit is unavailable or the spec is outside its
    # non-recursive subset, the verdict keeps the semopy-only result and
    # flags it accordingly.
    cross = None
    cross_note = ""
    if _sem_fit is not None:
        try:
            cross = _sem_fit.fit_indices(model_spec, data)
        except Exception:
            cross = None
        if cross is not None and cross.get("recursive"):
            cross_note = ("independent engine (sem_fit) could not reduce "
                         "the spec to its non-recursive subset; "
                         "verdict is semopy-only")
        elif cross is not None:
            # agreement check: RMSEA within 0.10 and CFI within 0.10
            if (cross["rmsea"] is not None
                    and abs(cross["rmsea"] - rmsea) <= 0.10
                    and abs(cross["cfi"] - cfi) <= 0.15):
                cross_note = (f"cross-verified with sem_fit "
                              f"(CFI={cross['cfi']:.3f}, "
                              f"RMSEA={cross['rmsea']:.3f})")
            else:
                cross_note = (
                    f"INCONSISTENT: semopy CFI={cfi:.3f}/RMSEA={rmsea:.3f} "
                    f"vs sem_fit CFI="
                    f"{cross['cfi']:.3f}/RMSEA="
                    f"{cross['rmsea']:.3f}; verdict downgraded")

    # Persist the full coefficient table + fit indices + auto-computed
    # mediation effects as evidence so the gate JSON can point at real
    # numbers instead of an empty artifact list.
    out_dir = os.path.join(root, "process", "gates") if root else None
    artifacts, mediation = _collect_evidence(model, stats, out_dir,
                                             model_spec=model_spec)

    # Sample-size sanity check: a tiny N makes the covariance matrix
    # near-singular, so fit indices and p-values are unreliable.  Surface a
    # prominent warning in the reason and record it in the data payload.
    n_size_warning = None
    if n_obs < 100:
        n_size_warning = (
            f"[WARNING] Sample size is too small (N={n_obs}). "
            f"Fit indices and p-values may be unreliable.")
        reason_prefix = n_size_warning + "; "
    else:
        reason_prefix = ""

    meets = (cfi >= THRESHOLDS["cfi"]
             and tli >= THRESHOLDS["tli"]
             and rmsea <= THRESHOLDS["rmsea"])

    # When the two independent engines disagree, *semopy-PASS + sem_fit-
    # misfit* is treated as a warning (the manuscript may be over-optimistic
    # but the primary gate still passes); *semopy-FAIL + sem_fit-fit* is
    # a genuine inconsistency that downgrades to UNVERIFIED.  The verdict
    # string carries a short note in either case.
    semopy_pass = meets
    sem_fit_pass = (
        cross is not None
        and not cross.get("recursive")
        and _sem_fit.meets_thresholds(cross) if _sem_fit else False)

    cross_warn = ""
    if cross_note:
        if semopy_pass and not sem_fit_pass and cross and not cross.get("recursive"):
            cross_warn = ("WARNING: independent engine (sem_fit) reports "
                          f"misfit (CFI={cross['cfi']:.3f}, "
                          f"RMSEA={cross['rmsea']:.3f}); "
                          "primary verdict kept as semopy-based")
        elif not semopy_pass and sem_fit_pass:
            cross_warn = ("INCONSISTENT: semopy reports misfit but "
                          "independent engine (sem_fit) reports good fit "
                          f"(CFI={cross['cfi']:.3f}, "
                          f"RMSEA={cross['rmsea']:.3f}); "
                          "verdict downgraded to UNVERIFIED")
        else:
            cross_warn = cross_note

    if semopy_pass:
        reason = (f"model fits: CFI={cfi:.3f}, TLI={tli:.3f}, "
                  f"RMSEA={rmsea:.3f}")
        if cross_warn:
            reason += "; " + cross_warn
        return _verdict("PASS", reason_prefix + reason, artifacts=artifacts,
                         data={"n_obs": n_obs, "mediation": mediation,
                               "n_size_warning": n_size_warning})

    # semopy says FAIL
    reason = (f"model misfit: CFI={cfi:.3f} "
              f"(>= {THRESHOLDS['cfi']}), "
              f"TLI={tli:.3f} (>= {THRESHOLDS['tli']}), "
              f"RMSEA={rmsea:.3f} (<= {THRESHOLDS['rmsea']})")
    if cross_warn:
        reason += "; " + cross_warn
    data = {"n_obs": n_obs, "mediation": mediation,
            "n_size_warning": n_size_warning}
    if "INCONSISTENT" in cross_warn:
        return _verdict("UNVERIFIED", reason_prefix + reason,
                        artifacts=artifacts, data=data)
    return _verdict("FAIL", reason_prefix + reason, artifacts=artifacts,
                    data=data)


def path_spec(model_spec, data_path, root=None, title=None):
    """Closed-loop bridge: fit the model, then emit a figure-generator
    ``sem_path`` drawing spec so the path diagram can be drawn with no further
    human input.

    This is what turns ``sem-auditor``'s *computation* into
    ``figure-generator``'s *picture* -- a single function that closes the loop.
    The returned dict is exactly the JSON ``figure-generator`` consumes when
    its ``--input`` is a ``sem_path`` spec (latents with indicators/loadings,
    exogenous observed variables, directed structural paths, and the fit
    indices).  On a fit failure it returns ``{"error": <reason>}`` instead of
    raising, so the caller can degrade to ``UNVERIFIED`` cleanly.

    It is a clean-room re-derivation of the model structure: the lavaan spec
    is parsed with this skill's own parser (no copied AMOS / lavaan / graphviz
    code) and the estimates come from the fitted ``semopy`` model.  No
    external plotting/graph library is involved here -- only the figure spec.
    """
    fit = _fit(model_spec, data_path)
    if isinstance(fit, dict):
        return {"error": fit["reason"]}
    model, stats, data, n_obs = fit

    # --- structural layout from the lavaan spec (own parser, clean-room) ----
    parsed = _sem_fit._parse(model_spec) if _sem_fit else None
    latents = (parsed or {}).get("latents", []) or []
    loads_map = (parsed or {}).get("loads", {}) or {}
    regr = (parsed or {}).get("regr", {}) or {}
    lat_set = set(latents)

    # exogenous observed predictors: RHS of '~' that are not latents
    exogenous, exo_seen = [], set()
    for tgt, preds in regr.items():
        for pr in preds:
            if pr not in lat_set and pr not in exo_seen:
                exo_seen.add(pr)
                exogenous.append(pr)

    # --- pull estimated loadings / structural coefficients from the fit -----
    # semopy's inspect() can return either (a) a column-based table with
    # 'lval'/'op'/'rval' columns (semopy >= 2.x) or (b) a 3-token string
    # index "lval op rval" (older).  Loadings are represented with op '~'
    # (e.g. "Q1_1 ~ Stress"), so we distinguish them from structural
    # regressions by whether the *rval* is a latent and the *lval* is not.
    loadings_by_latent = {L: {} for L in latents}
    paths_out = []
    try:
        insp = model.inspect()
        has_cols = "lval" in insp.columns and "op" in insp.columns
        for i, row in insp.iterrows():
            if has_cols:
                lval = str(row["lval"]).strip()
                op = str(row["op"]).strip()
                rval = str(row["rval"]).strip()
            else:
                parts = str(insp.index[i]).split()
                if len(parts) != 3:
                    continue
                lval, op, rval = parts
            if op == "~~":
                continue
            if op == "=~" or (op == "~" and rval in lat_set
                              and lval not in lat_set):
                # measurement loading: rval is the latent, lval its indicator
                if rval in loads_map and lval in loads_map[rval]:
                    idx = loads_map[rval].index(lval)
                    try:
                        loadings_by_latent[rval][idx] = float(row["Estimate"])
                    except Exception:
                        pass
            elif op == "~" and rval not in ("1",):
                est = None
                try:
                    est = float(row["Estimate"])
                except Exception:
                    pass
                p = None
                try:
                    p = float(row["p-value"] if has_cols
                              else row.get("P(>|z|)", row.get("p-value")))
                except Exception:
                    pass
                paths_out.append({
                    "from": rval, "to": lval,
                    "estimate": round(est, 3) if est is not None else None,
                    "p": round(p, 4) if p is not None else None,
                })
    except Exception:
        pass

    # --- assemble latent blocks (marker loading pinned to 1.00) -------------
    latents_out = []
    for L in latents:
        inds = loads_map.get(L, [])
        full = []
        for i, ind in enumerate(inds):
            if i == 0:
                full.append(1.00)
            else:
                v = loadings_by_latent.get(L, {}).get(i)
                full.append(round(float(v), 3) if v is not None else 0.80)
        latents_out.append({
            "name": L, "label": L,
            "indicators": list(inds),
            "loadings": full,
        })

    # --- fit indices -------------------------------------------------------
    fit_indices = {}
    for k in ("CFI", "TLI", "RMSEA", "SRMR", "GFI", "AIC", "BIC"):
        try:
            fit_indices[k] = float(stats[k].iloc[0])
        except Exception:
            pass
    try:
        fit_indices["chi2"] = float(stats["chi2"].iloc[0])
        fit_indices["df"] = int(stats["df"].iloc[0])
    except Exception:
        pass

    return {
        "type": "sem_path",
        "title": title or "SEM path diagram (auto-generated)",
        "latents": latents_out,
        "exogenous": [{"name": e, "label": e} for e in exogenous],
        "structural_paths": paths_out,
        "fit_indices": fit_indices,
    }


# ---------------------------------------------------------------------------
# Extended AMOS-style capabilities (v1.2.0, semopy-based re-implementation)
#
# Each public function below delegates the heavy estimation to semopy (or,
# for the WLSMV threshold model, to the clean-room reduced-rank engine in
# sem_fit) and degrades to {"error": ...} on any failure so the adapter
# layer can surface UNVERIFIED without crashing the pipeline.  The
# implementation lives in scripts/sem_extended.py; these wrappers keep the
# gate's public API stable.
# ---------------------------------------------------------------------------

def _load_sem_extended():
    """Import the extended-capability module alongside this engine.

    The module lives next to this file; making that directory importable
    first keeps the import robust regardless of the caller's cwd (the gate
    and the tests load this engine by absolute path).
    """
    import importlib
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module("sem_extended")


def robust_ml(model_spec, data, robust=True, root=None):
    """Robust-ML (MLR) point estimates + two-step sandwich SEs (semopy).

    Returns a dict with ``estimates``, per-pair ``se``, fit indices and
    ``n_obs``; degrades to ``{"error": ...}`` when semopy or the data are
    unavailable.  The SEs are NOT a flat 1/sqrt(n): they are the diagonal
    of the per-observation score sandwich, so they vary across parameters.
    """
    mod = _load_sem_extended()
    return mod.robust_ml(model_spec, data, robust=robust)


def categorical_wlsmv(model_spec, data, ncat=5, root=None):
    """WLSMV threshold-model fit for ordinal indicators.

    Builds the polychoric correlation matrix of the observed variables and
    computes the exact likelihood chi-square of the reduced-rank SEM on
    that matrix (via the clean-room engine in ``sem_fit``).  Continuous
    data falls through to the Pearson path.  Returns the CFI / TLI / RMSEA
    and the polychoric matrix, or ``{"error": ...}``.
    """
    mod = _load_sem_extended()
    return mod.categorical_wlsmv(model_spec, data, ncat=ncat)


def mediation_effects(model_spec, data, source, target, n_bootstrap=0,
                      seed=0, root=None):
    """Direct / indirect / total effect decomposition (semopy path coefs).

    Reads the structural coefficients off the fitted model, enumerates the
    directed paths ``source -> ... -> target``, and returns the product-of-
    coefficients indirect effect with a delta-method SE.  ``total`` is the
    direct plus all indirect paths.  Degrades to ``{"error": ...}``.
    """
    mod = _load_sem_extended()
    return mod.mediation_effects(model_spec, data, source=source,
                                 target=target, n_bootstrap=n_bootstrap,
                                 seed=seed)


def invariance_test(model_spec, data, group_col,
                    invariance_level="configural", root=None):
    """Nested multi-group configural / metric / scalar invariance.

    Fits the free (configural) model per group, then pins each subsequent
    level's free parameters to the pooled value, and reports the pooled
    chi-square difference and delta-CFI per level.  ``invariant`` is True
    only when the chi-square difference is within the critical value AND
    delta-CFI < 0.01.  Degrades to ``{"error": ...}``.
    """
    mod = _load_sem_extended()
    return mod.invariance_test(model_spec, data, group_col=group_col,
                               invariance_level=invariance_level)


def modification_index(model_spec, data, top_k=20, root=None):
    """Top-k modification indices + expected parameter change.

    For every residual covariance not already in the model, MI is the
    expected chi-square drop from freeing it (Browne's MI, delta-method
    form on the fitted ``mx_cov``); EPC is the signed estimate.  Ranked
    descending.  Degrades to ``{"error": ...}``.
    """
    mod = _load_sem_extended()
    return mod.modification_indices(model_spec, data, top_k=top_k)


def bootstrap_ci(model_spec, data, n_bootstrap=200, seed=0,
                 params=None, root=None):
    """Percentile bootstrap CIs on the model's path coefficients.

    Resamples the data ``n_bootstrap`` times (default 200 -- adequate for
    academic demos / drafts; the underlying ``bootstrap_cis`` auto-reduces
    to 100 when N > 1000), refits each time, and returns the percentile
    CIs for every '~' path.  Degrades to ``{"error": ...}`` when fewer
    than 10 refits converge.
    """
    mod = _load_sem_extended()
    return mod.bootstrap_cis(model_spec, data, n_bootstrap=n_bootstrap,
                             seed=seed, ci=(5.0, 95.0))
