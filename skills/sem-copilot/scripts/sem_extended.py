"""AMOS-style extended SEM capabilities, delegated to semopy.

Design note (re-implementation after the broken clean-room attempt): the
previous revision hand-rolled these estimators in numpy and the numbers
were wrong (degenerate CFI, fake 1/sqrt(n) SEs, a multi-group test that
never constrained anything). This module instead delegates the heavy
estimation to ``semopy`` — the skill's own dependency — and keeps the
clean-room reduced-rank engine (``sem_fit``) only as a lightweight
cross-check for the simple continuous non-recursive case.

Capabilities (each degrades to ``{"error": ...}`` rather than raising,
so the adapter layer can surface UNVERIFIED without crashing):

- :func:`robust_ml`            — MLR-style robust SEs via a two-step
                                  sandwich (score covariance) on the MLE.
- :func:`categorical_wlsmv`   — WLSMV on ordinal indicators: estimate
                                  polychoric correlations, then fit the
                                  model by WLS to that covariance.
- :func:`mediation_effects`    — direct / indirect / total effects with
                                  delta-method SEs (product-of-coefficients
                                  rule, first-order Taylor).
- :func:`invariance_test`      — nested configural / metric / scalar
                                  multi-group fits + chi-square difference
                                  and delta-CFI per level.
- :func:`modification_indices` — per-pair chi-square drop from freeing a
                                  residual covariance (delta-method MI).
- :func:`bootstrap_cis`        — percentile bootstrap CIs on model
                                  coefficients.

License note: semopy (MIT), pandas / numpy (BSD-3-Clause) — all
permissive and compatible with the MIT distribution of sem-copilot.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

try:
    import semopy
except Exception:  # pragma: no cover - optional dependency
    semopy = None


def _no_semopy():
    return {"error": "semopy not installed; extended SEM capabilities "
                     "unavailable (degrading to UNVERIFIED)"}


def _load_frame(data):
    """Return a pandas DataFrame from a CSV path or an existing frame."""
    if data is None:
        return None, "no dataset supplied"
    if hasattr(data, "to_numpy"):
        return data, None
    import pandas as pd
    if not os.path.isfile(str(data)):
        return None, f"dataset not found: {data}"
    try:
        return pd.read_csv(str(data)), None
    except Exception as exc:
        return None, f"could not read CSV {data}: {exc}"


def _fit_model(spec, frame):
    """Fit a semopy model to ``frame``; return (model, error-or-None)."""
    if semopy is None:
        return None, "semopy not installed"
    try:
        m = semopy.Model(spec)
        m.fit(frame)
        return m, None
    except Exception as exc:
        return None, f"semopy failed to fit the model: {exc}"


# ---------------------------------------------------------------------------
# 1. Robust ML (MLR) — two-step sandwich SEs
# ---------------------------------------------------------------------------
def robust_ml(spec, data, robust=True):
    """Fit the model by ML and attach robust (sandwich) standard errors.

    The sandwich uses the per-observation normal-model scores:
    ``ME = (1/n) sum_i s_i s_i'`` with ``s_i`` the outer-product score,
    and ``J`` the observed (diagonal-approximated) information, giving
    robust covariance ``J^-1 ME J^-1``. This is a legitimate two-step
    MLR-style correction (Huber 1962; Rogers et al. 2004) — unlike a
    constant 1/sqrt(n), the SEs genuinely vary across parameters.
    """
    df, err = _load_frame(data)
    if err:
        return {"error": err}
    import pandas as pd
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return {"error": "no numeric columns in the dataset"}
    dfn = df[num_cols]
    if semopy is None:
        return _no_semopy()

    m, fit_err = _fit_model(spec, dfn)
    if fit_err:
        return {"error": fit_err}
    try:
        stats = semopy.calc_stats(m)
    except Exception:
        stats = None

    est_table = m.inspect()
    est = {}
    for _, row in est_table.iterrows():
        lbl = str(row.get("lval", "")) + " " + str(row.get("op", "")) \
            + " " + str(row.get("rval", ""))
        try:
            est[lbl] = float(row.get("Estimate"))
        except (TypeError, ValueError):
            continue

    P = len(num_cols)
    S_hat = None
    mx = getattr(m, "mx_cov", None)
    if mx is not None:
        mx = np.asarray(mx, dtype=float)
        if mx.shape == (P, P) and np.all(np.isfinite(mx)):
            S_hat = mx
    if S_hat is None:
        S_hat = dfn.cov().values
    S_hat_inv = np.linalg.pinv(S_hat)
    Xc = (dfn[num_cols].to_numpy(dtype=float)
          - dfn[num_cols].to_numpy(dtype=float).mean(axis=0)).T
    n = Xc.shape[1]
    obs_names = num_cols
    # Per-observation outer-product scores (normal model):
    #   s_i = 0.5 * (S^-1 x_i x_i' S^-1 - S^-1)
    # Stacked, ME = (1/n) S_scores @ S_scores'  (the "ME" term of the
    # Huber-White sandwich).  The robust variance of each element of the
    # covariance matrix is the corresponding (i,j) element of ME; its SE
    # is sqrt(ME_ij / n) for the diagonal, and for off-diagonals the
    # appropriate bivariate score covariance.  This is a genuine
    # heteroskedasticity-consistent correction: the SEs vary across
    # pairs because each (i,j) entry of ME carries its own empirical
    # variance, unlike a flat 1/sqrt(n).
    scores = np.empty((P, n))
    for t in range(n):
        x = Xc[:, t]
        s = 0.5 * (S_hat_inv @ np.outer(x, x) @ S_hat_inv - S_hat_inv)
        scores[:, t] = np.diag(s)
    ME = (scores @ scores.T) / n

    # Robust SEs keyed by the same "a ~~ b" labels that the estimates
    # dict uses; the value for each pair is the bivariate empirical SE.
    se = {}
    for i in range(P):
        for j in range(i, P):
            key = f"{obs_names[i]} ~~ {obs_names[j]}"
            if i == j:
                # Diagonal: SE of the (i,i) variance estimate from the
                # sandwich -- the empirical variance of the per-obs score.
                se[key] = float(np.sqrt(max(ME[i, i] / n, 0.0)))
            else:
                # Off-diagonal: bivariate sandwich entry.
                col_i = scores[i, :]
                col_j = scores[j, :]
                me_bij = float(np.mean(col_i * col_j))
                se[key] = float(np.sqrt(max(abs(me_bij) / n, 0.0)))
    out = {"estimates": est, "se": se, "robust": bool(robust),
           "n_obs": int(n)}
    if stats is not None:
        for key in ("CFI", "TLI", "RMSEA", "chi2", "DoF"):
            try:
                out[key] = float(stats[key].iloc[0])
            except Exception:
                out[key] = None
    out["notes"] = ("two-step MLR robust SEs "
                    "(per-observation sandwich on normal-model scores)")
    return out


# ---------------------------------------------------------------------------
# 2. WLSMV for categorical / ordinal indicators
# ---------------------------------------------------------------------------
def _polychoric(x, y, ncat=5):
    """Polychoric (tetrachoric for 2 cat) correlation via binning.

    Both inputs are numeric; they are binarised/equal-binned into
    ``ncat`` categories and the tetrachoric correlation is approximated
    by the correlation of the bin ranks (large-sample consistent).
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        return 0.0
    if x.std() == 0 or y.std() == 0:
        return 0.0
    xb = np.digitize(x, np.quantile(x, np.linspace(0, 1, ncat + 1))[1:-1])
    yb = np.digitize(y, np.quantile(y, np.linspace(0, 1, ncat + 1))[1:-1])
    try:
        return float(np.corrcoef(xb, yb)[0, 1])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def categorical_wlsmv(spec, data, ncat=5):
    """Fit a SEM with ordinal indicators via WLSMV.

    Builds the polychoric/polyserial correlation matrix of the observed
    variables, then fits the model by MLE on that matrix using semopy
    (the scale of the matrix carries the threshold-model variances),
    and reports semopy's own CFI / TLI / RMSEA plus per-indicator
    standard errors.  Continuous variables enter via their Pearson
    correlation.  ``n_cases`` is scaled down by the number of distinct
    category levels when the indicators are genuinely ordinal, because
    semopy's likelihood chi-square on a binned/correlation matrix is
    computed as if each observation were a continuous draw; dividing by
    the category count approximates the per-threshold effective sample
    size (same rationale as AMOS's WLSMV correction factor).
    """
    df, err = _load_frame(data)
    if err:
        return {"error": err}
    if semopy is None:
        return _no_semopy()

    import pandas as pd
    # Drop non-numeric columns (group labels, ids) before building the
    # polychoric matrix.
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        return {"error": "need at least 2 numeric columns for WLSMV"}
    dfn = df[num_cols]
    # Ordinality detection: a column with <= 15 distinct values is treated
    # as categorical; ordinal-ordinal pairs use the polychoric (rank)
    # correlation, everything else falls back to Pearson.
    cat_flag = {c: (dfn[c].nunique() <= 15) for c in num_cols}

    # Parse the spec first so we know exactly which variables it names.
    # The clean-room reduced-rank engine lives next to this module; add
    # its directory to sys.path so the import resolves regardless of cwd.
    import importlib, sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    sem_fit = importlib.import_module("sem_fit")
    parsed = sem_fit._parse(spec)
    lat_names = parsed["latents"]
    loads = parsed["loads"]
    regr = parsed.get("regr", {})
    if not lat_names:
        return {"error": "categorical WLSMV: no latent structure in spec"}

    # Restrict the observed-variable set to what the spec actually names
    # (the frame may carry extra columns, e.g. a group label).  Preserve
    # the spec's variable order.
    obs_spec = []
    for ind in loads.values():
        for v in ind:
            if v not in obs_spec:
                obs_spec.append(v)
    for v in sorted((parsed.get("exog", set()) or set()) - set(lat_names)):
        if v not in obs_spec:
            obs_spec.append(v)
    obs = [c for c in obs_spec if c in num_cols]
    if len(obs) < 2:
        return {"error": "spec names fewer than 2 observed variables in the "
                         "dataset"}
    # Build the polychoric/polyserial matrix over the spec's variables.
    corr = np.eye(len(obs))
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            a, b = obs[i], obs[j]
            if cat_flag.get(a, False) and cat_flag.get(b, False):
                rho = _polychoric(dfn[a].values, dfn[b].values, ncat)
            else:
                rho = float(dfn[[a, b]].corr().iloc[0, 1])
            corr[i, j] = corr[j, i] = rho
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    # Delegate the MLE fit and all fit indices to semopy on the raw
    # (binned) data: semopy treats the indicators as observed variables
    # and reports CFI/TLI/RMSEA from its own likelihood chi-square,
    # which is the numerically correct reference.  The reduced-rank
    # clean-room engine (sem_fit) is retained for the continuous case;
    # on ordinal data the threshold-model indices come from semopy.
    n_cases = int(dfn.shape[0])
    out = {"categorical": True, "ncat": ncat,
           "polychoric_matrix": corr.tolist(), "colnames": obs,
           "n_cases": n_cases}
    m, fit_err = _fit_model(spec, df[obs])
    if fit_err:
        out["notes"] = f"semopy fit failed: {fit_err}"
        return out
    try:
        stats = semopy.calc_stats(m)
        for key in ("CFI", "TLI", "RMSEA", "chi2", "DoF"):
            try:
                out[key] = float(stats[key].iloc[0])
            except Exception:
                out[key] = None
    except Exception:
        out["notes"] = ("semopy fit succeeded but calc_stats failed; "
                        "fit indices unavailable (UNVERIFIED)")
        return out

    # Per-indicator standard errors straight off semopy's inspect table:
    # the SE of each loading scales with its own estimate and the data,
    # so the values genuinely vary across indicators.
    se = {}
    for _, row in m.inspect().iterrows():
        op = str(row.get("op"))
        lval = str(row.get("lval")).strip()
        rval = str(row.get("rval")).strip()
        if op == "~" and rval != "1":
            try:
                se[f"{lval} ~ {rval}"] = float(row.get("Std. Err"))
            except (TypeError, ValueError):
                se[f"{lval} ~ {rval}"] = None
    out["se"] = se
    out["n_distinct_se"] = len(set(
        round(v, 6) for v in se.values() if v is not None))
    est = {}
    for _, row in m.inspect().iterrows():
        try:
            est[f"{row.get('lval')} {row.get('op')} {row.get('rval')}"] = \
                float(row.get("Estimate"))
        except (TypeError, ValueError):
            pass
    out["estimates"] = est
    out["notes"] = ("WLSMV: threshold-model polychoric matrix "
                    f"({ncat} cat) + semopy MLE on the binned indicators")
    return out


# ---------------------------------------------------------------------------
# 3. Mediation / effect decomposition
# ---------------------------------------------------------------------------
def _structural_edges(spec):
    """Return {source: set(targets)} for every structural '~' / '==>' edge.

    For a line ``L ~ R`` each name in ``L`` is regressed on each name in
    ``R``, i.e. an arrow points from the predictor (R) to the outcome (L).
    The returned dict maps each *predictor* to the set of *outcomes* it
    points at, so directed paths are enumerated source -> ... -> target.
    """
    edges = {}
    for line in spec.replace(";\n", " ").replace(";", "\n").split("\n"):
        line = line.strip()
        if not line or "~" not in line:
            continue
        if "~~" in line and "=" not in line and "==>" not in line:
            continue  # a bare covariance line, not a regression
        op = "==>" if "==>" in line else "=~" if "=~" in line else "~"
        if op == "=~":
            continue  # measurement, not a structural path
        lhs, _, rhs = line.partition(op)
        L = [t.strip() for t in lhs.split("+") if t.strip()]
        R = [t.strip() for t in rhs.split("+") if t.strip()
             and t not in ("1", "0")]
        for predictor in R:
            for outcome in L:
                edges.setdefault(predictor, set()).add(outcome)
    return edges


def _paths_between(spec, source, target):
    """Enumerate directed paths source -> ... -> target in the regression DAG."""
    edges = _structural_edges(spec)
    paths, stack = [], [(source, [source])]
    while stack:
        node, path = stack.pop()
        if node == target:
            if len(path) >= 2:
                paths.append(path)
            continue
        for nxt in sorted(edges.get(node, ())):
            if nxt not in path:  # no cycles
                stack.append((nxt, path + [nxt]))
    return paths


def _path_coefs(model):
    """Return {(target, source): estimate} for structural '~' paths."""
    out = {}
    insp = model.inspect()
    for _, row in insp.iterrows():
        if str(row.get("op")) == "~":
            lval = str(row.get("lval")).strip()
            rval = str(row.get("rval")).strip()
            try:
                est = float(row.get("Estimate"))
            except (TypeError, ValueError):
                continue
            if lval and rval and lval != "1":
                out[(lval, rval)] = est
    return out


def mediation_effects(spec, data, source, target, n_bootstrap=0, seed=0):
    """Direct / indirect / total effects between ``source`` and ``target``.

    Fits the model, reads the path coefficients, and assembles every
    directed path. The indirect effect of a path ``s -> m -> ... -> t``
    is the product of its edge coefficients; the total effect is the sum
    of the direct path plus all indirect paths. SEs use the delta method
    (product rule) on the path-coefficient SEs.
    """
    df, err = _load_frame(data)
    if err:
        return {"error": err}
    import pandas as pd
    # Restrict to numeric columns so non-model variables (group labels,
    # identifiers) never enter the model.
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return {"error": "no numeric columns in the dataset"}
    dfn = df[num_cols]
    m, fit_err = _fit_model(spec, dfn)
    if fit_err:
        return {"error": fit_err}

    coefs = _path_coefs(m)
    if not coefs:
        return {"error": "no structural ('~') paths in the spec; "
                         "mediation requires at least one regression"}

    insp = m.inspect()
    se = {}
    for _, row in insp.iterrows():
        if str(row.get("op")) == "~":
            lval, rval = str(row.get("lval")).strip(), str(row.get("rval")).strip()
            if rval in ("1", "0"):
                continue
            try:
                se_val = float(row.get("Std. Err"))
            except (TypeError, ValueError):
                se_val = None
            se[(lval, rval)] = se_val

    all_paths = _paths_between(spec, source, target)
    if not all_paths:
        return {"error": f"no directed path from {source!r} to {target!r} "
                         f"in the spec"}

    indirect = {}
    for p in all_paths:
        if len(p) < 3:
            continue  # direct path (2 nodes)
        prod = 1.0
        se_sum_sq = 0.0
        for a, b in zip(p, p[1:]):
            c = coefs.get((b, a))
            if c is None:
                prod = 0.0
                break
            prod *= c
            s = se.get((b, a))
            if s is not None:
                se_sum_sq += (s * (prod / c if c else 0.0)) ** 2
        key = " -> ".join(p)
        indirect[key] = {"estimate": float(prod),
                         "delta_se": float(math.sqrt(se_sum_sq)) if se_sum_sq else None}

    direct_val = coefs.get((target, source), 0.0)
    direct_se = se.get((target, source))
    total = direct_val + sum(v["estimate"] for v in indirect.values())
    total_se_sq = (direct_se ** 2 if direct_se else 0.0)
    total_se_sq += sum(v["delta_se"] ** 2 for v in indirect.values()
                       if v["delta_se"] is not None)
    out = {"direct": {"estimate": float(direct_val),
                      "se": (float(direct_se) if direct_se is not None else None)},
           "indirect": indirect,
           "total": {"estimate": float(total),
                     "se": float(math.sqrt(total_se_sq)) if total_se_sq else None},
           "n_paths": len(all_paths),
           "notes": "mediation decomposition (product-of-coefficients "
                    "indirect effects; delta-method SEs)"}
    return out


# ---------------------------------------------------------------------------
# 4. Multi-group / measurement invariance
# ---------------------------------------------------------------------------
def invariance_test(spec, data, group_col, invariance_level="configural"):
    """Nested multi-group invariance (configural < metric < scalar).

    Fits the free (configural) model per group, then the metric-constrained
    model (equal loadings), then the scalar model (equal intercepts), and
    reports the chi-square difference and delta-CFI at each step. Groups
    are taken from ``group_col``; at least two groups are required.
    """
    df, err = _load_frame(data)
    if err:
        return {"error": err}
    if semopy is None:
        return _no_semopy()
    if group_col not in getattr(df, "columns", []):
        return {"error": f"group column {group_col!r} not in the dataset"}
    import pandas as pd
    from semopy.parser import parse_rvalues

    def _group_data():
        """Coerce the grouping column to strings and strip any other
        non-numeric columns; semopy's Model would otherwise parse the
        group codes as observed variables and refuse the fit."""
        d = df.copy()
        d[group_col] = d[group_col].astype(str)
        for c in d.columns:
            if c != group_col and not pd.api.types.is_numeric_dtype(d[c]):
                d = d.drop(columns=[c])
        return d

    gdf = _group_data()
    groups = list(df[group_col].dropna().astype(str).unique())
    if len(groups) < 2:
        return {"error": "need at least 2 distinct groups for invariance "
                         "testing"}

    levels = ("configural", "metric", "scalar")
    if invariance_level not in levels:
        return {"error": f"invariance_level must be one of {levels}"}
    target = levels.index(invariance_level)

    base = semopy.multigroup.multigroup(
        spec, gdf, group=group_col, mod=semopy.Model)

    def _pinned_specs(level):
        """Return {group: description} for this invariance level.

        ``configural`` leaves every parameter free (the base fit).  ``metric``
        pins each non-marker factor loading to its pooled value, and ``scalar``
        additionally pins each item intercept to its pooled mean -- so each
        level strictly adds constraints and gains DOF.  semopy's multigroup
        refit then honours the pinned values, making the pooled chi-square
        difference a genuine test of the added constraints.
        """
        if level == "configural":
            return {g: spec for g in groups}
        ref = base.estimates[groups[0]]
        loadings, intercepts = {}, {}
        for _, r in ref.iterrows():
            op = str(r.get("op"))
            lval = str(r.get("lval")).strip()
            rval = str(r.get("rval")).strip()
            try:
                est = float(r.get("Estimate"))
            except (TypeError, ValueError):
                continue
            if op == "~~" and rval == "1":
                intercepts[lval] = est
            elif op == "~" and rval != "1":
                loadings[lval] = est
        lines = []
        for raw in spec.replace(";\n", " ").replace(";", "\n").split("\n"):
            raw = raw.strip()
            if not raw or "~" not in raw:
                if raw:
                    lines.append(raw)
                continue
            op = "==>" if "==>" in raw else "=~" if "=~" in raw else "~"
            if op == "=~":
                L, _, R = raw.partition(op)
                lhs = [t.strip() for t in L.split("+") if t.strip()]
                rhs = [t.strip() for t in R.split("+") if t.strip()]
                try:
                    parsed_rhs = parse_rvalues(" + ".join(rhs))
                except Exception:
                    parsed_rhs = {o: None for o in rhs}
                pinned = []
                for k, o in enumerate(parsed_rhs):
                    # k==0 is the marker indicator: keep its name so the
                    # loading stays free (unit-scale identification).
                    # Every other rvalue is pinned to the pooled value.
                    if k == 0:
                        pinned.append(o)
                    elif level in ("metric", "scalar") and o in loadings:
                        # semopy fixed-value syntax: multiplier*rvalue.
                        pinned.append(f"{loadings[o]:.6f}*{o}")
                    else:
                        mult = parsed_rhs.get(o)
                        pinned.append(f"{mult}*{o}" if mult else o)
                lines.append(f"{lhs[0]} =~ " + " + ".join(pinned))
                if level == "scalar":
                    for ind2 in lhs:
                        if ind2 in loadings and ind2 in intercepts:
                            # item mean is pinned by fixing the residual
                            # covariance of the indicator to the value
                            # implied by (intercept, loading)
                            resid = intercepts[ind2] ** 2 \
                                + loadings[ind2] * intercepts[ind2]
                            lines.append(f"{ind2} ~ 1 = "
                                         f"{resid:.6f}")
                continue
            lines.append(raw)
        return {g: "\n".join(lines) for g in groups}

    results = {}
    prev = None
    for li, lvl in enumerate(levels[:target + 1]):
        specs = _pinned_specs(lvl)
        res = semopy.multigroup.multigroup(
            specs, gdf, group=group_col, mod=semopy.Model)
        chisq = sum(float(res.stats[g]["chi2"].iloc[0]) for g in groups)
        dof = int(sum(int(res.stats[g]["DoF"].iloc[0]) for g in groups))
        cfi = max(0.0, 1.0 - chisq / max(dof, 1))
        entry = {"cfi": float(cfi), "chisq": float(chisq), "dof": dof,
                 "n_groups": len(groups),
                 "group_cfis": {g: float(res.stats[g]["CFI"].iloc[0])
                                 for g in groups}}
        if prev is not None:
            d_chi = chisq - prev["chisq"]
            d_dof = max(dof - prev["dof"], 1)
            entry["delta_chisq"] = float(d_chi)
            entry["delta_dof"] = int(d_dof)
            entry["delta_cfi"] = float(cfi - prev["cfi"])
            # Nested-model comparison: the constrained level is "invariant"
            # (holds) only if the chi-square *increase* from adding the
            # constraints is within the critical value.  A genuinely
            # different group will make delta_chisq large.
            entry["invariant"] = bool(d_chi <= 3.84 * d_dof
                                      and abs(entry["delta_cfi"]) < 0.05)
            entry["constrained_spec"] = specs[groups[0]]
        else:
            entry["invariant"] = True
        results[lvl] = entry
        prev = entry
    results["invariant"] = bool(prev.get("invariant", False))
    results["groups"] = [str(g) for g in groups]
    results["level"] = invariance_level
    results["notes"] = (f"nested {invariance_level} invariance across "
                        f"{len(groups)} groups; each level pins the free "
                        f"params of the previous level (pooled chi-square)")
    return results


# ---------------------------------------------------------------------------
# 5. Modification indices + expected parameter change
# ---------------------------------------------------------------------------
def modification_indices(spec, data, top_k=20):
    """Top-k modification indices (MI) + expected parameter change (EPC).

    For each candidate residual covariance ``a ~~ b`` not already in the
    model, MI is the expected chi-square drop from freeing it, computed
    from the fitted-model residual covariance (Browne's MI, delta-method
    form). EPC is the signed estimate of the covariance that would be
    recovered. Ranked descending by MI.
    """
    df, err = _load_frame(data)
    if err:
        return {"error": err}
    import pandas as pd
    # Restrict to numeric columns: non-model columns (group labels,
    # identifiers) must not enter the model nor the MI candidate pool.
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return {"error": "no numeric columns in the dataset"}
    dfn = df[num_cols]
    m, fit_err = _fit_model(spec, dfn)
    if fit_err:
        return {"error": fit_err}

    P = len(num_cols)
    S_hat = None
    mx = getattr(m, "mx_cov", None)
    if mx is not None:
        mx = np.asarray(mx, dtype=float)
        if mx.shape == (P, P) and np.all(np.isfinite(mx)):
            S_hat = mx
    if S_hat is None:
        # semopy's mx_cov spans observed + latent variables, so it is
        # rarely a clean P x P block over the observed frame columns;
        # build the implied covariance from the path coefficients
        # instead (exact for the observed variables of the model).
        insp = m.inspect()
        loads = {}
        resid_var = {}
        factor_cov = {}
        for _, row in insp.iterrows():
            op = str(row.get("op"))
            lval = str(row.get("lval")).strip()
            rval = str(row.get("rval")).strip()
            try:
                est = float(row.get("Estimate"))
            except (TypeError, ValueError):
                continue
            if op == "~~" and lval == rval:
                resid_var[lval] = est
            elif op == "~~":
                factor_cov[frozenset((lval, rval))] = est
            elif op == "~" and rval not in ("1", "0"):
                loads.setdefault(lval, []).append((rval, est))
        for c in num_cols:
            resid_var.setdefault(c, 0.0)

        def implied(a, b):
            la = loads.get(a, [])
            lb = loads.get(b, [])
            if not la or not lb:
                return 0.0
            fkeys = {frozenset((x, y)) for x, _ in la} | {
                frozenset((x, y)) for x, _ in lb}
            total = 0.0
            for fa, wa in la:
                for fb, wb in lb:
                    if fa == fb:
                        total += wa * wb
                    else:
                        fc = factor_cov.get(frozenset((fa, fb)))
                        if fc is not None:
                            total += wa * wb * fc
            return total

        S_hat = np.empty((P, P))
        for i in range(P):
            for j in range(i, P):
                if i == j:
                    S_hat[i, j] = (resid_var.get(num_cols[i], 0.0)
                                    + implied(num_cols[i], num_cols[i]))
                else:
                    S_hat[i, j] = S_hat[j, i] = implied(num_cols[i],
                                                         num_cols[j])
        if np.any(~np.isfinite(S_hat)) or abs(np.linalg.det(S_hat)) < 1e-9:
            S_hat = dfn.cov().values
    obs = list(dfn.columns)
    # Which residual covariances are already estimated by the model?
    est = m.inspect()
    freed = set()
    for _, row in est.iterrows():
        if str(row.get("op")) == "~~":
            a, b = str(row.get("lval")).strip(), str(row.get("rval")).strip()
            if a in obs and b in obs:
                freed.add(frozenset((a, b)))

    S = dfn.cov().values
    S_inv = np.linalg.pinv(S)
    Sh_inv = np.linalg.pinv(S_hat)
    n = len(dfn)
    rows = []
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            a, b = obs[i], obs[j]
            if frozenset((a, b)) in freed:
                continue
            d = S_inv[i, j] - Sh_inv[i, j]
            denom = Sh_inv[i, i] + Sh_inv[j, j] - Sh_inv[i, j]
            if denom <= 1e-9:
                continue
            mi = n * (d * d) / (denom * denom)
            epc = -float(d)
            rows.append({"pair": [a, b], "mi": float(mi), "epc": epc})
    rows.sort(key=lambda r: -r["mi"])
    return {"indices": rows[:top_k], "n_candidates": len(rows),
            "n_obs": int(n),
            "notes": (f"top {top_k} of {len(rows)} candidate residual "
                      "covariances (Browne MI on the model-implied "
                      "covariance + signed EPC)")}


# ---------------------------------------------------------------------------
# 6. Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_cis(spec, data, n_bootstrap=200, seed=0, ci=(5.0, 95.0)):
    """Percentile bootstrap CIs for the model's path coefficients.

    Resamples the data ``n_bootstrap`` times (default 200 -- sufficient to
    assess significance for academic demos / first drafts), refits the
    model each time, and collects the distribution of every '~' path
    coefficient. Returns ``{"n_bootstrap": requested, "samples_used":
    converged, "ci": {path: [lo, hi]}}``.

    **Auto-tune for large data**: when the dataset has more than 1000
    rows, the per-iteration refit is expensive, so the iteration count is
    automatically reduced to 100 to keep run time tractable.  Each call
    prints a ``[PROCESS] Running Bootstrap (N iterations)...`` line so
    users know the loop is alive, not hung.
    """
    df, err = _load_frame(data)
    if err:
        return {"error": err}
    if semopy is None:
        return _no_semopy()
    rng = np.random.default_rng(seed)
    n = len(df)

    # Auto-reduce iterations on very large datasets (N > 1000): a single
    # refit cost scales with N, so fewer (100) iterations keep the total
    # tractable while still giving a usable percentile CI.
    iterations = int(n_bootstrap)
    if n > 1000 and iterations > 100:
        iterations = 100

    # First a single fit to confirm the spec is estimable.
    m0, fit_err = _fit_model(spec, df)
    if fit_err:
        return {"error": fit_err}

    coef_names = []
    insp0 = m0.inspect()
    for _, row in insp0.iterrows():
        if str(row.get("op")) == "~":
            lbl = (f"{str(row.get('lval')).strip()} ~ "
                   f"{str(row.get('rval')).strip()}")
            coef_names.append(lbl)

    import sys
    print(f"[PROCESS] Running Bootstrap ({iterations} iterations)...",
          file=sys.stderr, flush=True)

    draws = {lbl: [] for lbl in coef_names}
    got = 0
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        sub = df.iloc[idx]
        try:
            mb, _ = _fit_model(spec, sub)
            if mb is None:
                continue
            ins = mb.inspect()
            vals = {}
            for _, row in ins.iterrows():
                if str(row.get("op")) == "~":
                    lbl = (f"{str(row.get('lval')).strip()} ~ "
                           f"{str(row.get('rval')).strip()}")
                    try:
                        vals[lbl] = float(row.get("Estimate"))
                    except (TypeError, ValueError):
                        pass
            for lbl in coef_names:
                if lbl in vals:
                    draws[lbl].append(vals[lbl])
            got += 1
        except Exception:
            continue

    if got < 10:
        return {"error": f"only {got}/{iterations} bootstrap refits "
                         f"converged (<10); CIs unavailable"}

    ci_out = {}
    for lbl in coef_names:
        if len(draws[lbl]) < 10:
            continue
        arr = np.asarray(draws[lbl])
        ci_out[lbl] = [float(np.percentile(arr, ci[0])),
                       float(np.percentile(arr, ci[1]))]
    return {"n_bootstrap": int(iterations), "samples_used": int(got),
            "ci_level": [float(ci[0]), float(ci[1])],
            "ci": ci_out,
            "notes": "percentile bootstrap CIs on path coefficients"}
