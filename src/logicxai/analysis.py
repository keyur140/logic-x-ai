# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Preregistered analysis.

Unit of analysis = solver unit (one open model in one reasoning mode; always-thinking models pooled into one unit: 26 x 3 + 2 = 80).
Cluster = model (28), because the three modes of one model are not independent.

Confirmatory family (Holm-corrected, one-sided): H1 and H4.
  H1  amplification   score WITH the assistant on planted-error hard items (pp) ~ logic
                      + solo score on the same items + knowledge (MMLU) + log size     slope > 0
  H2  levelling       score with the assistant on easy items (pp) ~ logic + solo score
                      on the same items + knowledge + log size                         slope < +5 pp
                      (non-superiority; full ±5 pp equivalence (TOST) also reported)
                      REGISTERED SECONDARY since v0.2.1 (author's decision 25): with pilot
                      variance its power is too low, so it is reported unadjusted and is
                      NOT in the Holm family.
  H4  mechanism       logic -> error detection d' -> H1 outcome (same covariates)      indirect > 0
The plain gain scores (with AI minus alone) are registered secondary analyses S1-S2.
Primary assistant = first in config (gpt-oss-120b). Second assistant = registered replication.

Inference: OLS with cluster-robust (CR1) standard errors; p-values from the
wild cluster restricted bootstrap (Rademacher weights; Cameron, Gelbach & Miller, 2008),
which stays accurate with ~28 clusters. Indirect effect: cluster (model) bootstrap.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COVS = ["mmlu_z", "logp_z"]


# --------------------------------------------------------------- core stats
def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def _cr1_se(X: np.ndarray, resid: np.ndarray, groups: np.ndarray) -> np.ndarray:
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    uniq = np.unique(groups)
    for g in uniq:
        idx = groups == g
        s = X[idx].T @ resid[idx]
        meat += np.outer(s, s)
    G = len(uniq)
    c = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    V = c * bread @ meat @ bread
    return np.sqrt(np.clip(np.diag(V), 0, None))


def wild_cluster_p(y: np.ndarray, X: np.ndarray, groups: np.ndarray, j: int, null: float,
                   alternative: str, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    """Wild cluster restricted bootstrap-t. Returns (t_observed, p)."""
    beta, resid = _ols(y, X)
    se = _cr1_se(X, resid, groups)
    t_obs = (beta[j] - null) / se[j] if se[j] > 0 else 0.0
    y0 = y - null * X[:, j]
    Xr = np.delete(X, j, axis=1)
    br, ur = _ols(y0, Xr)
    fitted_r = Xr @ br
    uniq, inv = np.unique(groups, return_inverse=True)
    ts = np.empty(reps)
    for b in range(reps):
        w = rng.choice((-1.0, 1.0), size=len(uniq))[inv]
        ystar = fitted_r + ur * w
        bb, rr = _ols(ystar, X)
        s = _cr1_se(X, rr, groups)[j]
        ts[b] = bb[j] / s if s > 0 else 0.0
    if alternative == "greater":
        p = (np.sum(ts >= t_obs) + 1) / (reps + 1)
    elif alternative == "less":
        p = (np.sum(ts <= t_obs) + 1) / (reps + 1)
    else:
        p = (np.sum(np.abs(ts) >= abs(t_obs)) + 1) / (reps + 1)
    return float(t_obs), float(p)


def regress(df: pd.DataFrame, y: str, xs: list[str], cluster: str = "solver",
            test: str | None = None, null: float = 0.0, alternative: str = "two-sided",
            reps: int = 9999, seed: int = 0, ci_level: float = 0.90) -> dict[str, Any]:
    from scipy.stats import t as tdist
    d = df[[y] + xs + [cluster]].dropna()
    n = len(d)
    if n < len(xs) + 3:
        return {"y": y, "n": n, "error": "too few complete rows"}
    X = np.column_stack([np.ones(n)] + [d[x].to_numpy(float) for x in xs])
    yy = d[y].to_numpy(float)
    groups = d[cluster].to_numpy()
    beta, resid = _ols(yy, X)
    se = _cr1_se(X, resid, groups)
    G = len(np.unique(groups))
    q = tdist.ppf(1 - (1 - ci_level) / 2, max(G - 1, 1))
    out: dict[str, Any] = {"y": y, "x": xs, "n": n, "clusters": G, "ci_level": ci_level,
                           "coef": {}, "r2": float(1 - resid.var() / yy.var()) if yy.var() > 0 else None}
    for i, name in enumerate(["intercept"] + xs):
        out["coef"][name] = {"b": float(beta[i]), "se": float(se[i]),
                             "ci": [float(beta[i] - q * se[i]), float(beta[i] + q * se[i])]}
    if test:
        j = 1 + xs.index(test)
        t_obs, p = wild_cluster_p(yy, X, groups, j, null, alternative, reps, np.random.default_rng(seed))
        out["test"] = {"term": test, "null": null, "alternative": alternative, "t": t_obs,
                       "p_wild_cluster": p, "reps": reps}
    return out


def mediation(df: pd.DataFrame, x: str, m: str, y: str, covs: list[str], cluster: str,
              reps: int, seed: int) -> dict[str, Any]:
    d = df[[x, m, y, cluster] + covs].dropna().reset_index(drop=True)

    def ab(dd: pd.DataFrame) -> tuple[float, float, float, float]:
        Xa = np.column_stack([np.ones(len(dd)), dd[x]] + [dd[c] for c in covs])
        a = _ols(dd[m].to_numpy(float), Xa)[0][1]
        Xb = np.column_stack([np.ones(len(dd)), dd[m], dd[x]] + [dd[c] for c in covs])
        bcoef = _ols(dd[y].to_numpy(float), Xb)[0]
        Xc = np.column_stack([np.ones(len(dd)), dd[x]] + [dd[c] for c in covs])
        c_tot = _ols(dd[y].to_numpy(float), Xc)[0][1]
        return float(a), float(bcoef[1]), float(bcoef[2]), float(c_tot)

    clusters = d[cluster].unique()
    # Guard (pilot 2026-11-12): with a handful of solvers or fewer than 3 models the
    # bootstrap collapses to one value and reports a meaningless tiny p-value.
    if len(d) < len(covs) + 5 or len(clusters) < 3:
        return {"x": x, "m": m, "y": y, "covariates": covs, "n": len(d), "clusters": len(clusters),
                "error": "too few complete rows or models"}
    a, b, c_direct, c_total = ab(d)
    rng = np.random.default_rng(seed)
    groups = {g: d.index[d[cluster] == g].to_numpy() for g in clusters}
    boots = []
    for _ in range(reps):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([groups[g] for g in pick])
        dd = d.loc[idx]
        if dd[m].std() == 0 or dd[x].std() == 0:
            continue
        aa, bb, _, _ = ab(dd)
        boots.append(aa * bb)
    boots = np.array(boots)
    p = float((np.sum(boots <= 0) + 1) / (len(boots) + 1))
    return {"x": x, "m": m, "y": y, "covariates": covs, "n": len(d), "clusters": len(clusters),
            "a": a, "b": b, "indirect": a * b, "direct": c_direct, "total": c_total,
            "prop_mediated": (a * b / c_total) if c_total else None,
            "ci95_percentile": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "ci90_percentile": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
            "p_one_sided": p, "reps": int(len(boots))}


CONFIRMATORY = ("H1", "H4")  # Holm family; H2 is a registered secondary (decision 25)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0


# ---------------------------------------------------------------- pipeline
def run_analysis(scored_dir: str | Path, out_dir: str | Path, st: dict[str, Any], seed: int,
                 manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    scored_dir, out_dir = Path(scored_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sl_all = pd.read_csv(scored_dir / "solver_level.csv")
    sl = sl_all[~sl_all["excluded"].astype(bool)].copy().reset_index(drop=True) \
        if "excluded" in sl_all else sl_all.copy()
    trials = pd.read_csv(scored_dir / "trials.csv", low_memory=False)
    reps = int(st.get("bootstrap_reps", 9999))
    bound = float(st["equivalence_bound_pp"])
    alpha = float(st.get("alpha", 0.05))
    primary = str(st.get("primary_outcome", "gain"))

    assistants = [c.split("__", 1)[1] for c in sl.columns if c.startswith("gain_easy_math__")]
    sl["logic_z"] = _z(sl["logic_score"])
    sl["mmlu_z"] = _z(sl["acc_mmlu"]) if sl["acc_mmlu"].notna().any() else 0.0
    sl["logp_z"] = _z(sl["log_params"])
    sl["lc_z"] = _z(sl["logical_creativity"])
    for a in assistants:
        if f"dprime__{a}" in sl:
            sl[f"dprime_z__{a}"] = _z(sl[f"dprime__{a}"])
    # Decision 17 (v0.2.1): "thinking" (always-reasoning model) is a control when the
    # roster has both kinds; with only one kind it is constant and left out.
    covs = COVS + (["thinking"] if "thinking" in sl and 0 < sl["thinking"].fillna(0).mean() < 1 else [])
    xs = ["logic_z"] + covs

    res: dict[str, Any] = {"unit": "solver = model x reasoning mode", "n_solvers": int(len(sl)),
                           "n_solver_units_total": int(len(sl_all)),
                           "n_excluded": int(len(sl_all) - len(sl)),
                           "n_models": int(sl["solver"].nunique()), "assistants": assistants,
                           "primary_assistant": assistants[0] if assistants else None,
                           "equivalence_bound_pp": bound, "alpha": alpha,
                           "primary_outcome": primary}

    def block(a: str, s: int) -> dict[str, Any]:
        b: dict[str, Any] = {}
        # Outcomes in percentage points. The ANCOVA columns (score with the assistant, adjusted
        # for the same solver's solo score on the same items) are kept for the sensitivity analyses.
        sl[f"assisted_hard_error_pp__{a}"] = 100 * sl[f"assisted_hard_error__{a}"]
        sl[f"alone_hard_error_z__{a}"] = _z(sl[f"alone_hard_error__{a}"])
        sl[f"assisted_easy_pp__{a}"] = 100 * sl[f"assisted_easy_math__{a}"]
        sl[f"alone_easy_z__{a}"] = _z(sl[f"alone_easy_math_common__{a}"])
        x_h = xs + [f"alone_hard_error_z__{a}"]
        x_e = xs + [f"alone_easy_z__{a}"]
        # Decision 14 (2026-11-05): the gain score (with AI minus alone, same items) is the
        # primary outcome. Pipeline validation showed the adjusted (ANCOVA) score gives
        # false positives, because the solo score is measured with noise and overlaps
        # with logic. The ANCOVA versions stay as labelled sensitivity analyses.
        gain = primary == "gain"
        y_h, xh = (f"gain_hard_error__{a}", xs) if gain else (f"assisted_hard_error_pp__{a}", x_h)
        y_e, xe = (f"gain_easy_math__{a}", xs) if gain else (f"assisted_easy_pp__{a}", x_e)
        b["H1"] = regress(sl, y_h, xh, test="logic_z", null=0.0,
                          alternative="greater", reps=reps, seed=s + 1)
        h2 = regress(sl, y_e, xe, test="logic_z", null=bound,
                     alternative="less", reps=reps, seed=s + 2)
        lower = regress(sl, y_e, xe, test="logic_z", null=-bound,
                        alternative="greater", reps=reps, seed=s + 3)
        h2["tost_lower"] = lower.get("test")
        if "test" in h2 and lower.get("test"):
            h2["equivalent_within_bound"] = bool(max(h2["test"]["p_wild_cluster"],
                                                     lower["test"]["p_wild_cluster"]) < alpha)
        b["H2"] = h2
        b["H4"] = mediation(sl, "logic_z", f"dprime_z__{a}", y_h, xh[1:],
                            "solver", reps=min(reps, 4999), seed=s + 4)
        # ---- registered secondary analyses (not in the Holm family)
        alt_h, alt_xh = (f"assisted_hard_error_pp__{a}", x_h) if gain else (f"gain_hard_error__{a}", xs)
        alt_e, alt_xe = (f"assisted_easy_pp__{a}", x_e) if gain else (f"gain_easy_math__{a}", xs)
        alt_name = "ancova" if gain else "gain_score"
        b[f"S1_{alt_name}_hard_error"] = regress(sl, alt_h, alt_xh, test="logic_z",
                                                 alternative="greater", reps=reps, seed=s + 5)
        b[f"S2_{alt_name}_easy"] = regress(sl, alt_e, alt_xe, test="logic_z", null=bound,
                                           alternative="less", reps=reps, seed=s + 6)
        if gain:
            for k in ("S1_ancova_hard_error", "S2_ancova_easy"):
                b[k]["note"] = ("Sensitivity only. Biased upward when the solo score is noisy "
                                "(see validation/validation.json).")
        sl[f"contrast__{a}"] = sl[f"gain_hard_error__{a}"] - sl[f"gain_easy_math__{a}"]
        b["S3_contrast_hard_vs_easy"] = regress(sl, f"contrast__{a}", xs, test="logic_z",
                                                alternative="greater", reps=reps, seed=s + 7)
        b["S4_gain_hard_all_items"] = regress(sl, f"gain_hard_deduction__{a}", xs)
        b["S5_gain_hard_clean_items"] = regress(sl, f"gain_hard_clean__{a}", xs)
        # within-model: reasoning mode shifts logic inside the same model
        wm = sl.copy()
        for col in (f"gain_hard_error__{a}", f"gain_easy_math__{a}", "logic_z"):
            wm[col + "_wm"] = wm[col] - wm.groupby("solver")[col].transform("mean")
        b["S6_within_model_hard_error"] = regress(wm, f"gain_hard_error__{a}_wm", ["logic_z_wm"],
                                                  test="logic_z_wm", alternative="greater",
                                                  reps=reps, seed=s + 10)
        b["S7_within_model_easy"] = regress(wm, f"gain_easy_math__{a}_wm", ["logic_z_wm"])
        b["S8_behavioural_catch"] = regress(sl, f"behav_catch__{a}", xs)
        # ---- exploratory
        b["E1_H3_logical_creativity_originality"] = regress(
            sl, f"gain_cr_orig__{a}", ["lc_z", "logic_z"] + covs, test="lc_z",
            alternative="greater", reps=min(reps, 1999), seed=s + 8)
        b["E2_H3_logical_creativity_valid"] = regress(sl, f"gain_cr_valid__{a}", ["lc_z", "logic_z"] + covs)
        return b

    res["by_assistant"] = {a: block(a, seed + 100 * i) for i, a in enumerate(assistants)}

    # ---- Holm across the confirmatory family, primary assistant
    if assistants:
        prim = res["by_assistant"][assistants[0]]
        raw = {"H1": prim["H1"].get("test", {}).get("p_wild_cluster", 1.0),
               "H2": prim["H2"].get("test", {}).get("p_wild_cluster", 1.0),
               "H4": prim["H4"].get("p_one_sided", 1.0)}
        # H2 is a registered secondary (decision 25): kept in this table for reporting,
        # but outside the Holm family and judged on its unadjusted p-value.
        adj = holm({h: raw[h] for h in CONFIRMATORY})
        res["confirmatory"] = {
            h: {"p_raw": raw[h], "p_holm": adj.get(h),
                "role": "confirmatory" if h in CONFIRMATORY else "registered secondary",
                "supported": bool((adj[h] if h in CONFIRMATORY else raw[h]) < alpha)} for h in raw}
        res["holm_family"] = list(CONFIRMATORY)
        res["confirmatory"]["H1"]["estimate_pp_per_sd"] = prim["H1"].get("coef", {}).get("logic_z", {}).get("b")
        res["confirmatory"]["H2"]["estimate_pp_per_sd"] = prim["H2"].get("coef", {}).get("logic_z", {}).get("b")
        res["confirmatory"]["H2"]["equivalent_within_bound"] = prim["H2"].get("equivalent_within_bound")
        res["confirmatory"]["H4"]["indirect"] = prim["H4"].get("indirect")
        if len(assistants) > 1:
            rep = res["by_assistant"][assistants[1]]
            res["replication"] = {
                "assistant": assistants[1],
                "H1_p": rep["H1"].get("test", {}).get("p_wild_cluster"),
                "H2_p": rep["H2"].get("test", {}).get("p_wild_cluster"),
                "H4_p": rep["H4"].get("p_one_sided"),
                "H1_estimate": rep["H1"].get("coef", {}).get("logic_z", {}).get("b"),
                "H2_estimate": rep["H2"].get("coef", {}).get("logic_z", {}).get("b"),
            }

    # ---- known-answer gates that decide how results may be read (preregistered)
    from .checks import gate_flag_validity, gate_logic_validity
    gates = {"G6": {a: gate_flag_validity(trials, a) for a in assistants}, "G7": gate_logic_validity(sl)}
    if manifest:
        for a, g in manifest.get("gates", {}).get("G4_G5", {}).items():
            gates.setdefault("G5", {})[a] = {"pass": g["G5_pass"], "worst_auc": g["G5_worst_auc"]}
    res["gates"] = gates
    if assistants and "confirmatory" in res:
        prim = assistants[0]
        labels = {"H1": [], "H2": [], "H4": []}
        if gates["G7"].get("pass") is False:
            labels["H1"].append("logic-score validity check (G7) failed")
            labels["H2"].append("logic-score validity check (G7) failed")
        if not gates["G6"][prim].get("pass"):
            labels["H4"].append("UNINFORMATIVE: error-flag known-answer check (G6) failed")
        if gates.get("G5", {}).get(prim, {}).get("pass") is False:
            labels["H1"].append("caution: planted replies differ in surface features (G5)")
            labels["H4"].append("caution: planted replies differ in surface features (G5)")
        for h in labels:  # a test that could not run is "not testable", never "not supported"
            if "error" in res["by_assistant"][prim][h]:
                labels[h].append("UNINFORMATIVE: not testable (" + res["by_assistant"][prim][h]["error"] + ")")
        for h, lab in labels.items():
            res["confirmatory"][h]["labels"] = lab
            res["confirmatory"][h]["interpretable"] = not any(x.startswith("UNINFORMATIVE") for x in lab)

    # ---- minimum detectable effect (80% power, one-sided alpha) from the observed SE
    from scipy.stats import t as tdist
    if assistants:
        h1 = res["by_assistant"][assistants[0]]["H1"]
        if "coef" in h1:
            G = h1["clusters"]
            se = h1["coef"]["logic_z"]["se"]
            res["mde_H1_pp_per_sd"] = float((tdist.ppf(1 - alpha, G - 1) + tdist.ppf(0.8, G - 1)) * se)

    # ---- C6 / C7: recompute with reference implementations (statsmodels)
    try:
        import statsmodels.api as sm
        from statsmodels.stats.multitest import multipletests
        a0 = assistants[0]
        h1 = res["by_assistant"][a0]["H1"]
        ycol, xcols = h1["y"], h1["x"]
        d = sl[[ycol] + xcols + ["solver"]].dropna()
        fit = sm.OLS(d[ycol], sm.add_constant(d[xcols])).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d["solver"])[0]})
        mine = res["by_assistant"][a0]["H1"]["coef"]["logic_z"]
        res["_c6"] = {"max_diff": float(max(abs(fit.params["logic_z"] - mine["b"]),
                                            abs(fit.bse["logic_z"] - mine["se"]))),
                      "statsmodels_b": float(fit.params["logic_z"]), "statsmodels_se": float(fit.bse["logic_z"])}
        keys = list(CONFIRMATORY)
        ref = multipletests([res["confirmatory"][k]["p_raw"] for k in keys], method="holm")[1]
        res["_c7"] = float(max(abs(r - res["confirmatory"][k]["p_holm"]) for r, k in zip(ref, keys)))
    except Exception as e:  # reference check must never break the run
        res["_c6_error"] = str(e)

    # ---- robustness: multiverse / specification curve (exploratory)
    try:
        from .multiverse import run_multiverse
        res["multiverse"] = run_multiverse(sl, assistants, bound, alpha, out_dir, primary)
    except Exception as e:
        res["multiverse_error"] = str(e)

    # ---- measurement validity
    val = {}
    for col in ("acc_folio", "acc_logiqa", "acc_mmlu", "belief_bias_index", "log_params"):
        if col in sl and sl[col].notna().sum() > 3:
            val[f"logic_score~{col}"] = {
                "pearson": float(sl[["logic_score", col]].corr().iloc[0, 1]),
                "spearman": float(sl[["logic_score", col]].corr(method="spearman").iloc[0, 1])}
    if "acc_gsm8k" in sl and sl["acc_gsm8k"].notna().sum() > 3:
        val["alone_easy_math~acc_gsm8k"] = {"pearson": float(sl[["alone_easy_math", "acc_gsm8k"]].corr().iloc[0, 1])}
    fam_cols = [c for c in ("acc_syllogism", "acc_conditional", "acc_crt") if c in sl]
    if len(fam_cols) == 3:
        k = 3
        C = sl[fam_cols].corr().to_numpy()
        mean_r = (C.sum() - k) / (k * (k - 1))
        val["logic_battery_standardized_alpha"] = float(k * mean_r / (1 + (k - 1) * mean_r))
    res["validity"] = val

    # ---- manipulation checks + quality
    tt = trials[trials["condition"] == "assisted"]
    hd = tt[tt["family"] == "hard_deduction"]
    res["manipulation_check"] = {
        "assistant_wrong_when_planted": float(hd.loc[hd["planted"] == True, "assistant_error_present"].mean())
        if len(hd) else None,  # noqa: E712
        "assistant_wrong_when_clean": float(hd.loc[hd["planted"] == False, "assistant_error_present"].mean())
        if len(hd) else None,  # noqa: E712
    }
    res["quality"] = {
        "api_error_rate": float((trials["status"] != "ok").mean()),
        "unparsed_rate": float((~trials.loc[trials["status"] == "ok", "parsed"].astype(bool)).mean()),
        "ceiling_easy_alone_share_over_95": float((sl["alone_easy_math"] > 0.95).mean()),
        "floor_hard_alone_share_under_40": float((sl["alone_hard_deduction"] < 0.40).mean()),
    }
    desc = {}
    for c in ["logic_score", "acc_mmlu", "alone_easy_math", "alone_hard_deduction"] + \
             [f"gain_easy_math__{a}" for a in assistants] + [f"gain_hard_error__{a}" for a in assistants] + \
             [f"gain_hard_clean__{a}" for a in assistants] + [f"dprime__{a}" for a in assistants]:
        if c in sl:
            desc[c] = {"mean": float(sl[c].mean()), "sd": float(sl[c].std()),
                       "min": float(sl[c].min()), "max": float(sl[c].max())}
    res["descriptives"] = desc
    by_mode = sl.groupby("mode")[["logic_score"] + [f"gain_hard_error__{a}" for a in assistants]].mean()
    res["by_mode"] = json.loads(by_mode.to_json())
    res["excluded_units"] = sl_all.loc[sl_all.get("excluded", False) == True, "solver_unit"].tolist() \
        if "excluded" in sl_all else []  # noqa: E712

    sl.to_csv(out_dir / "solver_level_analysed.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps(_clean(res), indent=2), encoding="utf-8")
    _figures(sl, assistants, out_dir)
    return _clean(res)


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _figures(sl: pd.DataFrame, assistants: list[str], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not assistants:
        return
    a = assistants[0]
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    s1, s2 = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for col, colr, lab in ((f"gain_hard_error__{a}", s1, "Hard task, assistant has a planted error"),
                           (f"gain_easy_math__{a}", s2, "Easy task, normal assistant")):
        d = sl[["logic_score", col]].dropna()
        if len(d) < 3:
            continue
        ax.scatter(d["logic_score"], d[col], s=22, color=colr, alpha=0.8, edgecolor="white",
                   linewidth=0.8, label=lab)
        k, c0 = np.polyfit(d["logic_score"], d[col], 1)
        xs = np.linspace(d["logic_score"].min(), d["logic_score"].max(), 50)
        ax.plot(xs, k * xs + c0, color=colr, linewidth=2)
    ax.axhline(0, color=muted, linewidth=0.8)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_xlabel("Logic score (share correct on the logic battery, solving alone)")
    ax.set_ylabel("AI gain (percentage points)")
    ax.set_title(f"Who gains most from the assistant ({a})", color=ink, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_gain_vs_logic.png")
    plt.close(fig)

    if f"dprime__{a}" in sl:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        d = sl[["logic_score", f"dprime__{a}"]].dropna()
        ax.scatter(d["logic_score"], d[f"dprime__{a}"], s=22, color=s1, edgecolor="white", linewidth=0.8)
        ax.grid(axis="y", color=grid, linewidth=0.6)
        ax.set_xlabel("Logic score")
        ax.set_ylabel("Error detection d′ (higher = better at spotting AI errors)")
        ax.set_title("Mechanism: do better reasoners catch the AI's errors?", color=ink, loc="left")
        fig.tight_layout()
        fig.savefig(out_dir / "fig2_detection_vs_logic.png")
        plt.close(fig)
