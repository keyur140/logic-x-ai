# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Multiverse (specification curve) for H1 and H2. EXPLORATORY robustness check.

Re-runs the H1 and H2 models under every combination of defensible choices:
  outcome     gain score (registered) | ANCOVA score with the AI
  logic score battery | battery + FOLIO + LogiQA | syllogisms only
  covariates  none | knowledge | knowledge + size
  modes       all three | without "direct"
  assistant   each assistant
Inference here uses cluster-robust t-tests (fast), not the bootstrap.
A result counts as robust when most specifications agree with the registered one.
ANCOVA specifications are known to be biased upward when the solo score is noisy
(see validation); they are kept to show that, not to vote on robustness.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as tdist


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0


def _fit(d: pd.DataFrame, y: str, xs: list[str]) -> tuple[float, float, int]:
    from .analysis import _cr1_se, _ols
    d = d[[y] + xs + ["solver"]].dropna()
    X = np.column_stack([np.ones(len(d))] + [d[x].to_numpy(float) for x in xs])
    b, r = _ols(d[y].to_numpy(float), X)
    se = _cr1_se(X, r, d["solver"].to_numpy())
    return float(b[1]), float(se[1]), int(d["solver"].nunique())


def run_multiverse(sl: pd.DataFrame, assistants: list[str], bound: float, alpha: float,
                   out_dir: Path, primary: str = "gain") -> dict[str, Any]:
    logic_defs = {"battery": ["acc_syllogism", "acc_conditional", "acc_crt"],
                  "battery+benchmarks": ["acc_syllogism", "acc_conditional", "acc_crt", "acc_folio", "acc_logiqa"],
                  "syllogisms": ["acc_syllogism"]}
    cov_sets = {"none": [], "knowledge": ["mmlu_z"], "knowledge+size": ["mmlu_z", "logp_z"]}
    mode_sets = {"all": ["direct", "brief", "careful"], "no-direct": ["brief", "careful"]}
    rows = []
    for a, (ld, cols), (cn, covs), (mn, modes), outcome in itertools.product(
            assistants, logic_defs.items(), cov_sets.items(), mode_sets.items(), ("gain", "ancova")):
        d = sl[sl["mode"].isin(modes)].copy()
        cols_ok = [c for c in cols if c in d and d[c].notna().any()]
        if not cols_ok:
            continue
        d["lz"] = _z(d[cols_ok].mean(axis=1))
        for hyp in ("H1", "H2"):
            if hyp == "H1":
                y = f"assisted_hard_error_pp__{a}" if outcome == "ancova" else f"gain_hard_error__{a}"
                base = [f"alone_hard_error_z__{a}"] if outcome == "ancova" else []
            else:
                y = f"assisted_easy_pp__{a}" if outcome == "ancova" else f"gain_easy_math__{a}"
                base = [f"alone_easy_z__{a}"] if outcome == "ancova" else []
            if y not in d:
                continue
            b, se, G = _fit(d, y, ["lz"] + covs + base)
            q = tdist.ppf(1 - alpha, max(G - 1, 1))
            if hyp == "H1":
                p = float(1 - tdist.cdf(b / se, G - 1)) if se > 0 else 1.0
                ok = b > 0 and p < alpha
            else:  # non-superiority against +bound
                p = float(tdist.cdf((b - bound) / se, G - 1)) if se > 0 else 1.0
                ok = p < alpha
            rows.append({"hypothesis": hyp, "assistant": a, "logic": ld, "covariates": cn, "modes": mn,
                         "outcome": outcome, "estimate": b, "se": se, "ci90_low": b - q * se,
                         "ci90_high": b + q * se, "p_one_sided": p, "agrees": bool(ok),
                         "registered": (ld == "battery" and cn == "knowledge+size" and mn == "all"
                                        and outcome == primary and a == assistants[0])})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "multiverse.csv", index=False)
    summary = {}
    for hyp, g in df.groupby("hypothesis"):
        summary[hyp] = {"n_specifications": int(len(g)), "share_supporting": float(g["agrees"].mean()),
                        "share_positive_estimate": float((g["estimate"] > 0).mean()),
                        "median_estimate": float(g["estimate"].median())}
    _plot(df, out_dir)
    return summary


def _plot(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    muted, grid, s1, s2 = "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150)
    for ax, hyp, title in ((axes[0], "H1", "H1 amplification (planted-error items)"),
                           (axes[1], "H2", "H2 levelling (easy items)")):
        g = df[df["hypothesis"] == hyp].sort_values("estimate").reset_index(drop=True)
        if g.empty:
            continue
        x = np.arange(len(g))
        colors = [s1 if a else "#9a9893" for a in g["agrees"]]
        ax.vlines(x, g["ci90_low"], g["ci90_high"], color=grid, linewidth=1)
        ax.scatter(x, g["estimate"], s=10, c=colors, zorder=3)
        reg = g[g["registered"]]
        if len(reg):
            ax.scatter(reg.index, reg["estimate"], s=60, facecolors="none", edgecolors=s2, linewidths=1.6,
                       zorder=4, label="registered specification")
        ax.axhline(0, color=muted, linewidth=0.8)
        if hyp == "H2":
            ax.axhline(5, color=s2, linewidth=0.8, linestyle="--")
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("specifications, sorted by estimate", fontsize=8)
        ax.set_ylabel("logic effect (pp per SD)", fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if len(reg):
            ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Multiverse: blue = specification agrees with the hypothesis, grey = does not",
                 fontsize=8.5, color=muted, y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_multiverse.png")
    plt.close(fig)
