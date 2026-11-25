# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Build tidy tables from raw trials.

  scored/trials.csv        one row per trial (no reply text; text stays in raw/trials.jsonl)
  scored/solver_level.csv  one row per solver (= model x reasoning mode): every
                           preregistered score, gain and mechanism measure
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

LOGIC_FAMS = ("syllogism", "conditional", "crt")


def _dprime(hits: int, n_signal: int, fas: int, n_noise: int) -> float:
    """Signal-detection sensitivity with the log-linear correction (Hautus, 1995)."""
    if n_signal == 0 or n_noise == 0:
        return float("nan")
    h = (hits + 0.5) / (n_signal + 1)
    f = (fas + 0.5) / (n_noise + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def _belief_bias(alone: pd.DataFrame) -> float:
    syl = alone[alone["family"] == "syllogism"]
    if "belief_conflict" not in syl or syl.empty:
        return float("nan")
    conf = syl[syl["belief_conflict"] == True]["correct"].mean()  # noqa: E712
    non = syl[(syl["belief_conflict"] == False) & (syl["content"] != "neutral")]["correct"].mean()  # noqa: E712
    return float(conf - non)


def _originality(trials: list[dict[str, Any]]) -> dict[int, float]:
    """For each creative trial: mean surprisal (-log2 share) of its valid conclusions,
    where share = fraction of all creative replies to that item that contain it."""
    by_item: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(trials):
        if t["family"] == "logic_creative" and t["status"] == "ok":
            by_item[t["item_id"]].append(i)
    out: dict[int, float] = {}
    for idxs in by_item.values():
        counts: dict[str, int] = defaultdict(int)
        for i in idxs:
            for k in set(trials[i]["creative"]["valid_keys"]):
                counts[k] += 1
        n = len(idxs)
        for i in idxs:
            keys = set(trials[i]["creative"]["valid_keys"])
            out[i] = float(np.mean([-math.log2(counts[k] / n) for k in keys])) if keys else 0.0
    return out


def build_tables(trials: list[dict[str, Any]], solvers: list[dict], out_dir: str | Path,
                 excluded: set[str] | None = None) -> dict[str, Path]:
    """solver_level.csv keeps every solver unit, with an `excluded` column set by the
    preregistered gates G2/G3; the analysis uses only units with excluded == False."""
    excluded = excluded or set()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    orig = _originality(trials)
    flat = []
    for i, t in enumerate(trials):
        row = {k: v for k, v in t.items() if k not in ("text", "creative")}
        if t.get("creative"):
            c = t["creative"]
            row.update({"cr_valid": c["n_valid_distinct"], "cr_parsed": c["n_parsed"],
                        "cr_invalid": c["n_invalid"], "cr_originality": orig.get(i, 0.0)})
        flat.append(row)
    df = pd.DataFrame(flat)
    p_trials = out_dir / "trials.csv"
    df.to_csv(p_trials, index=False)

    ok = df[df["status"] == "ok"].copy()
    params = {s["alias"]: s.get("params_b") for s in solvers}
    fams = {s["alias"]: s.get("family") for s in solvers}
    assistants = sorted(a for a in ok["assistant"].dropna().unique())
    synthetic_kind = {s["alias"]: s.get("synthetic_kind", "") for s in solvers}
    thinking = {s["alias"]: int(bool(s.get("thinking"))) for s in solvers}
    rows = []
    for (solver, mode), g in ok.groupby(["solver", "mode"]):
        r: dict[str, Any] = {"solver": solver, "mode": mode, "solver_unit": f"{solver}|{mode}",
                             "synthetic_kind": synthetic_kind.get(solver, ""),
                             "model_family": fams.get(solver), "params_b": params.get(solver),
                             "thinking": thinking.get(solver, 0),
                             "log_params": math.log10(params[solver]) if params.get(solver) else np.nan}
        alone = g[g["condition"] == "alone"]
        # ---- logic battery (moderator): mean of family accuracies, equal weights
        fam_acc = {f: alone.loc[alone["family"] == f, "correct"].mean() for f in LOGIC_FAMS}
        for f, v in fam_acc.items():
            r[f"acc_{f}"] = v
        r["logic_score"] = float(np.nanmean(list(fam_acc.values())))
        # belief-bias resistance: accuracy on conflict minus non-conflict syllogisms
        r["belief_bias_index"] = _belief_bias(alone)
        for f in ("folio", "logiqa", "gsm8k"):
            r[f"acc_{f}"] = alone.loc[alone["family"] == f, "correct"].mean()
        # knowledge control is per model (run once, in direct mode)
        mm = ok[(ok["solver"] == solver) & (ok["family"] == "mmlu")]
        r["acc_mmlu"] = mm["correct"].mean() if len(mm) else np.nan
        # ---- tasks
        for fam in ("easy_math", "hard_deduction"):
            a_rows = alone[alone["family"] == fam].set_index("item_id")["correct"]
            r[f"alone_{fam}"] = a_rows.mean()
            for ast in assistants:
                s_rows = g[(g["family"] == fam) & (g["assistant"] == ast)]
                if not len(s_rows):
                    continue
                paired = s_rows.set_index("item_id")
                common = paired.index.intersection(a_rows.index)
                r[f"assisted_{fam}__{ast}"] = paired.loc[common, "correct"].mean()
                r[f"gain_{fam}__{ast}"] = 100 * (paired.loc[common, "correct"].mean() - a_rows.loc[common].mean())
                r[f"alone_{fam}_common__{ast}"] = a_rows.loc[common].mean()
                if fam == "hard_deduction":
                    for cond, sub in (("error", paired[paired["planted"]]), ("clean", paired[~paired["planted"]])):
                        cc = sub.index.intersection(a_rows.index)
                        r[f"gain_hard_{cond}__{ast}"] = 100 * (sub.loc[cc, "correct"].mean() - a_rows.loc[cc].mean()) if len(cc) else np.nan
                        r[f"assisted_hard_{cond}__{ast}"] = sub.loc[cc, "correct"].mean() if len(cc) else np.nan
                        r[f"alone_hard_{cond}__{ast}"] = a_rows.loc[cc].mean() if len(cc) else np.nan
                    sig = paired[paired["assistant_error_present"] == 1]
                    noi = paired[paired["assistant_error_present"] == 0]
                    hits = int((sig["flag"] == True).sum())  # noqa: E712
                    fas = int((noi["flag"] == True).sum())  # noqa: E712
                    r[f"hit_rate__{ast}"] = hits / len(sig) if len(sig) else np.nan
                    r[f"fa_rate__{ast}"] = fas / len(noi) if len(noi) else np.nan
                    r[f"dprime__{ast}"] = _dprime(hits, len(sig), fas, len(noi))
                    r[f"flag_missing__{ast}"] = float(paired["flag"].isna().mean())
                    # behavioural catch: did not copy the wrong assistant answer
                    if len(sig) and "pred" in sig:
                        r[f"behav_catch__{ast}"] = float((sig["correct"] == 1).mean())
        # ---- logical creativity (exploratory H3)
        cr = g[g["family"] == "logic_creative"]
        cra = cr[cr["condition"] == "alone"]
        # validity x originality: total surprisal of the valid conclusions, averaged over items
        r["logical_creativity"] = float((cra["cr_valid"] * cra["cr_originality"]).mean()) if len(cra) else np.nan
        r["cr_valid_alone"] = cra["cr_valid"].mean() if len(cra) else np.nan
        r["cr_orig_alone"] = cra["cr_originality"].mean() if len(cra) else np.nan
        for ast in assistants:
            crs = cr[cr["assistant"] == ast]
            if len(crs) and len(cra):
                r[f"gain_cr_orig__{ast}"] = crs["cr_originality"].mean() - cra["cr_originality"].mean()
                r[f"gain_cr_valid__{ast}"] = crs["cr_valid"].mean() - cra["cr_valid"].mean()
        for ast in assistants:
            cg = g[(g["family"] == "catch") & (g["assistant"] == ast)]
            if len(cg):
                r[f"catch_flag_rate__{ast}"] = float((cg["flag"] == True).mean())  # noqa: E712
        r["excluded"] = f"{solver}|{mode}" in excluded
        r["n_trials"] = len(g)
        r["unparsed_rate"] = float(1 - g["parsed"].mean())
        rows.append(r)
    sl = pd.DataFrame(rows).sort_values(["solver", "mode"])
    p_sl = out_dir / "solver_level.csv"
    sl.to_csv(p_sl, index=False)
    return {"trials": p_trials, "solver_level": p_sl}
