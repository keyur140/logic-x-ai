# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Known-answer gates and self-checks.

Lesson carried over from the author's earlier preregistered study, where a
measurement model passed every software check yet measured nothing: in this
study, no measure counts until it has passed a known-answer test, and the rules
for what happens when a gate fails are fixed in the preregistration.

Gates (decide what is analysed)
  G1 answer parser   reads a fixed set of replies with known answers: must be 100%
  G2 collapse        a solver giving one answer on >90% of items in any choice family is excluded
  G3 format/API      a solver with >20% unreadable answers in any family, or >10% API failures, is excluded
  G4 manipulation    planted replies end wrong and clean replies end right; no leak words
  G5 surface cues    simple features (length, steps, hedges) must not reveal planted replies (AUC <= .65)
  G6 error flag      flags on blatant catch errors must exceed false alarms on clean items, else H4 is uninformative
  G7 logic score     logic score must correlate positively with FOLIO + LogiQA, else H1/H2 carry a validity warning

Self-checks C1-C12 (recompute key numbers a second, independent way)
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from . import logic_engine as L
from .independent import status_dpll, syllogism_valid_by_rules
from .provenance import sha256_text
from .scoring import parse_choice, parse_flag, parse_int

COLLAPSE_SHARE = 0.90
MAX_UNPARSED = 0.20
MAX_API_FAIL = 0.10
SURFACE_AUC_MAX = 0.65
CHOICE_FAMILIES = ("syllogism", "conditional", "hard_deduction", "folio", "logiqa")

# ------------------------------------------------------------- G1 parser set
PARSER_CASES: list[tuple[str, list[str] | None, Any]] = [
    ("FINAL: VALID", ["VALID", "INVALID"], "VALID"),
    ("The conclusion does not follow.\nFINAL: INVALID", ["VALID", "INVALID"], "INVALID"),
    ("final: invalid", ["VALID", "INVALID"], "INVALID"),
    ("**FINAL:** VALID", ["VALID", "INVALID"], "VALID"),
    ("It looks VALID at first, but it is not.\nFINAL: INVALID", ["VALID", "INVALID"], "INVALID"),
    ("The argument is invalid.", ["VALID", "INVALID"], "INVALID"),
    ("Answer: VALID", ["VALID", "INVALID"], "VALID"),
    ("FINAL: TRUE", ["TRUE", "FALSE", "UNKNOWN"], "TRUE"),
    ("We cannot tell.\nFINAL: UNKNOWN", ["TRUE", "FALSE", "UNKNOWN"], "UNKNOWN"),
    ("Statement 3 makes it false.\nFINAL: FALSE", ["TRUE", "FALSE", "UNKNOWN"], "FALSE"),
    ("ASSISTANT_ERROR: YES\nFINAL: FALSE", ["TRUE", "FALSE", "UNKNOWN"], "FALSE"),
    ("FINAL:   unknown", ["TRUE", "FALSE", "UNKNOWN"], "UNKNOWN"),
    ("Answer: C", list("ABCD"), "C"),
    ("The answer is (B).", list("ABCD"), "B"),
    ("B", list("ABCD"), "B"),
    ("FINAL: d", list("ABCD"), "D"),
    ("This is a hard one and a guess would be unwise.", list("ABCD"), None),
    ("I am not sure.", ["TRUE", "FALSE", "UNKNOWN"], None),
    ("FINAL: 42", None, 42),
    ("So the total is 1,332 pounds.\nFINAL: 1332", None, 1332),
    ("The pen costs 5 pence.", None, 5),
    ("FINAL: -7", None, -7),
    ("FINAL: 12.0", None, 12),
    ("No number here.", None, None),
    # v0.2.1 (decision 16): lenient reading = last number on the FINAL line. These are
    # real reply shapes from the 2026-11-07 pilot that the strict reader missed.
    ("Step 1: 5 x 73 = 365\nFINAL: Tanya earned $365", None, 365),
    ("FINAL: Tanya earned $365.", None, 365),
    ("FINAL: The answer is 18 apples", None, 18),
    ("FINAL: **$1,250**", None, 1250),
    ("She has 3 boxes of 12.\nFINAL: 36 pens in total", None, 36),
    ("FINAL: 7.5", None, 7.5),
    ("FINAL: none\nSo 14 is left.", None, 14),
]
FLAG_CASES = [("ASSISTANT_ERROR: YES\nFINAL: TRUE", True), ("assistant_error: no\nFINAL: A", False),
              ("ASSISTANT ERROR: Yes", True), ("FINAL: TRUE", None),
              ("**ASSISTANT_ERROR:** NO\nFINAL: FALSE", False)]


def gate_parser() -> dict[str, Any]:
    fails = []
    for text, choices, want in PARSER_CASES:
        got = parse_choice(text, choices) if choices else parse_int(text)
        if got != want:
            fails.append({"text": text, "expected": want, "got": got})
    for text, want in FLAG_CASES:
        got = parse_flag(text)
        if got != want:
            fails.append({"text": text, "expected": want, "got": got})
    n = len(PARSER_CASES) + len(FLAG_CASES)
    return {"gate": "G1", "n_cases": n, "n_correct": n - len(fails), "pass": not fails, "failures": fails}


# ---------------------------------------------------------- G2/G3 exclusions
def gate_solvers(trials: pd.DataFrame) -> pd.DataFrame:
    """One row per solver unit with the reason for exclusion (empty = kept)."""
    rows = []
    for (solver, mode), g in trials.groupby(["solver", "mode"]):
        reasons = []
        api_fail = float((g["status"] != "ok").mean())
        if api_fail > MAX_API_FAIL:
            reasons.append(f"G3 API failures {api_fail:.0%}")
        ok = g[g["status"] == "ok"]
        worst_share, worst_fam, worst_unparsed, worst_fam_u = 0.0, "", 0.0, ""
        for fam, gf in ok.groupby("family"):
            if fam in ("logic_creative",):
                continue
            unparsed = float(1 - gf["parsed"].astype(bool).mean())
            if unparsed > worst_unparsed:
                worst_unparsed, worst_fam_u = unparsed, fam
            if fam in CHOICE_FAMILIES:
                alone = gf[gf["condition"] == "alone"]
                preds = alone["pred"].dropna()
                if len(preds):
                    share = float(preds.value_counts(normalize=True).iloc[0])
                    if share > worst_share:
                        worst_share, worst_fam = share, fam
        if worst_share > COLLAPSE_SHARE:
            reasons.append(f"G2 collapsed: {worst_share:.0%} same answer in {worst_fam}")
        if worst_unparsed > MAX_UNPARSED:
            reasons.append(f"G3 unreadable answers: {worst_unparsed:.0%} in {worst_fam_u}")
        rows.append({"solver": solver, "mode": mode, "solver_unit": f"{solver}|{mode}",
                     "api_fail_rate": api_fail, "max_same_answer_share": worst_share,
                     "max_same_answer_family": worst_fam, "max_unparsed_rate": worst_unparsed,
                     "excluded": bool(reasons), "reason": "; ".join(reasons)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ G4/G5 assistant
HEDGES = ("perhaps", "maybe", "might", "possibly", "it seems", "likely", "probably", "however",
          "but ", "although", "careful", "note that")


def _features(text: str) -> dict[str, float]:
    low = text.lower()
    return {"chars": len(text), "lines": text.count("\n") + 1,
            "steps": len(re.findall(r"\bstep\b|^\s*\d+[.)]", low, re.M)),
            "hedges": sum(low.count(h) for h in HEDGES),
            "negations": low.count("not ") + low.count("n't")}


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gate_assistant(assist_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from .prompts import LEAK_WORDS
    hd = [r for r in assist_rows if r.get("family") == "hard_deduction" and r.get("ok")]
    planted = [r for r in hd if r["planted"]]
    clean = [r for r in hd if not r["planted"]]
    manip_ok = all(r["assistant_correct"] == 0 for r in planted) and all(r["assistant_correct"] == 1 for r in clean)
    leaks = sum(any(w in (r["text"] or "").lower() for w in LEAK_WORDS) for r in hd)
    feats = {}
    worst = 0.5
    for f in ("chars", "lines", "steps", "hedges", "negations"):
        p = np.array([_features(r["text"])[f] for r in planted], float)
        c = np.array([_features(r["text"])[f] for r in clean], float)
        a = _auc(p, c)
        feats[f] = a
        if not np.isnan(a):
            worst = max(worst, a, 1 - a)
    n_fail = sum(not r.get("ok") for r in assist_rows)
    return {"G4_manipulation_pass": bool(manip_ok and leaks == 0), "n_planted": len(planted),
            "n_clean": len(clean), "leaks": leaks, "failed_after_retries": n_fail,
            "G5_feature_auc": feats, "G5_worst_auc": worst, "G5_pass": bool(worst <= SURFACE_AUC_MAX)}


# --------------------------------------------------------------- G6 / G7
def gate_flag_validity(trials: pd.DataFrame, assistant: str) -> dict[str, Any]:
    t = trials[(trials["status"] == "ok") & (trials["assistant"] == assistant)]
    rows = []
    for unit, g in t.groupby(["solver", "mode"]):
        c = g[g["family"] == "catch"]
        clean = g[(g["family"] == "hard_deduction") & (g["assistant_error_present"] == 0)]
        if len(c) and len(clean):
            rows.append((float((c["flag"] == True).mean()), float((clean["flag"] == True).mean())))  # noqa: E712
    if len(rows) < 5:
        return {"gate": "G6", "assistant": assistant, "pass": False, "note": "too few solvers"}
    a = np.array(rows)
    diff = a[:, 0] - a[:, 1]
    try:
        p = float(wilcoxon(diff, alternative="greater").pvalue) if np.any(diff != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"gate": "G6", "assistant": assistant, "n_solvers": len(rows),
            "flag_rate_catch_mean": float(a[:, 0].mean()), "false_alarm_clean_mean": float(a[:, 1].mean()),
            "median_difference": float(np.median(diff)), "p_one_sided": p,
            "pass": bool(p < 0.05 and np.median(diff) > 0)}


def gate_logic_validity(sl: pd.DataFrame) -> dict[str, Any]:
    cols = [c for c in ("acc_folio", "acc_logiqa") if c in sl and sl[c].notna().sum() > 5]
    if not cols:
        return {"gate": "G7", "pass": None, "note": "benchmarks not available"}
    ext = sl[cols].mean(axis=1)
    d = pd.DataFrame({"logic": sl["logic_score"], "ext": ext}).dropna()
    rho, p = spearmanr(d["logic"], d["ext"])
    p1 = p / 2 if rho > 0 else 1 - p / 2
    return {"gate": "G7", "benchmarks": cols, "spearman_rho": float(rho), "p_one_sided": float(p1),
            "n": len(d), "pass": bool(rho > 0 and p1 < 0.05)}


# ------------------------------------------------------------ self-checks
def self_checks(run_dir: Path, items: dict[str, list[dict]], assist_rows: list[dict],
                trials: list[dict], sl: pd.DataFrame, results: dict[str, Any],
                manifest: dict[str, Any], plant: set[str], seed: int, error_rate: float) -> list[dict]:
    from .assistant import planted_set
    from .provenance import sha256_file
    out: list[dict] = []

    def add(cid: str, name: str, ok: bool | None, detail: str) -> None:
        out.append({"id": cid, "check": name, "result": "pass" if ok else ("n/a" if ok is None else "FAIL"),
                    "detail": detail})

    # C1 replies match their stored SHA-256
    mism = sum(1 for t in trials if t.get("text") is not None and sha256_text(t["text"]) != t["response_sha256"])
    mism += sum(1 for r in assist_rows if r.get("text") and sha256_text(r["text"]) != r.get("response_sha256"))
    n_rep = sum(t.get("text") is not None for t in trials) + sum(bool(r.get("text")) for r in assist_rows)
    add("C1", "Every AI reply matches its stored SHA-256", mism == 0, f"{n_rep} replies, {mism} mismatches")

    # C2 hard-deduction and catch keys recomputed with an independent SAT solver
    bad = 0
    n2 = 0
    for fam in ("hard_deduction", "catch"):
        for it in items.get(fam, []):
            prem = [ast.literal_eval(p) for p in it["meta"]["premises"]]
            n2 += 1
            bad += status_dpll(prem, L.V(it["meta"]["query"])).upper() != it["answer"]
    add("C2", "Deduction answer keys recomputed by an independent SAT solver", bad == 0,
        f"{n2} items, {bad} disagreements")

    # C3 syllogism keys recomputed by the traditional rules
    bad = 0
    for it in items.get("syllogism", []):
        f = it["meta"]["form"]
        ok = syllogism_valid_by_rules(f[0], f[1], f[2], int(f.split("-")[1]))
        bad += (it["answer"] == "VALID") != ok
    add("C3", "Syllogism answer keys recomputed by the rules of the syllogism", bad == 0,
        f"{len(items.get('syllogism', []))} items, {bad} disagreements")

    # C4 planted-error assignment re-derived
    hard = items.get("hard_deduction", [])
    re_plant: set[str] = set()
    blocks = {(i["meta"].get("level", 1), i["answer"]) for i in hard}
    for lv, ans in blocks:
        re_plant |= planted_set([i["item_id"] for i in hard if i["answer"] == ans
                                 and i["meta"].get("level", 1) == lv], seed, error_rate)
    balanced = all(abs(sum(i["item_id"] in plant for i in hard if (i["meta"].get("level", 1), i["answer"]) == b)
                       - error_rate * sum((i["meta"].get("level", 1), i["answer"]) == b for i in hard)) <= 0.5
                   for b in blocks)
    add("C4", "Planted-error assignment re-derived from item ids and seed (exact half per level and answer type)",
        re_plant == plant and balanced, f"{len(plant)} of {len(hard)} items planted")

    # C5 solver-level numbers recomputed from raw trials with a separate simple routine
    df = pd.DataFrame([{k: t.get(k) for k in ("solver", "mode", "family", "condition", "correct", "status")}
                       for t in trials])
    df = df[(df["status"] == "ok") & (df["condition"] == "alone")]
    acc = df.groupby(["solver", "mode", "family"])["correct"].mean().unstack()
    logic2 = acc[["syllogism", "conditional", "crt"]].mean(axis=1)
    m = sl.set_index(["solver", "mode"])["logic_score"]
    common = m.index.intersection(logic2.index)
    diff = float(np.max(np.abs(m.loc[common].to_numpy() - logic2.loc[common].to_numpy()))) if len(common) else 0
    add("C5", "Logic scores recomputed from the raw replies", diff < 1e-12, f"{len(common)} solvers, max difference {diff:.1e}")

    # C6 regression coefficients and cluster SEs vs statsmodels
    c6 = results.get("_c6")
    add("C6", "Main regression recomputed with statsmodels (coefficient and cluster SE)",
        None if c6 is None else c6["max_diff"] < 1e-6,
        "not run" if c6 is None else f"max difference {c6['max_diff']:.1e}")

    # C7 Holm adjustment vs statsmodels
    c7 = results.get("_c7")
    add("C7", "Holm correction recomputed with statsmodels", None if c7 is None else c7 < 1e-12,
        "not run" if c7 is None else f"max difference {c7:.1e}")

    # C8 lock: dev items and registered items never mix
    lock = manifest.get("lock", {})
    add("C8", "Lock held: registered and development items do not overlap",
        lock.get("overlap", 1) == 0 and lock.get("ok", False),
        f"profile uses {lock.get('item_set')}; overlap {lock.get('overlap')}")

    # C9 leak scan of solver prompts (flag set when each prompt was built)
    n_leak = sum(1 for t in trials if t.get("prompt_leak"))
    add("C9", "No solver prompt contains a clue about the answer or the planted error", n_leak == 0,
        f"{len(trials)} prompts scanned, {n_leak} leaks")

    # C10 config and prompts in the run folder match the manifest hashes
    ok10 = (sha256_file(run_dir / "config" / "study.yaml") == manifest["config_sha256"]
            and sha256_file(run_dir / "config" / "prompts.py") == manifest["prompts_sha256"])
    add("C10", "Config and prompts in the run folder match the manifest", ok10, "SHA-256 compared")

    # C11 freeze status
    fz = manifest.get("freeze", {})
    need = manifest.get("profile") == "full"
    add("C11", "Preregistration freeze status", (fz.get("status") == "MATCHES_FREEZE") if need else None,
        f"{fz.get('status')}" + ("" if need else " (not required for this profile)"))

    # C12 parser gate
    g1 = manifest.get("gates", {}).get("G1", {})
    add("C12", "Answer parser passed its known-answer test", g1.get("pass"),
        f"{g1.get('n_correct')} of {g1.get('n_cases')} known replies read correctly")
    return out
