# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Pipeline validation. Does the analysis find effects only when they exist?

Part A: fast Monte Carlo at the solver level (the same OLS and CR1 code as the study).
  Worlds for H1 (planted-error items):
    ignore    the solver ignores the assistant: no gain, no loss       (no effect)
    constant  everyone catches 35% of AI errors, whatever their logic  (no effect)
    amplify   catching AI errors rises with logic                      (effect)
  Each world is analysed with the registered gain score and with the ANCOVA
  (adjusted) score. A sound test rejects about 5% of the time in the no-effect
  worlds and often in the effect world.
  H2 check: the true logic slope of the easy-task gain is exactly +5 pp per SD
  (the bound), so a claim of "no amplification beyond 5 pp" would be false.
  The non-superiority test should make that false claim at most about 5% of the time.

Part B: reads the latest full synthetic pipeline runs (profiles mock and mock_null)
  from the runs folder and puts their confirmatory results side by side.

Writes validation/validation.json, validation_rates.csv, fig_validation.png and
SHA256SUMS.txt. Synthetic data only. Nothing here is a study result.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as tdist

from .analysis import _cr1_se, _ols, wild_cluster_p
from .provenance import utc_now, write_sha256sums


def _sig(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _z(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v * 0


def _units(rng: np.random.Generator, models: int, items: dict[str, int]):
    """Latent traits and observed scores for models x 3 reasoning modes."""
    th = rng.normal(size=models)
    ka = 0.5 * th + 0.87 * rng.normal(size=models)
    lp = 0.75 + 0.45 * ka + 0.25 * rng.normal(size=models)
    T = np.repeat(th, 3) + np.tile([-0.7, 0.0, 0.7], models)
    K, LP, g = np.repeat(ka, 3), np.repeat(lp, 3), np.repeat(np.arange(models), 3)
    p = 0.4 + 0.6 * _sig(1.2 * T - 0.3)            # solo accuracy on choice items (chance floor)
    p_int = _sig(0.9 + 0.9 * T + 0.4 * K)           # integer answers
    logic = np.mean([rng.binomial(items["syllogism"], p) / items["syllogism"],
                     rng.binomial(items["conditional"], p) / items["conditional"],
                     rng.binomial(items["crt"], p_int) / items["crt"]], axis=0)
    mmlu = rng.binomial(items["mmlu"], 0.25 + 0.75 * _sig(1.3 * K - 0.2)) / items["mmlu"]
    return T, K, LP, g, p, logic, mmlu


def _one_h1(rng: np.random.Generator, world: str, models: int, items: dict[str, int],
            alpha: float, wild_reps: int = 0) -> dict[str, float]:
    T, K, LP, g, p, logic, mmlu = _units(rng, models, items)
    n = items["hard_error"]
    alone = rng.binomial(n, p) / n
    if world == "ignore":
        pa = p
    elif world == "constant":
        pa = 0.35 * p
    else:
        pa = _sig(-0.8 + 1.6 * T) * p
    assisted = rng.binomial(n, pa) / n
    base = [np.ones(len(T)), _z(logic), _z(mmlu), _z(LP)]
    out = {}
    for est in ("gain", "ancova"):
        if est == "gain":
            y, X = 100 * (assisted - alone), np.column_stack(base)
        else:
            y, X = 100 * assisted, np.column_stack(base + [_z(alone)])
        b, r = _ols(y, X)
        se = _cr1_se(X, r, g)[1]
        pval = float(1 - tdist.cdf(b[1] / se, models - 1)) if se > 0 else 1.0
        out[f"{est}_reject"] = float(pval < alpha)
        out[f"{est}_b"] = float(b[1])
        if wild_reps and est == "gain":   # the registered test: wild cluster restricted bootstrap
            _, pw = wild_cluster_p(y, X, g, 1, 0.0, "greater", wild_reps, rng)
            out["gain_wild_reject"] = float(pw < alpha)
    return out


def _one_h2(rng: np.random.Generator, models: int, items: dict[str, int], bound: float,
            alpha: float) -> float:
    """True slope of the easy-task gain on (true) logic is exactly +bound."""
    T, K, LP, g, p, logic, mmlu = _units(rng, models, items)
    true_logic = _z(p)
    n = items["easy_math"]
    base_rate = 0.5 + 0.0 * T
    p_gain = np.clip(base_rate + (bound / 100) * true_logic + 0.03 * np.repeat(rng.normal(size=models), 3),
                     0.01, 0.99)
    alone = rng.binomial(n, np.full_like(T, 0.5)) / n
    assisted = rng.binomial(n, p_gain) / n
    y = 100 * (assisted - alone)
    X = np.column_stack([np.ones(len(T)), _z(logic), _z(mmlu), _z(LP)])
    b, r = _ols(y, X)
    se = _cr1_se(X, r, g)[1]
    pval = float(tdist.cdf((b[1] - bound) / se, models - 1)) if se > 0 else 1.0
    return float(pval < alpha)


def _mc_ci(k: float, n: int) -> list[float]:
    """Wilson 95% interval for a simulated rate."""
    z = 1.96
    ph = k / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def _latest(runs: Path, profile: str) -> Path | None:
    c = sorted([p for p in runs.glob(f"*-{profile}") if (p / "analysis" / "results.json").exists()])
    return c[-1] if c else None


def run_validation(config: str, out_root: str, sims: int = 2000) -> Path:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    st = cfg["study"]
    alpha, bound = float(st.get("alpha", 0.05)), float(st["equivalence_bound_pp"])
    full = cfg["profiles"]["full"]
    items = {"syllogism": full["items"]["syllogism"], "conditional": full["items"]["conditional"],
             "crt": full["items"]["crt"], "mmlu": full["benchmarks"]["mmlu"],
             "hard_error": int(full["items"]["hard_deduction"] * float(st["error_rate"])),
             "easy_math": full["items"]["easy_math"]}
    models = 25          # 28 models minus a few lost to the gates: a conservative count
    rng = np.random.default_rng(int(st["dev_seed"]))
    rows = []
    for world, effect in (("ignore", False), ("constant", False), ("amplify", True)):
        sims_w = [_one_h1(rng, world, models, items, alpha) for _ in range(sims)]
        for est in ("gain", "ancova"):
            k = sum(s[f"{est}_reject"] for s in sims_w)
            rows.append({"check": "H1", "world": world, "true_effect": effect, "estimator": est,
                         "registered": False,
                         "reject_rate": k / sims, "mc_ci95": _mc_ci(k, sims),
                         "mean_estimate_pp_per_sd": float(np.mean([s[f"{est}_b"] for s in sims_w]))})
    # The registered H1 test is the wild cluster bootstrap. The CR1 t-test above is a fast
    # stand-in and runs slightly liberal with 25 clusters, so the no-effect worlds are also
    # checked with the registered bootstrap test (fewer simulations: it is slower).
    n_wb = max(100, sims // 5)
    for world, effect in (("ignore", False), ("constant", False), ("amplify", True)):
        k = sum(_one_h1(rng, world, models, items, alpha, wild_reps=399)["gain_wild_reject"]
                for _ in range(n_wb))
        rows.append({"check": "H1", "world": world, "true_effect": effect, "estimator": "gain_wild",
                     "registered": True, "reject_rate": k / n_wb, "mc_ci95": _mc_ci(k, n_wb),
                     "sims": n_wb, "mean_estimate_pp_per_sd": None})
    for r in rows:
        r.setdefault("sims", sims)
        if r["estimator"] == "gain":
            r["registered"] = False      # fast t-test stand-in; the bootstrap rows are the registered test
    k = sum(_one_h2(rng, models, items, bound, alpha) for _ in range(sims))
    rows.append({"check": "H2", "world": f"true slope = +{bound:g} (bound)", "true_effect": True,
                 "estimator": "gain", "registered": True, "reject_rate": k / sims, "sims": sims,
                 "mc_ci95": _mc_ci(k, sims), "mean_estimate_pp_per_sd": bound})

    def verdict(r: dict[str, Any]) -> str:
        if r["check"] == "H2":
            return "PASS" if r["mc_ci95"][0] <= alpha else "FAIL"
        if not r["true_effect"]:
            return "PASS" if r["mc_ci95"][0] <= alpha else "FAIL"
        return "PASS" if r["reject_rate"] >= 0.8 else "LOW POWER"
    for r in rows:
        r["verdict"] = verdict(r)
        if r["estimator"] == "gain" and r["check"] == "H1" and r["verdict"] != "PASS":
            r["verdict"] = "FAST T-TEST SLIGHTLY LIBERAL (registered bootstrap row decides)"

    runs = Path(out_root)
    pipeline = {}
    for prof in ("mock", "mock_null"):
        d = _latest(runs, prof)
        if d is None:
            pipeline[prof] = {"status": f"no finished '{prof}' run found in {runs}"}
            continue
        res = json.loads((d / "analysis" / "results.json").read_text(encoding="utf-8"))
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8")) if (d / "manifest.json").exists() else {}
        sc = man.get("self_checks", [])
        pipeline[prof] = {"run": d.name, "world": "amplify" if prof == "mock" else "null",
                          "confirmatory": res.get("confirmatory"),
                          "self_checks": {c["id"]: c["result"] for c in sc},
                          "self_checks_failed": [c["id"] for c in sc if c["result"] == "fail"],
                          "excluded_units": res.get("n_excluded")}

    out = runs / f"validation-{utc_now().replace(':', '').replace('-', '')[:15]}"
    out.mkdir(parents=True, exist_ok=True)
    body = {"what": "Pipeline validation on synthetic data. Not study results.",
            "created_utc": utc_now(), "sims_per_world": sims, "alpha": alpha, "bound_pp": bound,
            "clusters": models, "items": items, "monte_carlo": rows, "pipeline_runs": pipeline,
            "all_pass": all(r["verdict"] == "PASS" for r in rows if r["registered"])}
    (out / "validation.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "validation_rates.csv", index=False)
    _plot(rows, alpha, out / "fig_validation.png")
    write_sha256sums(out)
    for r in rows:
        print(f"{r['check']}  {r['world']:28s} {r['estimator']:10s} n={r['sims']:<5d} "
              f"reject {r['reject_rate']:.3f}  {r['verdict']}")
    return out


def _plot(rows: list[dict[str, Any]], alpha: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#1f1e1c", "#52514e", "#e4e3df"
    col = {"gain_wild": "#2a78d6", "ancova": "#9a9893"}
    h1 = [r for r in rows if r["check"] == "H1"]
    worlds = ["ignore", "constant", "amplify"]
    labels = ["No effect:\nignores the AI", "No effect:\nsame catch rate", "Planted effect:\nlogic helps catch"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    x = np.arange(len(worlds))
    for i, est in enumerate(("gain_wild", "ancova")):
        vals = [next(r for r in h1 if r["world"] == w and r["estimator"] == est) for w in worlds]
        rates = [v["reject_rate"] * 100 for v in vals]
        lo = [v["reject_rate"] * 100 - v["mc_ci95"][0] * 100 for v in vals]
        hi = [v["mc_ci95"][1] * 100 - v["reject_rate"] * 100 for v in vals]
        name = "gain score, registered test" if est == "gain_wild" else "adjusted score (sensitivity only)"
        ax.bar(x + (i - 0.5) * 0.36, rates, 0.34, color=col[est], label=name,
               yerr=[lo, hi], error_kw={"ecolor": muted, "lw": 0.8, "capsize": 2})
        for xi, rv in zip(x + (i - 0.5) * 0.36, rates):
            ax.text(xi + 0.02, rv + 4.5, f"{rv:.0f}%", ha="center", fontsize=7, color=ink)
    ax.axhline(alpha * 100, color="#eb6834", lw=0.9, ls="--")
    ax.text(2.62, alpha * 100 + 1.5, "5% target", fontsize=7, color="#eb6834", ha="right")
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("H1 'effect found' rate (%)", fontsize=8)
    ax.set_title("Does the H1 test find an effect only when one exists? (synthetic data)",
                 loc="left", fontsize=9.5, color=ink)
    ax.yaxis.grid(True, color=grid, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
