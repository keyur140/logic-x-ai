# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Power simulation for the preregistration (solver-level design, clustered by model).

Simulates: 28 models x 3 reasoning modes; per-solver AI gain measured on paired
items; gain slope on logic. Reports
  * H1 power: chance the one-sided test detects a positive slope of a given size
  * H2 power: chance the 90% CI lies within +/- bound when the true slope is 0
Assumptions are written into the output so reviewers can check them.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import t as tdist

ASSUME = {"models": 28, "modes": 3, "items_easy": 200, "items_hard_error": 150,
          "sd_gain_between_models_pp": 8.0, "sd_gain_within_model_pp": 5.0,
          "logic_sd_between_models": 1.0, "mode_logic_shift_sd": 0.5,
          "bound_pp": 5.0, "alpha": 0.05, "reps": 2000}


def _one(rng, slope_pp: float, n_items: int, a: dict) -> tuple[float, float, float]:
    G, K = a["models"], a["modes"]
    model_logic = rng.normal(0, a["logic_sd_between_models"], G)
    shift = np.array([-1, 0, 1]) * a["mode_logic_shift_sd"]
    logic = (model_logic[:, None] + shift[None, :K]).ravel()
    lz = (logic - logic.mean()) / logic.std()
    model_u = np.repeat(rng.normal(0, a["sd_gain_between_models_pp"], G), K)
    base = np.clip(0.55 + 0.12 * lz, 0.05, 0.95)
    true_gain = (slope_pp * lz + model_u + rng.normal(0, a["sd_gain_within_model_pp"], G * K)) / 100
    p_ai = np.clip(base + true_gain, 0.01, 0.99)
    obs = (rng.binomial(n_items, p_ai) - rng.binomial(n_items, base)) / n_items * 100
    X = np.c_[np.ones(G * K), lz]
    beta, *_ = np.linalg.lstsq(X, obs, rcond=None)
    resid = obs - X @ beta
    groups = np.repeat(np.arange(G), K)
    bread = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[groups == g].T @ resid[groups == g], X[groups == g].T @ resid[groups == g])
               for g in range(G))
    c = (G / (G - 1)) * ((G * K - 1) / (G * K - 2))
    se = float(np.sqrt((c * bread @ meat @ bread)[1, 1]))
    return float(beta[1]), se, float(tdist.ppf(1 - a["alpha"], G - 1))


def simulate(a: dict | None = None, seed: int = 20261026) -> dict:
    a = {**ASSUME, **(a or {})}
    rng = np.random.default_rng(seed)
    out = {"assumptions": a, "H1_power": {}, "H2_power_true_zero": None}
    for slope in (2.0, 3.0, 5.0, 8.0):
        hits = 0
        for _ in range(a["reps"]):
            b, se, q = _one(rng, slope, a["items_hard_error"], a)
            hits += (b / se) > q
        out["H1_power"][f"slope_{slope}pp_per_sd"] = hits / a["reps"]
    eq = 0
    for _ in range(a["reps"]):
        b, se, q = _one(rng, 0.0, a["items_easy"], a)
        eq += (b - q * se > -a["bound_pp"]) and (b + q * se < a["bound_pp"])
    out["H2_power_true_zero"] = eq / a["reps"]
    return out


def main() -> None:
    print(json.dumps(simulate(), indent=2))


if __name__ == "__main__":
    main()
