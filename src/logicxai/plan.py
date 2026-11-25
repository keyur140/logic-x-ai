# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""`python -m logicxai plan`: how many calls, tokens and days a profile needs under the
free daily limits (decision 18, v0.2.1). An ESTIMATE for planning only: it changes no
data and sends nothing to any model. Token numbers are rough averages."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from .items import GENERATORS
from .llm import key_names
from .runner import BATTERY_FAMILIES, LOGIC_BENCH, TASK_FAMILIES

# Rough prompt sizes (tokens) for benchmark items, and typical reply lengths per mode.
BENCH_PROMPT = {"folio": 260, "logiqa": 330, "gsm8k": 140, "mmlu": 130}
REPLY = {"direct": 15, "brief": 110, "careful": 520}
ASSIST_REPLY = 220


def _tok(text: str) -> int:
    return max(1, len(text) // 4)


def estimate(cfg: dict[str, Any], profile: str) -> dict[str, Any]:
    prof = cfg["profiles"][profile]
    st = cfg["study"]
    seed = int(st["dev_seed"])
    fam_prompt: dict[str, float] = {}
    for fam, n in prof["items"].items():
        its = GENERATORS[fam](min(n, 60), seed)
        fam_prompt[fam] = sum(_tok(i.prompt) for i in its) / max(1, len(its))
    for b in (prof.get("benchmarks") or {}):
        fam_prompt[b] = BENCH_PROMPT.get(b, 200)
    n_items = {**prof["items"], **(prof.get("benchmarks") or {})}
    n_assist = len(cfg["assistants"])
    task_n = sum(n_items.get(f, 0) for f in TASK_FAMILIES)
    catch_n = n_items.get("catch", 0)

    per_model: list[dict[str, Any]] = []
    for s in cfg["solvers"]:
        calls = tok_in = tok_out = 0
        allowance = int(s.get("token_allowance") or 0) * 0.7
        for mode in st["modes"]:
            out = REPLY[mode] + allowance
            for fam in BATTERY_FAMILIES + LOGIC_BENCH + ("gsm8k",):
                n = n_items.get(fam, 0)
                calls += n
                tok_in += n * (fam_prompt.get(fam, 150) + 60)
                tok_out += n * out
            for fam in TASK_FAMILIES + ("catch",):
                n = n_items.get(fam, 0)
                k = (0 if fam == "catch" else 1) + n_assist
                calls += n * k
                tok_in += n * (fam_prompt.get(fam, 150) + 60) * k + n * n_assist * (ASSIST_REPLY + 80)
                tok_out += n * k * (out + 10)
        calls += n_items.get("mmlu", 0)
        tok_in += n_items.get("mmlu", 0) * (fam_prompt.get("mmlu", 130) + 60)
        tok_out += n_items.get("mmlu", 0) * (REPLY["direct"] + allowance)
        per_model.append({"alias": s["alias"], "provider": s["provider"], "params_b": s.get("params_b"),
                          "calls": int(calls), "tokens_in": int(tok_in), "tokens_out": int(tok_out)})
    for a in cfg["assistants"]:
        calls = (task_n + catch_n) * 1.3  # scripted replies are re-tried when they miss
        per_model.append({"alias": a["alias"] + " (assistant)", "provider": a["provider"],
                          "params_b": a.get("params_b"), "calls": int(calls),
                          "tokens_in": int(calls * 450), "tokens_out": int(calls * (ASSIST_REPLY + 300))})

    lim = cfg.get("free_tier_limits") or {}
    ollama_seconds = 0.0
    for m in per_model:
        p = m["provider"]
        L = lim.get(p, {})
        keys = max(1, len(key_names(p))) if p not in ("ollama", "mock") else 1
        m["keys"] = keys
        days = 0.0
        if p == "ollama":
            tps = float(lim.get("ollama", {}).get("tokens_per_second_7b", 10)) * 7.0 / max(0.5, float(m["params_b"] or 7))
            secs = m["tokens_out"] / tps + m["calls"] * 1.0 + m["tokens_in"] / (tps * 20)
            ollama_seconds += secs
            days = secs / 86400
        else:
            if L.get("requests_per_day"):
                days = max(days, m["calls"] / (L["requests_per_day"] * keys))
            if L.get("tokens_per_day"):
                days = max(days, (m["tokens_in"] + m["tokens_out"]) / (L["tokens_per_day"] * keys))
            if L.get("tokens_per_minute"):
                days = max(days, (m["tokens_in"] + m["tokens_out"]) / (L["tokens_per_minute"] * keys * 1440))
            if L.get("requests_per_second"):
                days = max(days, m["calls"] / (L["requests_per_second"] * 86400 * keys))
        m["days"] = days
    api_days = max([m["days"] for m in per_model if m["provider"] != "ollama"] or [0])
    return {"profile": profile, "models": per_model, "api_days": api_days,
            "ollama_days": ollama_seconds / 86400,
            "total_calls": sum(m["calls"] for m in per_model)}


def print_plan(config: str, profile: str) -> int:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    e = estimate(cfg, profile)
    print(f"Plan for profile '{profile}' (ESTIMATE; free limits from configs, check each provider's console)\n")
    print(f"{'model':30s} {'where':10s} {'keys':>4s} {'calls':>8s} {'tokens':>11s} {'days':>6s}")
    for m in e["models"]:
        print(f"{m['alias']:30s} {m['provider']:10s} {m['keys']:>4d} {m['calls']:>8d} "
              f"{m['tokens_in'] + m['tokens_out']:>11,d} {m['days']:>6.1f}")
    print(f"\nTotal calls: {e['total_calls']:,}")
    print(f"Free API part: about {math.ceil(e['api_days'])} day(s) (models run side by side; "
          "more keys = fewer days).")
    print(f"Your computer (Ollama, one model at a time): about {e['ollama_days']:.1f} day(s) of running time.")
    print("The run pauses by itself when a daily limit is used up; start it again (or use "
          "scripts/run_until_done.sh) and it continues from the saved answers.")
    return 0
