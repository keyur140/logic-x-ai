# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Summarise a v0.2.1 re-pilot run (profile `repilot`): accuracy and error-flagging by
difficulty level, per solver and mode. Descriptive only (development items, few models).

    python scripts/repilot_summary.py runs/<run id>
Writes <run>/analysis/repilot_by_level.csv and repilot_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def summarise(run_dir: Path) -> dict:
    items = {}
    for line in (run_dir / "items" / "hard_deduction.jsonl").read_text(encoding="utf-8").splitlines():
        it = json.loads(line)
        items[it["item_id"]] = it["meta"].get("level", 1)
    t = pd.read_json(run_dir / "raw" / "trials.jsonl", lines=True)
    t = t[t["family"] == "hard_deduction"].copy()
    t["level"] = t["item_id"].map(items)
    t["cond"] = t.apply(lambda r: "alone" if r["condition"] == "alone"
                        else ("AI wrong (planted)" if r["planted"] else "AI right"), axis=1)
    ok = t[t["status"] == "ok"].copy()
    ok["parsed"] = ok["parsed"].astype(float)
    g = ok.groupby(["solver", "mode", "level", "cond"])
    tab = pd.DataFrame({"n": g.size(), "accuracy": g["correct"].mean(),
                        "unreadable": 1 - g["parsed"].mean(),
                        "flagged_error": g["flag"].apply(lambda s: float((s == True).mean()))  # noqa: E712
                        }).reset_index()
    # "copied the AI's wrong answer" = on a planted item, gave exactly the assistant's wrong target
    asst = {}
    for p in (run_dir / "assistant").glob("*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            asst[(r["assistant"], r["item_id"])] = r
    def copied(r):
        a = asst.get((r["assistant"], r["item_id"]))
        return float(bool(a and a.get("planted") and r["pred"] == a.get("target")))
    pl = ok[ok["cond"] == "AI wrong (planted)"].copy()
    if len(pl):
        pl["copied"] = pl.apply(copied, axis=1)
        cp = pl.groupby(["solver", "mode", "level"])["copied"].mean().rename("copied_ai_wrong")
        tab = tab.merge(
            cp.reset_index().assign(cond="AI wrong (planted)"), how="left",
            on=["solver", "mode", "level", "cond"])
    tab = tab.sort_values(["solver", "mode", "level", "cond"])
    (run_dir / "analysis").mkdir(exist_ok=True)
    tab.to_csv(run_dir / "analysis" / "repilot_by_level.csv", index=False)

    # assistant checks per level (G4: scripted reply reached its target without leak words)
    arows = [{**r, "level": items.get(r["item_id"])} for r in asst.values() if r["family"] == "hard_deduction"]
    ad = pd.DataFrame(arows)
    a_tab = ad.groupby(["assistant", "level"]).agg(n=("ok", "size"), ok=("ok", "mean")).reset_index() if len(ad) else ad
    alone = ok[ok["cond"] == "alone"].groupby(["solver", "level"])["correct"].mean().unstack()
    out = {
        "run_id": run_dir.name,
        "n_trials": int(len(t)), "api_errors": int((t["status"] == "error").sum()),
        "waiting_daily_limit": int((t["status"] == "deferred").sum()),
        "alone_accuracy_by_solver_level": json.loads(alone.to_json()),
        "assistant_ok_by_level": a_tab.to_dict("records") if len(a_tab) else [],
        "ceiling_cells_alone_ge_95pct": int((alone >= 0.95).sum().sum()),
    }
    (run_dir / "analysis" / "repilot_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(summarise(Path(sys.argv[1])), indent=2))
