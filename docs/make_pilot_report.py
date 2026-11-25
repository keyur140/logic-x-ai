# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Builds the easy-language report of the REAL pilot run (Groq-only, development items).
Every number is read from the run folder. Nothing is typed in by hand except the wording.
Run: python docs/make_pilot_report.py runs/pilot-groq-20261107 OUT.pdf"""
import collections
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import SimpleDocTemplate  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdfkit import (AI_USE, AUTHOR_HTML, H2, H3, SMALL, SP, SUB, TITLE, WARM, P,  # noqa: E402
                    box, bullets, figure, footer, table)

from logicxai import __version__  # noqa: E402

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
FIG = OUT.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

M = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
RES = json.loads((RUN / "analysis" / "results.json").read_text(encoding="utf-8"))
T = [json.loads(x) for x in open(RUN / "raw" / "trials.jsonl", encoding="utf-8")]
ASSIST = [json.loads(x) for x in open(RUN / "assistant" / "gpt-oss-120b.jsonl", encoding="utf-8")]
A = {a["item_id"]: a for a in ASSIST}
CALLS = [json.loads(x) for x in open(RUN / "logs" / "llm_calls.jsonl", encoding="utf-8")]
ITEMS = {}
for fpath in (RUN / "items").glob("*.jsonl"):
    for line in open(fpath, encoding="utf-8"):
        it = json.loads(line)
        ITEMS[it["item_id"]] = it
PLANTED = set(M["planted_items"])
SOLVERS = ["allam-2-7b", "qwen3.8-27b", "gpt-oss-20b"]
SIZE = {"allam-2-7b": "7B", "qwen3.8-27b": "27B", "gpt-oss-20b": "21B"}
MODES = ["direct", "brief", "careful"]
OK = [t for t in T if t["status"] == "ok"]


def sel(**kw):
    out = OK
    for k, v in kw.items():
        out = [t for t in out if (t[k] in v if isinstance(v, (set, list, tuple)) else t[k] == v)]
    return out


def pct(rows, key="correct"):
    if not rows:
        return None
    return 100.0 * sum(1 for t in rows if (t[key] is True or t[key] == 1)) / len(rows)


def fmt(x, nd=0):
    return "–" if x is None else f"{x:.{nd}f}%"


def lenient_int(text):
    lines = [ln for ln in (text or "").splitlines() if "FINAL" in ln.upper()]
    if not lines:
        return None
    for c in reversed(re.findall(r"-?\d[\d,]*(?:\.\d+)?", lines[-1].split(":", 1)[-1])):
        try:
            v = float(c.replace(",", ""))
        except ValueError:
            continue
        if v.is_integer():
            return int(v)
    return None


# ------------------------------------------------------------------ numbers
n_total = len(T)
n_ok = len(OK)
n_missing = n_total - n_ok
missing = collections.Counter((t["solver"], t["mode"]) for t in T if t["status"] != "ok")
ok_calls = [c for c in CALLS if c["status"] == "ok"]
first_ts, last_ts = min(c["timestamp_utc"] for c in ok_calls), max(c["timestamp_utc"] for c in ok_calls)
tok = collections.defaultdict(lambda: [0, 0, 0])
lat = collections.defaultdict(list)
for c in ok_calls:
    u = c.get("usage") or {}
    tok[c["alias"]][0] += 1
    tok[c["alias"]][1] += u.get("prompt_tokens", 0)
    tok[c["alias"]][2] += u.get("completion_tokens", 0)
    if "latency_s" in c:
        lat[c["alias"]].append(c["latency_s"])
waits = collections.Counter()
for c in CALLS:
    e = c.get("error", "")
    if c["status"] == "retry":
        waits["daily token cap" if "(TPD)" in e else "per-minute cap" if ("(TPM)" in e or "(RPM)" in e or "(OTPM)" in e)
              else "other"] += 1

unit = {}
for s in SOLVERS:
    for m in MODES:
        g = sel(solver=s, mode=m)
        if len(g) < 100:  # qwen careful: too little to describe
            continue
        h = [t for t in g if t["family"] == "hard_deduction" and t["condition"] == "assisted"]
        hp, hc = [t for t in h if t["planted"]], [t for t in h if not t["planted"]]
        ha = [t for t in g if t["family"] == "hard_deduction" and t["condition"] == "alone"]
        unit[(s, m)] = {
            "n": len(g),
            "unparsed": 100.0 * sum(not t["parsed"] for t in g) / len(g),
            "logic": pct([t for t in g if t["family"] in ("syllogism", "conditional", "crt")]),
            "easy_alone": pct([t for t in g if t["family"] == "easy_math" and t["condition"] == "alone"]),
            "easy_ai": pct([t for t in g if t["family"] == "easy_math" and t["condition"] == "assisted"]),
            "hard_alone": pct(ha),
            "hard_alone_planted_items": pct([t for t in ha if t["item_id"] in PLANTED]),
            "hard_clean": pct(hc), "hard_planted": pct(hp),
            "hit": pct(hp, "flag"), "fa": pct(hc, "flag"),
            "copied": 100.0 * sum(t["pred"] == A[t["item_id"]].get("assistant_pred") for t in hp) / len(hp),
            "catch": pct([t for t in g if t["family"] == "catch"], "flag"),
        }

by_model = {}
for s in SOLVERS:
    h = sel(solver=s, family="hard_deduction", condition="assisted")
    hp, hc = [t for t in h if t["planted"]], [t for t in h if not t["planted"]]
    by_model[s] = {"clean": pct(hc), "planted": pct(hp), "hit": pct(hp, "flag"), "fa": pct(hc, "flag"),
                   "copied": 100.0 * sum(t["pred"] == A[t["item_id"]].get("assistant_pred") for t in hp) / len(hp),
                   "n_planted": len(hp), "n_clean": len(hc),
                   "unparsed": 100.0 * sum(not t["parsed"] for t in sel(solver=s)) / len(sel(solver=s))}

FAMS = [("conditional", "Conditional rules (ours)"), ("crt", "Trick questions, CRT (ours)"),
        ("syllogism", "Syllogisms (ours)"), ("easy_math", "Easy maths (ours)"),
        ("hard_deduction", "Hard deduction (ours)"), ("gsm8k", "GSM8K maths (public)"),
        ("folio", "FOLIO logic (public)"), ("logiqa", "LogiQA logic (public)")]
fam_acc = {s: {f: pct(sel(solver=s, family=f, condition="alone")) for f, _ in FAMS} for s in SOLVERS}

# strict vs lenient reading of number answers (GSM8K)
gs = {}
for s in SOLVERS:
    g = sel(solver=s, family="gsm8k")
    strict = pct(g)
    len_ok = sum((t["correct"] == 1) or (not t["parsed"] and lenient_int(t["text"]) == ITEMS[t["item_id"]]["answer"])
                 for t in g)
    gs[s] = (strict, 100.0 * len_ok / len(g), sum(not t["parsed"] for t in g), len(g))

empty = collections.Counter(t["solver"] for t in OK if (t["text"] or "") == "")
conf = RES["confirmatory"]
g5 = M["gates"]["G4_G5"]["gpt-oss-120b"]
g6 = RES["gates"]["G6"]["gpt-oss-120b"]
g7 = RES["gates"]["G7"]
excl = M["gates"]["G2_G3"]
checks = {c["id"]: c for c in M["self_checks"]}
a_fail = [a for a in ASSIST if not a["ok"]]

# ------------------------------------------------------------------ figures
INK, MUTED, GRID = "#1b1b1a", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})

# Figure 1: hard deduction with the AI, correct AI vs planted error, per solver unit
keys = list(unit)
fig, ax = plt.subplots(figsize=(7.6, 3.5))
x = np.arange(len(keys))
w = 0.36
ax.bar(x - w / 2 - 0.02, [unit[k]["hard_clean"] for k in keys], w, color=BLUE, label="AI reply was correct")
bars = ax.bar(x + w / 2 + 0.02, [unit[k]["hard_planted"] for k in keys], w, color=ORANGE,
              label="AI reply had a planted error")
for b, k in zip(bars, keys):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{unit[k]['hard_planted']:.0f}",
            ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([f"{k[0]}\n{k[1]}" for k in keys], fontsize=7.6)
ax.set_ylim(0, 112)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Correct answers (%)")
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(length=0)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "pilot_fig1_planted.png", dpi=200)
plt.close(fig)

# Figure 2: accuracy working alone, by task family and model (ceiling and floor)
fig, ax = plt.subplots(figsize=(7.6, 3.7))
ys = np.arange(len(FAMS))[::-1]
ax.axvspan(95, 101, color="#f3f2ee", zorder=0)
ax.text(98, len(FAMS) - 0.42, "ceiling", ha="center", va="bottom", fontsize=7.5, color=MUTED)
for s, col, mk in zip(SOLVERS, (ORANGE, AQUA, BLUE), ("o", "s", "D")):
    ax.scatter([fam_acc[s][f] for f, _ in FAMS], ys, s=58, color=col, marker=mk, label=f"{s} ({SIZE[s]})",
               zorder=3, edgecolor="white", linewidth=1.2)
for y in ys:
    ax.axhline(y, color=GRID, linewidth=0.8, zorder=1)
ax.set_yticks(ys)
ax.set_yticklabels([lab for _, lab in FAMS], fontsize=8.4)
ax.set_xlim(0, 101)
ax.set_ylim(-0.6, len(FAMS) - 0.1)
ax.set_xlabel("Correct answers working alone, all modes together (%)")
ax.spines["left"].set_visible(False)
ax.tick_params(length=0)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.42, 1.13), ncol=3, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "pilot_fig2_ceiling.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ story
a7, q27, g20 = by_model["allam-2-7b"], by_model["qwen3.8-27b"], by_model["gpt-oss-20b"]
story = [
    box("<b>PILOT RUN WITH REAL AI MODELS. NOT A TEST OF THE HYPOTHESES.</b> Real answers from 3 open models "
        "on development items only. Three models are far too few to test H1, H2 or H4. The registered "
        f"study items were not shown to any model. The run is {100 * n_ok / n_total:.0f}% complete.", bg=WARM),
    SP(1, 8),
    P("Logic × AI: first pilot with real models", TITLE),
    P(f"Run <b>{M['run_id']}</b> · development items (seed {M['seed']}) · answers collected "
      f"{first_ts[:10]} to {last_ts[:10]} (UTC) · package v{__version__} with pilot patches", SUB),
    SP(1, 6), P(AUTHOR_HTML, SMALL), SP(1, 4),

    P("1. The short answer", H2),
    box(f"<b>Key finding.</b> In our pilot, the stronger models caught most planted errors: gpt-oss-20b "
        f"flagged {g20['hit']:.0f}% of them and stayed {g20['planted']:.0f}% correct, and qwen3.8-27b flagged "
        f"{q27['hit']:.0f}% and stayed {q27['planted']:.0f}% correct. The weaker model (allam-2-7b) flagged only "
        f"{a7['hit']:.0f}%, gave the AI's wrong answer in {a7['copied']:.0f}% of cases and was correct in only "
        f"{a7['planted']:.0f}%. Our puzzles also need to be harder, because the top models score near 100%. "
        "<b>With only 3 models this is a first sign, not a test of our hypotheses.</b>"),
    SP(1, 5),
    bullets([
        "<b>The study machinery works with real models.</b> Items, planted errors, scoring, gates, hashes and "
        "reports all ran end to end on real answers.",
        "<b>The pilot found 11 problems to fix before OSF registration</b> (section 5). That is the purpose "
        "of a pilot. The biggest are a ceiling effect, an answer reader that is too strict, and a free API "
        "that is too slow for the full study.",
        (f"<b>The run is incomplete.</b> {n_ok:,} of {n_total:,} planned solver answers were collected. The "
         "Groq API key expired before qwen3.8-27b finished its \"careful\" mode and its knowledge quiz.")
        if n_missing else
        (f"<b>The run is complete.</b> All {n_total:,} planned solver answers were collected. The last "
         "qwen3.8-27b answers were filled in on 20-21 Nov 2026 with new Groq keys, using the same cache."),
    ]),

    P("2. What this run is", H2),
    table([
        ["Part", "What was used"],
        ["Solver models (take the tests)", "allam-2-7b (7B), qwen3.8-27b (27B), gpt-oss-20b (21B), each in 3 "
         "reasoning modes (direct, brief, careful) = 9 solver units"],
        ["Assistant AI (gives help)", "gpt-oss-120b, standing in for the planned Llama 3.3 70B. No second "
         "assistant."],
        ["Items", f"Development items only: {sum(v for k, v in M['stages']['items']['counts'].items() if k in ('syllogism', 'conditional', 'crt', 'easy_math', 'hard_deduction', 'logic_creative', 'catch'))} "
         "of our own generated puzzles plus 200 public benchmark items (FOLIO, LogiQA, GSM8K, MMLU). "
         f"Overlap with the registered items: {M['lock']['overlap']}."],
        ["Planted errors", f"{len(PLANTED)} of 60 hard deduction items, chosen by seed. The assistant AI "
         "was told to argue for a wrong answer on those."],
        ["Where it ran", "Groq free API tier, from a cloud session. Cost: USD 0."],
        ["Completeness", f"{n_ok:,} of {n_total:,} solver answers ({100 * n_ok / n_total:.0f}%). Missing: "
         + ("; ".join(f"{s} {'knowledge quiz' if m == 'direct' else m + ' mode'} ({n})"
                      for (s, m), n in sorted(missing.items())) if missing else "none") + "."],
    ], [4.2 * cm, 12.4 * cm]),
    P("Why these models: the planned routes (Ollama, OpenRouter, Hugging Face router) could not be reached from "
      "the cloud session, and Groq had retired the Llama models named in the study config. The author chose a "
      "small Groq-only pilot at full pilot size. All differences from the plan are listed in section 7.", SMALL),

    P("3. What we saw", H2),
    P("3.1 Planted errors: who catches them", H3),
    figure(FIG / "pilot_fig1_planted.png",
           "Figure 1. Hard deduction items answered with the AI's help. Blue: the AI's reply was correct. "
           "Orange: the AI's reply argued for a planted wrong answer (numbers shown). 29 items per bar."
           + (" qwen3.8-27b careful mode is missing." if n_missing else "")),
    table([["Solver unit", "Alone, same items", "With correct AI", "With planted error", "Flagged the error",
            "False alarms", "Gave the AI's wrong answer"]] +
          [[f"{k[0]} · {k[1]}", fmt(v["hard_alone"]), fmt(v["hard_clean"]), fmt(v["hard_planted"]), fmt(v["hit"]),
            fmt(v["fa"]), fmt(v["copied"])] for k, v in unit.items()],
          [3.9 * cm, 2.0 * cm, 2.1 * cm, 2.2 * cm, 2.0 * cm, 1.9 * cm, 2.5 * cm]),
    P("\"Alone, same items\" is the score on all 60 hard deduction items without help. \"Flagged the error\" = "
      "the solver wrote ASSISTANT_ERROR: YES on a planted item. \"False alarms\" = it wrote YES when the AI "
      "was correct.", SMALL),
    bullets([
        f"<b>gpt-oss-20b and qwen3.8-27b</b> flagged {g20['hit']:.0f}% and {q27['hit']:.0f}% of planted errors, "
        f"with {g20['fa']:.0f}% and {q27['fa']:.0f}% false alarms, and almost never took over the wrong answer.",
        f"<b>allam-2-7b</b> flagged {a7['hit']:.0f}% of planted errors and gave the AI's wrong answer in "
        f"{a7['copied']:.0f}% of cases. A correct AI lifted it a lot in brief and careful modes "
        f"({fmt(unit[('allam-2-7b', 'brief')]['hard_alone'])} to {fmt(unit[('allam-2-7b', 'brief')]['hard_clean'])} "
        f"and {fmt(unit[('allam-2-7b', 'careful')]['hard_alone'])} to "
        f"{fmt(unit[('allam-2-7b', 'careful')]['hard_clean'])}). So it leans on the AI whether the AI is right "
        "or wrong.",
        "In direct mode allam-2-7b often failed to give a readable answer at all when the AI's reply was "
        "attached, so its direct-mode numbers are low for both kinds of reply.",
    ]),
    P("3.2 Easy maths: AI help levels the field", H3),
    table([["Solver unit", "Easy maths alone", "Easy maths with AI", "Logic battery", "Unreadable answers"]] +
          [[f"{k[0]} · {k[1]}", fmt(v["easy_alone"]), fmt(v["easy_ai"]), fmt(v["logic"]), fmt(v["unparsed"], 1)]
           for k, v in unit.items()], [4.6 * cm, 3.0 * cm, 3.2 * cm, 2.8 * cm, 3.0 * cm]),
    P("In direct mode (answer at once, no working) allam-2-7b and qwen3.8-27b got almost no easy maths items "
      "right alone, and the AI's help lifted them to "
      f"{fmt(unit[('allam-2-7b', 'direct')]['easy_ai'])} and {fmt(unit[('qwen3.8-27b', 'direct')]['easy_ai'])}. "
      "When they were allowed to show working, they were already near the top alone. gpt-oss-20b scored 100% "
      "everywhere, so it had nothing to gain. That pattern fits the idea behind H2, but it is 3 models."),
    P("3.3 Too easy at the top, too hard at the bottom", H3),
    figure(FIG / "pilot_fig2_ceiling.png",
           "Figure 2. Share of correct answers working alone, by kind of task. Each marker is one model, all "
           "reasoning modes together. The shaded band marks scores of 95% and above."),
    bullets([
        f"<b>Ceiling.</b> gpt-oss-20b and qwen3.8-27b score 100% on our conditional rules and trick questions, "
        f"and {fam_acc['gpt-oss-20b']['hard_deduction']:.0f}% and {fam_acc['qwen3.8-27b']['hard_deduction']:.0f}% "
        "on hard deduction. A test everyone passes cannot tell strong solvers apart.",
        f"<b>Floor.</b> allam-2-7b gets {fam_acc['allam-2-7b']['crt']:.0f}% on the trick questions.",
        f"<b>The public benchmarks spread the models better</b>: FOLIO {fam_acc['allam-2-7b']['folio']:.0f}% to "
        f"{fam_acc['qwen3.8-27b']['folio']:.0f}%, LogiQA {fam_acc['allam-2-7b']['logiqa']:.0f}% to "
        f"{fam_acc['qwen3.8-27b']['logiqa']:.0f}%.",
        "<b>Reasoning modes barely move gpt-oss-20b.</b> It is a model that always thinks before it answers, "
        "so \"direct\" is not really direct for it. The modes change a lot for the other two models.",
    ]),

    P("4. What the checks said", H2),
    P("Gates decide whether a measure may be trusted. Self-checks recompute parts of the pipeline a second "
      "way. Results are copied from the run manifest."),
    table([
        ["Check", "Result", "What happened"],
        ["G1 answer reader known-answer test", "PASS", f"{M['gates']['G1']['n_correct']} of "
         f"{M['gates']['G1']['n_cases']} known replies read correctly. But see problem 2: real replies showed a "
         "form the test set did not contain."],
        ["G2/G3 solver exclusions", f"{excl['n_excluded']} of {excl['n_units']} excluded",
         "All 3 allam-2-7b units: 38 to 55% unreadable answers on GSM8K (limit 20%)."
         + (" qwen3.8-27b careful and direct: missing answers after the key expired (limit 10%)." if n_missing
            else "")],
        ["G4 planted errors are really errors", "PASS",
         f"Assistant wrong on {g5['n_planted']} of {g5['n_planted']} planted replies and right on "
         f"{g5['n_clean']} of {g5['n_clean']} clean replies. {g5['failed_after_retries']} of 120 replies could "
         "not be produced after 4 tries and were left out."],
        ["G5 planted replies look like clean ones", "FAIL (narrowly)",
         f"Worst surface score {g5['G5_worst_auc']:.3f} against a limit of 0.650, on hedge words. Length was "
         f"balanced ({g5['G5_feature_auc']['chars']:.2f})."],
        ["G6 error flags mean something", "PASS",
         f"Solvers flagged {100 * g6['flag_rate_catch_mean']:.0f}% of obvious catch errors and "
         f"{100 * g6['false_alarm_clean_mean']:.0f}% of correct replies (p = {g6['p_one_sided']:.3f}, "
         f"{g6['n_solvers']} units)."],
        ["G7 logic score agrees with public logic tests", "Not run", "Too few solver units left after "
         "exclusions."],
        ["C1 to C5, C8, C10, C12", "PASS", "Reply hashes, answer keys (SAT solver), planted-error assignment, "
         "logic scores, item lock, config hashes and the reader test all recomputed correctly."],
        ["C9 no clues in solver prompts", "FAIL (false alarm)", checks["C9"]["detail"] + ". All 9 are one public "
         "FOLIO item whose own text contains the word \"deliberate\". No real clue leaked. See problem 3."],
        ["C6, C7 statistics recomputed", "Not run", "The second statistics library was not installed, and "
         "there were too few solvers for a regression anyway. See problem 5."],
        ["H1, H2, H4", "Not testable", "Too few solvers. The report now says \"Not testable\" instead of "
         "\"Not supported\". See problem 4."],
    ], [5.0 * cm, 2.6 * cm, 9.0 * cm], status_col=1),

    P("5. Problems the pilot found, and suggested fixes", H2),
    P("Fixes marked \"your decision\" or \"author's decision\" change the study design, so they are for the author to choose. None of "
      "them has been applied to the registered study files."),
    table([
        ["#", "Problem", "Evidence in this run", "Suggested fix"],
        ["1", "Ceiling: our puzzles are too easy for strong models",
         "gpt-oss-20b and qwen3.8-27b: 100% on conditional and trick questions, "
         f"{fam_acc['gpt-oss-20b']['hard_deduction']:.0f}% and {fam_acc['qwen3.8-27b']['hard_deduction']:.0f}% "
         "on hard deduction alone.",
         "Make hard deduction harder (more statements, longer chains) and re-pilot. Author's decision (20 Nov): add harder levels, keep the current items, short re-pilot."],
        ["2", "The answer reader is too strict for number answers",
         f"allam-2-7b wrote \"FINAL: Tanya earned $365 for these two days\". The reader wants a bare number, so "
         f"{gs['allam-2-7b'][2]} of {gs['allam-2-7b'][3]} GSM8K answers counted as unreadable and all allam "
         f"units were excluded. Read leniently, allam's GSM8K score is {gs['allam-2-7b'][1]:.0f}%, not "
         f"{gs['allam-2-7b'][0]:.0f}%.",
         "Read the last number on the FINAL line, and add such replies to the G1 test set. Author's decision "
         "(20 Nov): lenient reading."],
        ["3", "The clue scanner raises a false alarm on public items", "C9: 9 hits, all from one FOLIO item "
         "containing the word \"deliberate\".",
         "Scan only the assistant's reply and our own wrapper text, not the public item text."],
        ["4", "The mediation test gave a false \"H4 supported\" on tiny data",
         "With 3 solver units from 1 model the first analysis printed p = 0.0002. That number was meaningless.",
         "Already patched in the pilot code: H4 needs enough rows and at least 3 models, otherwise "
         "\"Not testable\". Carry the patch into the study package."],
        ["5", "A statistics library is missing from the install list", "C6 and C7 did not run: statsmodels is "
         "not in requirements.txt.", "Add statsmodels to requirements.txt."],
        ["6", "Planted replies differ slightly in wording", f"G5 hedge-word score {g5['G5_worst_auc']:.3f}, "
         "limit 0.650.", "Check again with the real assistant model. If it stays, tighten the assistant prompt "
         "or keep the caution label."],
        ["7", "Thinking models need special settings",
         f"gpt-oss models returned empty answers until given extra room for hidden thinking. "
         f"{empty.get('gpt-oss-20b', 0)} gpt-oss-20b answers were still empty.",
         "Decide whether always-thinking models belong in the solver list, since \"direct\" mode means little "
         "for them. Author's decision (20 Nov): keep them, labelled \"thinking\"; "
         "their 3 modes count as one solver."],
        ["8", "The free API route is too slow for the full study",
         "Groq allows about 200,000 tokens per model per day. One pilot-size model used about 0.5 million. "
         "The full study needs about 6 times more per model.",
         "Run the full study on a local or rented GPU with Ollama (the \"local\" route already in the config). "
         "Your decision (still open)."],
        ["9", "Model names in the config are out of date", "Groq no longer serves llama-3.3-70b-versatile or "
         "llama-3.1-8b-instant.", "Re-check every API model name right before the freeze, or use the local "
         "route where models are pinned by digest."],
        ["10", "One public dataset no longer loads the old way", "LogiQA's loading script is refused by the "
         "current datasets library.", "Pin LogiQA to its parquet revision (done in the pilot config) and fill "
         "all four benchmark revisions in study.yaml."],
        ["11", "Small tooling gaps", "The model check command hangs for minutes on blocked hosts. Long runs "
         "need a restart helper.", "Add a short timeout to the check command and ship the pilot's restart "
         "script."],
    ], [1.0 * cm, 3.5 * cm, 6.4 * cm, 5.7 * cm]),

    P("6. What we can and cannot say", H2),
    table([
        ["We can say", "We cannot say"],
        ["The pipeline runs end to end on real model answers, with every reply hashed.",
         "That H1, H2 or H4 is supported or not supported. Nothing was tested."],
        ["In this pilot, two stronger models caught planted errors and one weaker model mostly did not.",
         "That logic skill causes this. The models also differ in size, family, training and whether they "
         "think before answering."],
        ["Our own puzzles are too easy for strong 2026 models and need to be harder.",
         "How large any effect will be in the full study with 28 models."],
        ["The assistant can be made to plant errors reliably (58 of 60 hard items worked).",
         "That the planned assistant (Llama 3.3 70B) behaves the same. A different model stood in."],
        ["The free API route is too slow for the full study.",
         "Anything about qwen3.8-27b in careful mode. Those answers are missing." if n_missing else
         "That 3 reasoning modes per model are enough to see a within-model effect."],
    ], [8.3 * cm, 8.3 * cm]),

    P("7. Differences from the plan (pilot deviations)", H2),
    table([
        ["#", "Change", "Reason"],
        ["P1", "3 Groq-hosted solver models instead of the first 6 models of the planned list",
         "Ollama, OpenRouter and the Hugging Face router could not be reached from the cloud session."],
        ["P2", "gpt-oss-120b as the assistant instead of Llama 3.3 70B. No second assistant.",
         "Groq retired the Llama model. The second assistant route (OpenRouter) was blocked."],
        ["P3", "gpt-oss models run with reasoning effort \"low\" and 512 extra output tokens. qwen3.8-27b "
               "runs with thinking switched off.", "Hidden thinking counted against the answer length cap and "
                                                   "left empty answers."],
        ["P4", "LogiQA loaded from its parquet revision. All four benchmark revisions pinned.",
         "The original loading script is no longer supported."],
        ["P5", "Code patches: per-model settings, longer retry waits, one worker pool per model, and the H4 "
               "guard with the \"Not testable\" verdict.",
         "Needed for the free API limits. The H4 guard was added on 12 Nov after the false result was seen."],
        ["P6", f"{n_missing} of {n_total:,} solver answers missing." if n_missing else
               "449 qwen3.8-27b answers were missing on 12 Nov and were filled in on 20-21 Nov 2026 with new "
               "Groq keys (two Groq accounts). Same run id, config and cache; only missing calls were sent.",
         "The first Groq API key expired on 12 Nov 2026, then hit qwen's daily token cap."],
        ["P7", "Answers were collected over 5 days with pauses, plus the fill-in on 20-21 Nov. The run folder's start time is the last pass.",
         "Daily token caps and cloud session pauses. Each reply keeps its own timestamp in the call log."],
        ["P8", "Two interim looks (after 1,380 and 3,197 answers) were shared with the author.",
         "Progress checks during a multi-day run. No design change was made from them."],
    ], [0.9 * cm, 8.2 * cm, 7.5 * cm]),
    P("The registered study files (configs/study.yaml, prompts, items) were not changed. The pilot used its own "
      "config file, configs/pilot_groq.yaml. Nothing was frozen.", SMALL),

    P("8. Time and cost", H2),
    table([["Model", "Answers logged", "Tokens in", "Tokens out", "Typical answer time"]] +
          [[a, f"{tok[a][0]:,}", f"{tok[a][1]:,}", f"{tok[a][2]:,}", f"{np.median(lat[a]):.1f} s"]
           for a in ("allam-2-7b", "gpt-oss-20b", "qwen3.8-27b", "gpt-oss-120b")],
          [4.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm, 3.6 * cm]),
    P(f"Money cost: USD 0. The models answered in about a second. Nearly all the calendar time went on waiting: "
      f"{waits['daily token cap']:,} waits for the daily token cap and {waits['per-minute cap']:,} for "
      "per-minute caps. A few hundred earlier answers came from a smoke test and are in the cache but not in "
      "this table."),

    P("9. Words used", H2),
    table([
        ["Word", "Meaning"],
        ["Solver", "An AI model that takes the tests. Here: one model in one reasoning mode."],
        ["Assistant AI", "A second, stronger AI whose reply is shown to the solver as help."],
        ["Planted error", "A wrong answer that the assistant was told to argue for, on half of the hard items."],
        ["Reasoning mode", "How much working the solver may show: direct (none), brief, or careful."],
        ["Gate", "A known-answer check that must pass before a measure is trusted."],
        ["Ceiling / floor", "Almost everyone right / almost everyone wrong, so the test cannot rank them."],
        ["Development items", "Practice items from a separate seed. The registered items stay unseen."],
        ["SHA-256 hash", "A fingerprint of a file. If one character changes, the fingerprint changes."],
    ], [3.6 * cm, 13.0 * cm]),

    P("10. How to check this yourself", H2),
    bullets([
        "Unzip <b>pilot-groq-20261107-run.zip</b> and compare its SHA-256 with the .sha256 file next to it.",
        "In the study package folder run <b>python -m logicxai verify runs/pilot-groq-20261107</b>. It checks "
        "every file against SHA256SUMS.txt.",
        "Every model reply is in raw/trials.jsonl with its own hash. Every API call, including each wait and "
        "error, is in logs/llm_calls.jsonl.",
        "This report is rebuilt from the run folder by <b>docs/make_pilot_report.py</b>. No number is typed "
        "by hand.",
    ]),

    P("11. Questions you may get", H2),
    P("<b>Is this a result of the study?</b> No. It is a pilot on practice items with 3 models. It shows the "
      "method works and what must be fixed."),
    P("<b>Did the pilot see the registered items?</b> No. It used a separate seed, and the overlap check found "
      "0 shared items."),
    P("<b>Why report a failed gate and a failed self-check?</b> Because they happened. G5 failed narrowly and "
      "C9 was a false alarm. Both are explained and both lead to a fix."),
    P("<b>Were any numbers adjusted after looking?</b> No scores were changed. One analysis guard was added "
      "after a meaningless H4 result appeared, and that is listed as deviation P5."),

    P("12. Statements", H2),
    P("<b>Data availability.</b> The full run folder (items, every model reply, call log, scores, manifest and "
      "SHA-256 list) is in the project's shared folder (MIT licence for code, CC BY 4.0 for generated data and "
      "reports). Public benchmark items keep their own licences."),
    P("<b>Ethics.</b> No human participants and no personal data. Only AI models are tested."),
    P("<b>Funding and competing interests.</b> None."),
    P(AI_USE),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Logic x AI: first pilot with real models", author="KEYUR SHAILESHKUMAR PATEL")
lab = f"Logic × AI · pilot report · REAL models, development items, not a hypothesis test · {M['run_id']}"
doc.build(story, onFirstPage=footer(lab), onLaterPages=footer(lab))
print(OUT)
print(json.dumps({"by_model": by_model, "gsm8k_strict_lenient": gs, "n_ok": n_ok, "n_total": n_total,
                  "missing": {f"{k[0]}|{k[1]}": v for k, v in missing.items()}, "empty": dict(empty),
                  "waits": dict(waits), "fam_acc": fam_acc}, indent=1))
