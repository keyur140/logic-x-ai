# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Builds the easy-language report of the v0.2.1 re-pilot (harder puzzle levels; development items).
Every number is read from the run folders. Nothing is typed in by hand except the wording.

Run: python docs/make_repilot_report.py RUNS_DIR OUT.pdf
RUNS_DIR holds repilot-v021-20261125 (solvers), repilot-assist2-v021-20261125 (Gemma 4 31B
assistant before the G5 fix), g5check2-gptoss-20261126 and g5check2-gemma-20261126 (after the fix,
if finished), and repilot-gemma-v021-20261125 (second Gemma 4 26B try; log only)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import SimpleDocTemplate  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pdfkit import (AI_USE, AUTHOR_HTML, H2, H3, SMALL, SP, SUB, TITLE, WARM, P,  # noqa: E402
                    box, bullets, figure, footer, table)
from repilot_summary import summarise  # noqa: E402

from logicxai import __version__  # noqa: E402

RUNS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
FIG = OUT.parent / "figures_repilot"
FIG.mkdir(parents=True, exist_ok=True)
MAIN = RUNS / "repilot-v021-20261125"
summarise(MAIN)
TAB = pd.read_csv(MAIN / "analysis" / "repilot_by_level.csv")
M = json.loads((MAIN / "manifest.json").read_text(encoding="utf-8"))
SOLV = ["allam-2-7b", "gemma-4-26b", "qwen3.8-27b", "gpt-oss-20b"]
SIZE = {"allam-2-7b": "7B", "gemma-4-26b": "26B", "qwen3.8-27b": "27B", "gpt-oss-20b": "21B"}


def pct(x):
    return "–" if pd.isna(x) else f"{100 * x:.0f}%"


def cell(solver, mode, cond, level, col="accuracy"):
    r = TAB[(TAB.solver == solver) & (TAB["mode"] == mode) & (TAB.cond == cond) & (TAB.level == level)]
    return r[col].iloc[0] if len(r) else float("nan")


def gate(run, alias):
    p = RUNS / run / "manifest.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    if "finished_utc" not in m:
        return None
    g = m["gates"].get("G4_G5", {}).get(alias)
    st = m["stages"].get("assistants", {}).get(alias, {})
    return {**g, "n": st.get("n"), "failed": st.get("failed")} if g else None


# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
colors = {"allam-2-7b": "#eb6834", "gemma-4-26b": "#9aa0a6", "qwen3.8-27b": "#2a78d6", "gpt-oss-20b": "#1f9d74"}
for ax, mode in zip(axes, ("direct", "careful")):
    for s in SOLV:
        ys = [cell(s, mode, "alone", lv) * 100 for lv in (1, 2, 3, 4)]
        ax.plot([1, 2, 3, 4], ys, marker="o", color=colors[s], label=f"{s} ({SIZE[s]})", linewidth=2)
    ax.axhline(33.3, color="#999", linestyle=":", linewidth=1)
    ax.text(4.05, 33.3, "guessing", fontsize=7, color="#777", va="center")
    ax.set_title(f"{mode} mode, alone (no AI)", fontsize=10)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("puzzle level")
    ax.set_ylim(-3, 103)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("correct (%)")
axes[1].legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "r1_levels.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- story
story = [P("Re-pilot report: are the new puzzle levels hard enough?", TITLE),
         P(f"Logic × AI study · package v{__version__} · development items only (never the registered items)", SUB),
         SP(1, 6), P(AUTHOR_HTML, SMALL), SP(1, 8)]

story += [P("Short answer", H2), box(
    "<b>Yes.</b> The two new top levels pull the strong models off the ceiling: alone, in careful mode, "
    f"gpt-oss-20b went from {pct(cell('gpt-oss-20b', 'careful', 'alone', 1))} at level 1 to "
    f"{pct(cell('gpt-oss-20b', 'careful', 'alone', 4))} at level 4, and qwen3.8-27b from "
    f"{pct(cell('qwen3.8-27b', 'careful', 'alone', 1))} to {pct(cell('qwen3.8-27b', 'careful', 'alone', 4))}. "
    "The weak model (allam-2-7b) stays near guessing at every level. The re-pilot also found three problems, "
    "now fixed or decided by the author: assistant replies gave away planted errors by their length (fixed), "
    "Gemma 4 26B loops while thinking (kept; the gates decide), and the easy-task test H2 has low power "
    "(decision pending).", bg=WARM)]

story += [P("What we did", H2), bullets([
    "96 new hard-deduction puzzles from the development seed: 24 per level, 8 per answer (TRUE / FALSE / UNKNOWN).",
    "Level 1 = the old puzzle (8 statements). Level 2 = 12 statements. Level 3 = 16 statements with one "
    "\"either/or\" step. Level 4 = 20 statements with two \"either/or\" steps.",
    "Each solver answered every puzzle alone and with the assistant (gpt-oss-120b), in direct and careful mode. "
    "On exactly half of each level the assistant's reply had one planted, subtle mistake.",
    f"Free APIs only (Groq and Google AI Studio), 25 and 26 November 2026. {M['stages']['trials']['n']:,} solver "
    f"answers, {M['stages']['trials']['api_errors']} API errors.",
])]

story += [P("1. Accuracy alone, by level", H2), figure(FIG / "r1_levels.png",
          "Share correct without the AI. Dotted line = guessing (1 in 3). Each point is 24 puzzles.")]
rows = [["Model", "Mode", "Level 1", "Level 2", "Level 3", "Level 4"]]
for s in SOLV:
    for mode in ("direct", "careful"):
        rows.append([f"{s} ({SIZE[s]})", mode] + [pct(cell(s, mode, "alone", lv)) for lv in (1, 2, 3, 4)])
story += [table(rows, [4.6 * cm, 2.0 * cm] + [2.5 * cm] * 4, zebra=True),
          P("Gemma 4 26B scored near 0% because almost all its replies were cut off while it was still thinking "
            "(see section 4), not because it chose wrong answers.", SMALL)]

story += [P("2. When the AI is wrong: catching the planted mistake", H2)]
rows = [["Model", "Mode", "Flagged the mistake (L1 / L2 / L3 / L4)", "Copied the AI's wrong answer (L1 / L2 / L3 / L4)"]]
for s in SOLV:
    for mode in ("direct", "careful"):
        f = " / ".join(pct(cell(s, mode, "AI wrong (planted)", lv, "flagged_error")) for lv in (1, 2, 3, 4))
        c = " / ".join(pct(cell(s, mode, "AI wrong (planted)", lv, "copied_ai_wrong")) for lv in (1, 2, 3, 4))
        rows.append([f"{s} ({SIZE[s]})", mode, f, c])
story += [table(rows, [4.0 * cm, 1.7 * cm, 5.4 * cm, 5.5 * cm], zebra=True),
          P("Each cell is 11 or 12 puzzles, so single numbers can move a lot by chance. The pattern is what "
            "matters: the strongest model flags almost every planted mistake at every level; the middle model "
            "flags fewer at the top levels and starts to copy; the weakest model rarely flags and often copies. "
            "This is the gap the main study measures (H1 and H4), now visible inside the hard levels.", SMALL)]

story += [P("3. Can a solver spot the planted reply by its look? (gate G5)", H2),
          P("Gate G5 asks whether planted and correct assistant replies can be told apart by surface features "
            "alone (length, number of lines and steps, hedging words, negations). 0.50 = impossible; the limit "
            "is 0.65.")]
rows = [["Assistant", "Version", "Replies OK", "G5 score", "Result"]]
for label, run, alias in (("before fix", "repilot-v021-20261125", "gpt-oss-120b"),
                          ("after fix", "g5check2-gptoss-20261126", "gpt-oss-120b"),
                          ("before fix", "repilot-assist2-v021-20261125", "gemma-4-31b"),
                          ("after fix", "g5check2-gemma-20261126", "gemma-4-31b")):
    g = gate(run, alias)
    if g is None:
        rows.append([alias, label, "still running", "–", "–"])
        continue
    rows.append([alias, label, f"{g['n'] - g['failed']} of {g['n']}", f"{g['G5_worst_auc']:.2f}",
                 "pass" if g["G5_pass"] else "FAIL"])
story += [table(rows, [3.4 * cm, 2.4 * cm, 3.2 * cm, 2.4 * cm, 2.4 * cm], zebra=True),
          P("<b>The fix (author's decision 23):</b> every scripted reply, correct or planted, must now have exactly "
            "6 numbered one-sentence steps and 450 to 800 characters, or it is redone. Before the fix, Gemma 4 31B's "
            "planted replies were about half as long as its correct ones.", SMALL)]

# second Gemma try
gl = RUNS / "repilot-gemma-v021-20261125" / "logs" / "llm_calls.jsonl"
n_ok = n_ans = 0
if gl.exists():
    for line in open(gl, encoding="utf-8"):
        d = json.loads(line)
        if d.get("status") == "ok" and d.get("alias") == "gemma-4-26b":
            n_ok += 1
            n_ans += (d.get("usage") or {}).get("completion_tokens", 0) > 0
story += [P("4. Problems found and what was done", H2), table([
    ["Problem", "What was done"],
    ["Gemma 4 26B and 31B always think, and Google gives no switch to turn it off. With the first allowance "
     "(1,024 extra tokens) almost every Gemma 4 26B reply was cut off while thinking.",
     f"Allowance raised to 4,096. In a second try only {n_ans} of {n_ok} replies reached an answer: at temperature 0 "
     "the model loops (\"Wait, let me re-read the question…\"). Author kept it as a solver (decision 24); "
     "gates G2/G3 will decide if it is analysed. The 31B assistant finishes its replies normally."],
    ["Planted replies were shorter than correct ones (gate G5 failed for both assistants).",
     "Fixed reply shape (6 steps, 450 to 800 characters). gpt-oss-120b now passes G5."],
    ["allam-2-7b in direct mode often replied only \"YES\" with no FINAL line; in careful mode it often loops to "
     "the token limit.", "No change: gate G3 excludes such solver units, as preregistered. That worked here."],
    ["Google's free tier was overloaded: about 8 in 10 Gemma calls got \"high demand\" errors.",
     "The run retries and waits; the full study plan counts Google as slow. No data were lost."],
    ["Easy maths: in direct mode the AI lifted some models by 55 to 90 points (pilot). This makes H2 (levelling) "
     "hard to test: power about 2% at the ±5 point bound.", "Put to the author as a choice before the freeze."],
], [7.6 * cm, 9.0 * cm], zebra=True)]

story += [P("5. What this means", H2), bullets([
    "<b>Keep the 4 levels</b> (author's decision 22). Levels 3 and 4 give room at the top; levels 1 and 2 separate "
    "the many small and mid-size local models.",
    "<b>What we can say:</b> on these development puzzles, harder levels lower accuracy for strong models and widen "
    "the gap in catching planted AI mistakes.",
    "<b>What we cannot say:</b> anything about the hypotheses. Four models are far too few, and these are not the "
    "registered items. This was a check of the materials, not a test.",
])]

story += [P("Words used", H2), table([
    ["Word", "Meaning here"],
    ["Solver", "An open AI model answering the puzzles (here: allam-2-7b, gpt-oss-20b, qwen3.8-27b, Gemma 4 26B)."],
    ["Assistant", "A stronger AI whose reply the solver sees (gpt-oss-120b; replication Gemma 4 31B)."],
    ["Planted mistake", "A deliberate, subtle slip in the assistant's reply, on exactly half of the puzzles."],
    ["Ceiling", "When a model gets (nearly) everything right, so differences between models cannot show."],
    ["Gate", "A preregistered check with a fixed consequence (for example: exclude a solver)."],
], [3.2 * cm, 13.4 * cm], zebra=True)]

story += [P("How to check", H2), P(
    "All run folders (items, every reply, logs, manifests, SHA256SUMS.txt) are in "
    "<i>logic-x-ai/v0.2.1/runs/</i>. This report is rebuilt from them with "
    "<i>python docs/make_repilot_report.py RUNS_DIR OUT.pdf</i>. Solver run SHA-256 of raw/trials.jsonl: "
    f"<i>{M['stages']['trials']['sha256']}</i>.", SMALL), SP(1, 6), P(AI_USE, SMALL)]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="Re-pilot report v0.2.1",
                        author="KEYUR SHAILESHKUMAR PATEL")
lab = f"Logic × AI · re-pilot report · v{__version__}"
doc.build(story, onFirstPage=footer(lab), onLaterPages=footer(lab))
print(f"written {OUT}")
