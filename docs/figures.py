# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)


"""Charts for the exploratory report and the project brief.
Reads exploratory_synthetic/ (synthetic data only) and writes docs/figures/*.png.
Run: python docs/figures.py"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EX = ROOT / "exploratory_synthetic"
OUT = Path(__file__).parent / "figures"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
WORLD = {"amplify_world": ("Planted effect world", BLUE), "null_world": ("No-effect world", ORANGE)}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.titlelocation": "left", "axes.titlesize": 10})


def res(w):
    return json.loads((EX / w / "analysis" / "results.json").read_text())


def sl(w):
    return pd.read_csv(EX / w / "analysis" / "solver_level_analysed.csv")


def prim(w):
    return res(w)["primary_assistant"]


def _grid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ design
def design():
    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def b(x, y, w, h, title, body, fc="#eef4fc", ec=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=8.6, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=7.2, color=MUTED, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1})

    b(1, 24, 20, 18, "80 solver units", "28 free open models\n× 3 thinking modes\n(2 thinking models pooled)")
    b(26, 24, 22, 18, "1. Logic score", "Solve alone:\nsyllogisms, conditionals,\nreflection puzzles\n(keys proved by code)")
    b(53, 24, 22, 18, "2. Solve alone", "Easy maths\nHard deduction\n(same items as step 3)")
    b(80, 24, 19, 18, "3. Solve with AI", "gpt-oss-120b helper\n(Gemma 4 31B replication)\nflags: ASSISTANT_ERROR")
    for x1, x2 in ((21.8, 25.4), (48.8, 52.4), (75.8, 79.4)):
        arrow(x1, 33, x2, 33)
    b(80, 1, 19, 17, "Planted errors", "Hard items: exactly 50%\nhave a wrong AI answer\n(checked by SAT solver)",
      fc="#fdf0ea", ec=ORANGE)
    arrow(89.5, 18.8, 89.5, 23.3)
    b(53, 1, 22, 17, "AI gain", "score with AI − score alone\n(same items, per solver)", fc="#e8f7f1", ec=AQUA)
    arrow(64, 23.3, 64, 18.8)
    arrow(81, 23.6, 74, 18.6)
    b(1, 1, 47, 17, "Tests (preregistered, Holm-corrected)",
      "H1 amplification: logic → bigger gain when the AI is wrong\n"
      "H2 levelling: logic adds < 5 pp per SD on easy tasks\n"
      "H4 mechanism: logic → spotting errors (d′) → gain\n"
      "Controls: knowledge (MMLU) + model size; clustered by model", fc="#f4f4f2", ec=GREY)
    arrow(52.4, 9.5, 48.8, 9.5)
    save(fig, "design.png")


# -------------------------------------------------------------- validation
def validation():
    v = json.loads((EX / "validation" / "validation.json").read_text())
    rows = [r for r in v["monte_carlo"] if r["check"] == "H1"]
    worlds = ["ignore", "constant", "amplify"]
    labels = ["No effect:\nsolver ignores the AI", "No effect:\nsame catch rate for all",
              "Planted effect:\nlogic helps catch errors"]
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    x = np.arange(3)
    for i, (est, colr, name) in enumerate((("gain_wild", BLUE, "Gain score, registered test (main)"),
                                           ("ancova", GREY, "Adjusted score (old plan, now sensitivity only)"))):
        vals = [next(r for r in rows if r["world"] == w and r["estimator"] == est) for w in worlds]
        rate = np.array([r["reject_rate"] * 100 for r in vals])
        lo = rate - np.array([r["mc_ci95"][0] * 100 for r in vals])
        hi = np.array([r["mc_ci95"][1] * 100 for r in vals]) - rate
        xs = x + (i - 0.5) * 0.38
        ax.bar(xs, rate, 0.36, color=colr, label=name, yerr=[lo, hi],
               error_kw={"ecolor": MUTED, "lw": 0.8, "capsize": 2})
        for xi, r in zip(xs, rate):
            ax.text(xi, r + 6, f"{r:.0f}%", ha="center", fontsize=7.5, color=INK)
    ax.axhline(5, color=ORANGE, lw=0.9, ls="--")
    ax.text(-0.45, 8, "5% = allowed false alarms", fontsize=7, color=MUTED)
    ax.set_xticks(x, labels, fontsize=7.8)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("How often H1 says 'effect found' (%)")
    ax.set_title("A sound test fires about 5% of the time when there is no effect, and often when there is")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2)
    _grid(ax)
    save(fig, "validation.png")


# ------------------------------------------------------- gain versus logic
def gain_scatter(col_tpl, name, title, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    for ax, w in zip(axes, WORLD):
        d = sl(w)
        a = prim(w)
        col = col_tpl.format(a=a)
        lab, colr = WORLD[w]
        mk = {"direct": "o", "brief": "s", "careful": "^"}
        for m, g in d.groupby("mode"):
            ax.scatter(g["logic_score"], g[col], s=20, color=colr, marker=mk[m], edgecolor="white",
                       lw=0.6, alpha=0.9, label=f"{m} mode")
        k, c0 = np.polyfit(d["logic_score"], d[col], 1)
        xs = np.linspace(d["logic_score"].min(), d["logic_score"].max(), 20)
        ax.plot(xs, k * xs + c0, color=INK, lw=1.6)
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_title(lab, color=colr)
        ax.set_xlabel("Logic score (share correct, alone)")
        _grid(ax)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    save(fig, name)


# ------------------------------------------------------------------ forest
def forest():
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ys, labels = [], []
    y = 0
    from scipy.stats import t as tdist
    for h, key in (("H1", "H1"), ("H2", "H2")):
        for w in WORLD:
            r = res(w)
            c = r["by_assistant"][r["primary_assistant"]][key]["coef"]["logic_z"]
            lab, colr = WORLD[w]
            ax.errorbar(c["b"], y, xerr=[[c["b"] - c["ci"][0]], [c["ci"][1] - c["b"]]], fmt="o",
                        color=colr, ms=6, lw=1.6, capsize=3)
            ax.text(c["ci"][1] + 0.8, y, f"{c['b']:+.1f}", va="center", fontsize=7.5, color=INK)
            labels.append(f"{h} · {lab}")
            ys.append(y)
            y += 1
        y += 0.6
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvspan(-5, 5, color="#eef4fc", zorder=0)
    ax.text(5.2, y - 1.2, "H2 band ±5 pp", fontsize=7, color=BLUE)
    ax.set_yticks(ys, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Logic effect on AI gain (percentage points per 1 SD of logic), 90% CI")
    ax.set_title("Main estimates in the two synthetic worlds")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    save(fig, "forest.png")


# ------------------------------------------------------------ mode effects
def modes():
    d = sl("amplify_world")
    a = prim("amplify_world")
    order = ["direct", "brief", "careful"]
    g = d.groupby("mode").agg(logic=("logic_score", "mean"), gain=(f"gain_hard_error__{a}", "mean"),
                              dp=(f"dprime__{a}", "mean")).reindex(order)
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
    for ax, col, t, fmt in ((axes[0], "logic", "Logic score", "{:.2f}"),
                            (axes[1], "dp", "Error spotting (d′)", "{:.2f}"),
                            (axes[2], "gain", "AI gain when AI is wrong (pp)", "{:+.1f}")):
        bars = ax.bar(order, g[col], color=[GREY, "#86b6ef", BLUE], width=0.6)
        ax.bar_label(bars, labels=[fmt.format(v) for v in g[col]], padding=2, fontsize=7.5)
        lo, hi = min(0, g[col].min()), max(0, g[col].max())
        ax.set_ylim(lo * 1.18, hi * 1.18 if hi > 0 else 2)
        ax.set_title(t, fontsize=9)
        ax.axhline(0, color=MUTED, lw=0.7)
        _grid(ax)
    fig.suptitle("Thinking mode changes logic inside the same model (planted effect world)",
                 x=0.01, ha="left", fontsize=10)
    save(fig, "modes.png")


# ---------------------------------------------------------------- gate G6
def g6():
    d = sl("amplify_world")
    a = prim("amplify_world")
    d = d.sort_values(f"catch_flag_rate__{a}").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    x = np.arange(len(d))
    ax.vlines(x, d[f"fa_rate__{a}"] * 100, d[f"catch_flag_rate__{a}"] * 100, color=GRID, lw=1.2)
    ax.scatter(x, d[f"catch_flag_rate__{a}"] * 100, s=12, color=BLUE, label="flags on obvious-error items (catch)")
    ax.scatter(x, d[f"fa_rate__{a}"] * 100, s=12, color=ORANGE, marker="s", label="flags on correct AI replies (false alarm)")
    ax.set_xlabel("Solvers, sorted")
    ax.set_ylabel("Says 'AI made an error' (%)")
    ax.set_title("Gate G6: the error flag means something (catch rate far above false-alarm rate)")
    ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(0.42, 0.42))
    ax.set_xticks([])
    _grid(ax)
    save(fig, "gate_g6.png")


# --------------------------------------------------------------- multiverse
def multiverse():
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    cats, vals, cols = [], [], []
    for h in ("H1", "H2"):
        for w in WORLD:
            mv = res(w).get("multiverse", {}).get(h, {})
            cats.append(f"{h} · {WORLD[w][0]}")
            vals.append(100 * mv.get("share_supporting", 0))
            cols.append(WORLD[w][1])
    y = np.arange(len(cats))
    ax.barh(y, vals, color=cols, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=7.5)
    ax.set_yticks(y, cats, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("Share of 72 analysis variants that support the hypothesis")
    ax.set_title("Multiverse: do the results survive other reasonable analysis choices?")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    save(fig, "multiverse.png")


# -------------------------------------------------------------------- power
def power():
    from logicxai.power import ASSUME, _one
    rng = np.random.default_rng(20261026)
    slopes = [0, 1, 2, 3, 4, 5, 6, 8]
    pw = []
    for s in slopes:
        hits = sum((lambda b, se, q: b / se > q)(*_one(rng, float(s), ASSUME["items_hard_error"], ASSUME))
                   for _ in range(600))
        pw.append(100 * hits / 600)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.plot(slopes, pw, color=BLUE, lw=2, marker="o", ms=5)
    ax.axhline(80, color=ORANGE, lw=0.9, ls="--")
    ax.text(0.1, 83, "80% power", fontsize=7.5, color=MUTED)
    for s, p in zip(slopes, pw):
        if s in (0, 3, 5):
            ax.text(s, p + 5, f"{p:.0f}%", ha="center", fontsize=7.5)
    ax.set_ylim(0, 108)
    ax.set_xlabel("True H1 effect (pp of AI gain per 1 SD of logic)")
    ax.set_ylabel("Chance the study detects it (%)")
    ax.set_title("Planned power for H1 (84 solvers, clustered by 28 models)")
    _grid(ax)
    save(fig, "power.png")


if __name__ == "__main__":
    design()
    validation()
    gain_scatter("gain_hard_error__{a}", "h1_scatter.png",
                 "H1: AI gain on items where the AI is wrong, by logic score",
                 "AI gain (pp), AI wrong")
    gain_scatter("gain_easy_math__{a}", "h2_scatter.png",
                 "H2: AI gain on easy maths, by logic score", "AI gain (pp), easy maths")
    gain_scatter("dprime__{a}", "h4_dprime.png",
                 "H4 path: spotting AI errors (d′) by logic score", "Error spotting d′")
    forest()
    modes()
    g6()
    multiverse()
    power()
    print("figures written to", OUT)
