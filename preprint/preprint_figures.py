# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Charts and diagrams for the illustrated protocol preprint.
Data charts read the re-pilot table, the pilot manifest and the synthetic results; diagrams
are drawn from the registered design (no data). Called by make_preprint.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# categorical order (fixed, never cycled): blue, orange, aqua, yellow, then violet for accents
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#257de0", "#f47230", "#00b894", "#f2a600", "#6a4ce2", "#ec4647"
INK, MUTED, GRID, SURF = "#191919", "#666666", "#e6e5e1", "#ffffff"
TINT = {BLUE: "#e6f0fc", ORANGE: "#feeee5", AQUA: "#e0f6f1", YELLOW: "#fdf3dc", VIOLET: "#eeeafc",
        RED: "#fde8e8", MUTED: "#f1f0ed"}
SOLVERS = [("allam-2-7b", BLUE), ("gemma-4-26b", ORANGE), ("qwen3.8-27b", AQUA), ("gpt-oss-20b", YELLOW)]
LEVELS = [1, 2, 3, 4]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.titlelocation": "left", "axes.titlesize": 10,
                     "axes.titleweight": "bold", "legend.frameon": False})


def _grid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _save(fig, out: Path):
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    return out


def _card(ax, x, y, w, h, title, body, color, title_size=9.5, body_size=7.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
                                fc=TINT[color], ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.115, body, ha="center", va="top", fontsize=body_size, color=INK,
            linespacing=1.35)


def _arrow(ax, a, b, color=MUTED, style="-|>", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def design_flow(out: Path):
    """Figure 1: one solver's path through the study."""
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _card(ax, 0.00, 0.58, 0.20, 0.38, "1. Solvers",
          "28 open-weight\nmodels\n× 3 reasoning modes\n= 80 solver units", BLUE)
    _card(ax, 0.26, 0.58, 0.22, 0.38, "2. Measured alone",
          "Logic battery\n(syllogisms, if-then,\nreflection)\n+ knowledge (MMLU)\n+ model size", VIOLET)
    _card(ax, 0.54, 0.58, 0.20, 0.38, "3. Tasks alone",
          "Easy arithmetic\n(200)\nHard deduction\n(300, 4 levels)", AQUA)
    _card(ax, 0.80, 0.58, 0.20, 0.38, "4. Same tasks\nwith the AI",
          "\n\nAssistant:\ngpt-oss-120b\n(replication:\nGemma 4 31B)", ORANGE)
    for a, b in [((0.20, 0.77), (0.26, 0.77)), ((0.48, 0.77), (0.54, 0.77)), ((0.74, 0.77), (0.80, 0.77))]:
        _arrow(ax, a, b)
    # branch: planted vs correct
    _card(ax, 0.48, 0.06, 0.22, 0.36, "AI is right",
          "hard items, half of them:\nthe helper's reply\nis correct", AQUA, body_size=7.8)
    _card(ax, 0.78, 0.06, 0.22, 0.36, "AI is wrong",
          "hard items, other half:\none subtle planted slip,\nsame look and length", RED, body_size=7.8)
    _arrow(ax, (0.86, 0.58), (0.62, 0.42), rad=0.15)
    _arrow(ax, (0.90, 0.58), (0.89, 0.42))
    _card(ax, 0.00, 0.06, 0.40, 0.36, "Recorded for every solver",
          "AI gain = score with AI − score alone\n(on the same items)\n"
          "Error catch (d′) = flags wrong replies,\nnot right ones\n"
          "Every request and reply hashed (SHA-256)", MUTED, body_size=7.6)
    _arrow(ax, (0.48, 0.24), (0.40, 0.24))
    return _save(fig, out)


def hypotheses_idea(out: Path):
    """Figure 2: what each hypothesis predicts (illustration, not data)."""
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9))
    x = np.linspace(-2, 2, 50)
    specs = [("H1  Amplification", "Hard items, AI can be wrong", 6 * x + 18, BLUE,
              "Higher logic → bigger AI gain"),
             ("H2  Levelling", "Easy, checkable items", 0.8 * x + 25, AQUA,
              "Logic adds little (< +5 pp per SD)"),
             ("H4  Mechanism", "Why H1 happens", None, ORANGE, "")]
    for ax, (t, sub, y, c, note) in zip(axes, specs):
        ax.set_title(t, fontsize=9.5)
        if y is not None:
            ax.plot(x, y, color=c, lw=2.2, solid_capstyle="round")
            ax.set_xlabel("Logic score (SD)")
            ax.set_ylabel("AI gain (pp)")
            ax.set_xticks([-2, 0, 2])
            ax.set_yticks([])
            ax.set_ylim(-5, 60)
            ax.text(0.03, 0.97, sub, transform=ax.transAxes, va="top", fontsize=7.6, color=MUTED)
            ax.text(0.03, 0.86, note, transform=ax.transAxes, va="top", fontsize=7.6, color=INK)
            _grid(ax)
        else:
            ax.axis("off")
            ax.text(0.5, 0.96, sub, transform=ax.transAxes, ha="center", va="top", fontsize=7.8, color=MUTED)
            for i, (lab, col) in enumerate([("Logic\nscore", VIOLET), ("Catches the\nAI's error (d′)", ORANGE),
                                           ("AI gain on\nhard items", BLUE)]):
                yy = 0.74 - i * 0.29
                ax.add_patch(FancyBboxPatch((0.18, yy - 0.085), 0.64, 0.17, boxstyle="round,pad=0.01",
                                            fc=TINT[col], ec=col, lw=1.3, transform=ax.transAxes))
                ax.text(0.5, yy, lab, ha="center", va="center", fontsize=7.8, transform=ax.transAxes)
                if i < 2:
                    ax.annotate("", xy=(0.5, yy - 0.205), xytext=(0.5, yy - 0.09), xycoords="axes fraction",
                                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    fig.text(0.01, -0.03, "Illustration of the predictions only. These lines are not data.", fontsize=7.5,
             color=MUTED, style="italic")
    fig.tight_layout()
    return _save(fig, out)


def repilot_alone(tab, out: Path):
    """Figure 3: re-pilot accuracy alone by difficulty level."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), sharey=True)
    for ax, mode in zip(axes, ("direct", "careful")):
        for s, c in SOLVERS:
            ys = []
            for lv in LEVELS:
                r = tab[(tab.solver == s) & (tab["mode"] == mode) & (tab.cond == "alone") & (tab.level == lv)]
                ys.append(100 * r.accuracy.iloc[0] if len(r) else np.nan)
            ax.plot(LEVELS, ys, color=c, lw=2, marker="o", ms=5.5, mec=SURF, mew=1.2, label=s)
            if not np.isnan(ys[-1]):
                ax.annotate(f"{ys[-1]:.0f}%", (4, ys[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color=INK)
        ax.set_title(f"{mode.capitalize()} mode")
        ax.set_xticks(LEVELS, ["L1\n8 stmts", "L2\n12", "L3\n16", "L4\n20"])
        ax.set_xlim(0.8, 4.6)
        ax.set_ylim(-3, 105)
        ax.axhline(33.3, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.text(0.85, 35.5, "chance (1 in 3)", fontsize=7, color=MUTED)
        _grid(ax)
    axes[0].set_ylabel("Correct when working alone (%)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.05, -0.28), ncol=4, fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


def planted(tab, out: Path):
    """Figure 4: on planted-error items, flagged vs copied, careful mode."""
    sol = [("allam-2-7b", BLUE), ("qwen3.8-27b", AQUA), ("gpt-oss-20b", YELLOW)]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8), sharey=True)
    w = 0.36
    for ax, (s, _c) in zip(axes, sol):
        f, cp = [], []
        for lv in LEVELS:
            r = tab[(tab.solver == s) & (tab["mode"] == "careful") & (tab.cond == "AI wrong (planted)")
                    & (tab.level == lv)]
            f.append(100 * r.flagged_error.iloc[0] if len(r) else np.nan)
            cp.append(100 * r.copied_ai_wrong.iloc[0] if len(r) else np.nan)
        xs = np.arange(4)
        ax.bar(xs - w / 2 - 0.01, f, w, color=BLUE, label="Flagged the planted error")
        ax.bar(xs + w / 2 + 0.01, cp, w, color=ORANGE, label="Copied the AI's wrong answer")
        ax.set_xticks(xs, ["L1", "L2", "L3", "L4"])
        ax.set_title(s, fontsize=9)
        ax.set_ylim(0, 105)
        _grid(ax)
    axes[0].set_ylabel("% of planted-error items")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    return _save(fig, out)


def power(out: Path):
    """Figure 5: simulated power for H1 with pilot variance (values from the preregistration)."""
    eff, pw = [3, 5, 8], [52, 86, 99.7]
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.plot(eff, pw, color=BLUE, lw=2, marker="o", ms=6, mec=SURF, mew=1.2)
    for e, p in zip(eff, pw):
        ax.annotate(f"{p:g}%", (e, p), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(80, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(8.1, 81.5, "80% target", fontsize=7, color=MUTED, ha="right")
    ax.set_xlabel("True effect of logic (pp of AI gain per SD)")
    ax.set_ylabel("Chance to detect it (%)")
    ax.set_ylim(40, 108)
    ax.set_xticks([3, 5, 8])
    ax.set_title("Power for H1 (simulated)", fontsize=9.5)
    _grid(ax)
    fig.tight_layout()
    return _save(fig, out)


def g5(before, gemma, after, out: Path):
    """Figure 6: surface-cue check AUC before/after the fixed reply shape."""
    labels = ["gpt-oss-120b\nre-pilot", "Gemma 4 31B\nre-pilot", "gpt-oss-120b\nfixed shape"]
    vals = [before, gemma, after]
    cols = [ORANGE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.bar(range(3), vals, 0.55, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v - 0.012, f"{v:.2f}", ha="center", va="top", fontsize=8.5, color="white",
                fontweight="bold")
    ax.axhline(0.65, color=RED, lw=1.1, ls=(0, (4, 3)))
    ax.text(2.93, 0.655, "limit .65", fontsize=7, color=RED, ha="right", va="bottom")
    ax.axhline(0.5, color=MUTED, lw=0.7)
    ax.text(2.93, 0.505, "0.50 =\nno cue", fontsize=7, color=MUTED, va="bottom", ha="right")
    ax.set_xticks(range(3), labels, fontsize=7.5)
    ax.set_ylim(0.4, max(vals) + 0.1)
    ax.set_xlim(-0.45, 2.95)
    ax.set_ylabel("Worst AUC (lower is better)")
    ax.set_title("Can wrong replies be spotted by looks?", fontsize=9.5)
    _grid(ax)
    fig.tight_layout()
    return _save(fig, out)


def synthetic(ca, cn, out: Path):
    """Figure 7: pipeline check on simulated worlds."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.7))
    worlds = ["Planted-effect\nworld", "No-effect\nworld"]
    h1 = [ca["H1"]["estimate_pp_per_sd"], cn["H1"]["estimate_pp_per_sd"]]
    h4p = [ca["H4"]["p_holm"], cn["H4"]["p_holm"]]
    h1p = [ca["H1"]["p_raw"], cn["H1"]["p_raw"]]
    ax = axes[0]
    ax.bar([0, 1], h1, 0.5, color=[BLUE, MUTED])
    ax.axhline(0, color=INK, lw=0.8)
    for i, (v, p) in enumerate(zip(h1, h1p)):
        ax.text(i, v + (1 if v >= 0 else -1), f"{v:+.1f} pp/SD\np = {p:.3f}".replace("0.", "."),
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax.set_xticks([0, 1], worlds)
    ax.set_ylabel("H1 estimate (pp per SD)")
    ax.set_ylim(min(h1) - 7, max(h1) + 8)
    ax.set_title("H1 amplification", fontsize=9.5)
    _grid(ax)
    ax = axes[1]
    ax.bar([0, 1], [-np.log10(max(p, 1e-4)) for p in h4p], 0.5, color=[ORANGE, MUTED])
    ax.axhline(-np.log10(0.05), color=RED, lw=1.1, ls=(0, (4, 3)))
    ax.text(1.35, -np.log10(0.05) + 0.05, "p = .05", fontsize=7, color=RED, ha="right", va="bottom")
    for i, p in enumerate(h4p):
        ax.text(i, -np.log10(max(p, 1e-4)) + 0.05, f"Holm p = {p:.3f}".replace("0.", "."), ha="center",
                va="bottom", fontsize=7.5)
    ax.set_xticks([0, 1], worlds)
    ax.set_ylabel("Strength of evidence\n(−log10 p)")
    ax.set_ylim(0, 3)
    ax.set_title("H4 mechanism", fontsize=9.5)
    _grid(ax)
    fig.text(0.01, -0.04, "Simulated models with known truth. These are pipeline checks, not study results.",
             fontsize=7.5, color=MUTED, style="italic")
    fig.tight_layout()
    return _save(fig, out)


def gates(out: Path):
    """Figure 8: the seven known-answer gates."""
    g = [("G1", "Parser reads\nknown replies", "if not:\nrun stops", RED),
         ("G2", "No answer\ncollapse", "if not:\nsolver out", ORANGE),
         ("G3", "Answers\nreadable", "if not:\nsolver out", ORANGE),
         ("G4", "Planted replies\nreally wrong", "if not:\nitem out", YELLOW),
         ("G5", "No look-alike\ncues", "if not:\ncaution label", BLUE),
         ("G6", "Error flags\nare valid", "if not: H4\nuninformative", VIOLET),
         ("G7", "Logic score\nis valid", "if not:\ncaution label", AQUA)]
    fig, ax = plt.subplots(figsize=(8.2, 2.1))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (k, what, then, c) in enumerate(g):
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.12), 0.86, 0.80, boxstyle="round,pad=0.01,rounding_size=0.06",
                                    fc=TINT[c], ec=c, lw=1.3))
        ax.text(i + 0.49, 0.82, k, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(i + 0.49, 0.58, what, ha="center", va="center", fontsize=7, color=INK, linespacing=1.2)
        ax.text(i + 0.49, 0.28, then, ha="center", va="center", fontsize=6.6, color=MUTED, style="italic")
    return _save(fig, out)


def timeline(out: Path):
    """Figure 9: where the study is now."""
    steps = [("Oct 2026", "Idea and design", True), ("Nov 2026", "Synthetic checks", True),
             ("Nov 2026", "Pilot + re-pilot\n(dev items)", True), ("27 Nov 2026", "Frozen and\nregistered", True),
             ("Next", "Main data\ncollection", False), ("After", "Results article\n(any direction)", False)]
    fig, ax = plt.subplots(figsize=(8.2, 1.6))
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0, len(steps) - 1], [0.62, 0.62], color=GRID, lw=3, zorder=1)
    ax.plot([0, 3], [0.62, 0.62], color=BLUE, lw=3, zorder=2)
    for i, (when, what, done) in enumerate(steps):
        ax.scatter([i], [0.62], s=150, color=BLUE if done else SURF, edgecolor=BLUE, lw=2, zorder=3)
        ax.text(i, 0.88, when, ha="center", fontsize=7.5, color=MUTED)
        ax.text(i, 0.40, what, ha="center", va="top", fontsize=7.8, color=INK, linespacing=1.2,
                fontweight="bold" if i == 3 else "normal")
    return _save(fig, out)
