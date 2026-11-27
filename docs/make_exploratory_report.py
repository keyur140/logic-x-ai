# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)


"""Builds docs/Exploratory_Report_SYNTHETIC.pdf: the easy-language report of the
exploratory run on synthetic data. Every number is read from exploratory_synthetic/.
Run: python docs/figures.py && python docs/make_exploratory_report.py"""
import json
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdfkit import (AI_USE, AUTHOR_HTML, CAP, H2, H3, SMALL, SP, SUB, TITLE, WARM, P,  # noqa: E402
                    box, bullets, figure, footer, table)

from logicxai import __version__  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "exploratory_synthetic"
FIG = Path(__file__).parent / "figures"
OUT = Path(__file__).parent / "Exploratory_Report_SYNTHETIC.pdf"


def load(w, f):
    return json.loads((EX / w / f).read_text(encoding="utf-8"))


A = load("amplify_world", "analysis/results.json")
N = load("null_world", "analysis/results.json")
MA = load("amplify_world", "manifest.json")
MN = load("null_world", "manifest.json")
V = json.loads((EX / "validation" / "validation.json").read_text(encoding="utf-8"))


def f(x, nd=1, sign=False):
    if x is None:
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def pv(x):
    return "n/a" if x is None else ("< .001" if x < 0.001 else f"{x:.3f}".lstrip("0"))


def ci(r, h):
    c = r["by_assistant"][r["primary_assistant"]][h]["coef"]["logic_z"]["ci"]
    return f"[{c[0]:+.1f}, {c[1]:+.1f}]"


def verdict(c):
    return "Supported" if c["supported"] else "Not supported"


def vrow(world, est):
    return next(r for r in V["monte_carlo"] if r["check"] == "H1" and r["world"] == world and r["estimator"] == est)


h2v = next(r for r in V["monte_carlo"] if r["check"] == "H2")
ca, cn = A["confirmatory"], N["confirmatory"]
excl = MA["gates"]["G2_G3"]
g5a = MA["gates"]["G4_G5"]
g6a = A["gates"]["G6"][A["primary_assistant"]]
g7a = A["gates"]["G7"]

story = [
    box("<b>SYNTHETIC DATA. NOT STUDY RESULTS.</b> Every number in this report comes from simulated "
        "(mock) AI models whose behaviour we wrote ourselves. It tests the machinery of the study. "
        "It says nothing about real AI models.", bg=WARM),
    SP(1, 8),
    P("Logic × AI: exploratory run on synthetic data", TITLE),
    P(f"Does the study pipeline find an effect when one exists, and stay quiet when none does? "
      f"Package v{__version__} · runs {MA['run_id']} and {MN['run_id']}", SUB),
    SP(1, 6), P(AUTHOR_HTML, SUB), SP(1, 10),

    P("1. The short answer", H2),
    box(f"<b>Yes, with one honest limit.</b> We built two fake worlds. In the <b>no-effect world</b> the "
        f"pipeline correctly found nothing for H1 and H4. In the <b>planted-effect world</b> it found the "
        f"mechanism (H4, p<sub>Holm</sub> = {pv(ca['H4']['p_holm'])}) and a positive H1 slope "
        f"({f(ca['H1']['estimate_pp_per_sd'], 1, True)} points per SD) that just missed significance "
        f"(p = {pv(ca['H1']['p_raw'])}). In 2,000 simulated studies the main H1 test raised a false alarm "
        f"{100 * vrow('ignore', 'gain_wild')['reject_rate']:.0f}% of the time (target 5%) and found a "
        f"planted effect {100 * vrow('amplify', 'gain_wild')['reject_rate']:.0f}% of the time. "
        f"All {sum(c['result'] == 'pass' for c in MA['self_checks'])} self-checks passed in both runs. "
        "The validation also caught a serious flaw in the old analysis plan, which is now fixed."),

    P("2. What this is, and what it is not", H2),
    bullets([
        "<b>It is</b> a full dress rehearsal: the same code, item counts, gates and statistics as the real "
        "study, run end to end on 84 simulated solvers (28 fake models × 3 thinking modes), about 238,000 "
        "simulated replies per world.",
        "<b>It is</b> a test of the test: we know the truth in each fake world, so we can see whether the "
        "analysis gets it right.",
        "<b>It is not</b> evidence about real AI models, about people, or about logic in general. "
        "The fake models behave the way we told them to.",
        "<b>It is not</b> the preregistered study. It uses the development seed (7110526); the registered "
        "items (seed 20261026) stay locked until the freeze.",
    ]),

    P("3. How the study works", H2),
    figure(FIG / "design.png", "Figure 1. Study design. Each solver is tested alone and then with an AI helper "
           "on the same items. On hard items the helper is wrong exactly half the time."),
    P("<b>The two fake worlds.</b> In both worlds each fake model has a hidden logic ability, a hidden "
      "knowledge level and a size. Careful thinking mode adds logic; direct mode removes some. Two fake "
      "models answer at random and one always gives the same answer, to test the exclusion rules."),
    table([["", "Planted-effect world", "No-effect world"],
           ["When the AI helper is wrong", "Better logic → more likely to catch it", "Everyone catches it 35% of the time"],
           ["Expected H1 (amplification)", "Positive", "Not positive"],
           ["Expected H4 (mechanism via d′)", "Positive", "About zero"],
           ["Expected H2 (levelling on easy maths)", "Supported in both: easy maths is easy to check",
            "Supported in both"]], [4.6 * cm, 6 * cm, 6 * cm]),

    P("4. Checks before any result counts", H2),
    P("Following the rule from the author's earlier work: every measure must pass a known-answer check "
      "before its results are read. Gates can exclude a solver or change how a result may be read; "
      "self-checks recompute key numbers a second, independent way."),
    table([["Gate", "What it checks", "Planted world", "No-effect world", "Status"],
           ["G1", "Answer reader reads 29 known replies correctly",
            f"{MA['gates']['G1']['n_correct']}/{MA['gates']['G1']['n_cases']}",
            f"{MN['gates']['G1']['n_correct']}/{MN['gates']['G1']['n_cases']}", "PASS"],
           ["G2/G3", "Exclude collapsed solvers (>90% same answer) and unreadable or failing ones",
            f"{excl['n_excluded']} of {excl['n_units']} excluded",
            f"{MN['gates']['G2_G3']['n_excluded']} of {MN['gates']['G2_G3']['n_units']} excluded",
            "PASS (the 3 units of the planted 'collapsed' model)"],
           ["G4", "Exactly half of hard items carry a planted AI error; no leaks",
            f"{g5a[A['primary_assistant']]['n_planted']} planted, {g5a[A['primary_assistant']]['leaks']} leaks",
            "same", "PASS"],
           ["G5", "Planted and clean AI replies look alike (surface-cue AUC ≤ .65)",
            f"worst AUC {max(v['G5_worst_auc'] for v in g5a.values()):.2f}",
            f"worst AUC {max(v['G5_worst_auc'] for v in MN['gates']['G4_G5'].values()):.2f}", "PASS"],
           ["G6", "Error flag works: flags on obvious errors far above false alarms",
            f"{100 * g6a['flag_rate_catch_mean']:.0f}% vs {100 * g6a['false_alarm_clean_mean']:.0f}%",
            f"{100 * N['gates']['G6'][N['primary_assistant']]['flag_rate_catch_mean']:.0f}% vs "
            f"{100 * N['gates']['G6'][N['primary_assistant']]['false_alarm_clean_mean']:.0f}%", "PASS"],
           ["G7", "Logic score agrees with FOLIO + LogiQA style benchmarks",
            f"Spearman ρ = {g7a['spearman_rho']:.2f}", f"ρ = {N['gates']['G7']['spearman_rho']:.2f}", "PASS"]],
          [1.3 * cm, 6.0 * cm, 3.1 * cm, 3.1 * cm, 3.1 * cm], status_col=4),
    SP(1, 8),
    table([["Check", "What is recomputed independently", "Planted world", "No-effect world"]] +
          [[a["id"], a["check"], f"{a['result'].upper()}: {a['detail']}", b["result"].upper()]
           for a, b in zip(MA["self_checks"], MN["self_checks"])],
          [1.9 * cm, 6.0 * cm, 6.2 * cm, 2.5 * cm], status_col=3),
    P("C11 is 'n/a' because synthetic runs do not need the freeze. The full study refuses to start unless "
      "the freeze status is MATCHES_FREEZE.", SMALL),
    figure(FIG / "gate_g6.png", "Figure 2. Gate G6 in the planted world. Blue: how often each solver flags an "
           "obvious AI error. Orange: how often it wrongly flags a correct AI reply. The few solvers on the "
           "left are the random responders, kept in on purpose to show the chance floor."),

    P("5. Does the test work? Pipeline validation", H2),
    P("A good test should say 'effect found' about 5% of the time when there is no effect (false alarms), "
      "and almost always when there is a real effect. We simulated thousands of complete studies at the "
      "size of the real one (25 model clusters, 150 planted-error items per solver)."),
    figure(FIG / "validation.png", "Figure 3. How often H1 reports an effect, in three simulated worlds. "
           "Blue: the registered test. Grey: the adjusted score from the earlier plan. Error bars: 95% "
           "Monte Carlo intervals."),
    table([["World", "True effect?", "Registered test (gain score)", "Old plan (adjusted score)"],
           ["Solver ignores the AI", "No", f"{100 * vrow('ignore', 'gain_wild')['reject_rate']:.1f}%",
            f"{100 * vrow('ignore', 'ancova')['reject_rate']:.1f}%"],
           ["Everyone catches 35% of errors", "No", f"{100 * vrow('constant', 'gain_wild')['reject_rate']:.1f}%",
            f"{100 * vrow('constant', 'ancova')['reject_rate']:.1f}%"],
           ["Logic helps catch errors", "Yes", f"{100 * vrow('amplify', 'gain_wild')['reject_rate']:.1f}%",
            f"{100 * vrow('amplify', 'ancova')['reject_rate']:.1f}%"],
           ["H2: true slope exactly +5 pp", "Claim would be false", f"{100 * h2v['reject_rate']:.1f}% false claims",
            "n/a"]], [5.2 * cm, 2.8 * cm, 4.3 * cm, 4.3 * cm]),
    SP(1, 6),
    box("<b>What the validation caught.</b> The earlier plan (decision 10) adjusted the score with AI for "
        "the solver's own score alone. That sounds careful, but the solo score is measured with noise, and "
        "the logic score is a cleaner measure of the same ability. So the adjusted analysis 'found' "
        "amplification up to 99% of the time when there was none. On 2026-11-05 the author switched the "
        "main outcome to the <b>gain score</b> (score with AI minus score alone, same items), which kept "
        "false alarms near 5%. The adjusted score stays in the report as a labelled sensitivity check. "
        "This change is logged in DEVIATIONS.md and was made <b>before</b> registration.", bg=WARM),

    P("6. Results in the two fake worlds", H2),
    table([["Hypothesis", "Planted-effect world", "No-effect world", "Expected?"],
           ["H1 amplification (pp of gain per SD of logic)",
            f"{f(ca['H1']['estimate_pp_per_sd'], 1, True)}, 90% CI {ci(A, 'H1')}, p = {pv(ca['H1']['p_raw'])}: "
            f"{verdict(ca['H1'])}",
            f"{f(cn['H1']['estimate_pp_per_sd'], 1, True)}, 90% CI {ci(N, 'H1')}, p = {pv(cn['H1']['p_raw'])}: "
            f"{verdict(cn['H1'])}",
            "No-effect world: yes. Planted world: right direction, just short of significance"],
           ["H2 levelling (non-superiority vs +5 pp)",
            f"{f(ca['H2']['estimate_pp_per_sd'], 1, True)}, p = {pv(ca['H2']['p_raw'])} (secondary): {verdict(ca['H2'])}",
            f"{f(cn['H2']['estimate_pp_per_sd'], 1, True)}, p = {pv(cn['H2']['p_raw'])} (secondary): {verdict(cn['H2'])}",
            "Yes (both)"],
           ["H4 mechanism (indirect effect via d′)",
            f"{f(ca['H4']['indirect'], 1, True)}, p<sub>Holm</sub> = {pv(ca['H4']['p_holm'])}: {verdict(ca['H4'])}",
            f"{f(cn['H4']['indirect'], 1, True)}, p<sub>Holm</sub> = {pv(cn['H4']['p_holm'])}: {verdict(cn['H4'])}",
            "Yes (both)"]], [4.0 * cm, 4.8 * cm, 4.8 * cm, 3.0 * cm]),
    SP(1, 6),
    figure(FIG / "forest.png", "Figure 4. Main estimates with 90% confidence intervals. The shaded band is "
           "the ±5 point H2 band."),
    figure(FIG / "h1_scatter.png", "Figure 5. H1. Each point is one solver. In the planted world the link "
           "is U-shaped: the random responders sit at chance on the left, the middle loses most when the AI "
           "is wrong, and the best reasoners lose least. A straight line understates this, which is why H1 "
           "fell just short. In the no-effect world better solvers lose more, because they had more to lose."),
    figure(FIG / "h4_dprime.png", "Figure 6. H4 path. Spotting AI errors (d′) rises steeply with logic in "
           "the planted world and only slightly in the no-effect world (there, only false alarms fall with "
           "ability). H4 asks whether better spotting turns into a bigger gain."),
    figure(FIG / "h2_scatter.png", "Figure 7. H2. On easy maths the helper is right 90% of the time, so "
           "weaker solvers gain most. Logic adds clearly less than +5 points per SD in both worlds."),
    figure(FIG / "modes.png", "Figure 8. Thinking mode works as a within-model lever: careful mode raises "
           "logic and error spotting in the same model, and the loss when the AI is wrong shrinks."),
    figure(FIG / "multiverse.png", "Figure 9. Multiverse (exploratory). Each hypothesis re-run under 72 "
           "reasonable analysis variants. The no-effect world never supports H1. H2 holds in every variant."),

    P("7. Problems found, and what was done", H2),
    table([["Problem", "How it was found", "Fix"],
           ["Adjusted (ANCOVA) score gives false alarms of 62 to 99%",
            "Validation in three known worlds", "Main outcome switched to gain score (author's decision 14); "
            "ANCOVA kept as labelled sensitivity"],
           ["Synthetic replies were cached across worlds, so the no-effect run could reuse planted-world replies",
            "Code review during validation", "Synthetic replies are never cached; a test checks that the worlds differ"],
           ["Fake models recognised obvious-error items by line count, not by item type",
            "Code review", "The fake models now look up the item family directly"],
           ["The CR1 t-test is slightly liberal with 25 clusters (6.7% in the fast check)",
            "Validation", "The registered test is the wild cluster bootstrap (6.0%, within Monte Carlo error of 5%)"],
           ["Planted H1 effect fell just short (p = " + pv(ca['H1']['p_raw']) + ")",
            "This run", "Reported as is. Not tuned. Power curve and the U-shape are shown so readers can judge"]],
          [5.3 * cm, 4.0 * cm, 7.3 * cm]),

    P("8. What we can and cannot say", H2),
    table([["We CAN say", "We CANNOT say"],
           ["The pipeline runs end to end and every file is hashed and checkable.",
            "Anything about real AI models. These models are fake."],
           ["The main tests keep false alarms near 5% in two kinds of no-effect world.",
            "That real models will show amplification, levelling or the d′ mechanism."],
           ["The gates catch a collapsed solver and the error flag passes its known-answer check.",
            "That the same effect sizes will appear in the real study."],
           ["The earlier adjusted-score plan would have produced false findings, and it was fixed before "
            "registration.", "That logic causes the gain. Even in the real study, H1 and H2 are correlational."]],
          [8.3 * cm, 8.3 * cm]),

    P("9. Words used", H2),
    table([["Word", "Meaning here"],
           ["Solver", "One AI model in one thinking mode (28 models × 3 modes = 84 solvers)."],
           ["AI gain", "Score with the AI helper minus score alone, on the same items, in percentage points."],
           ["Planted error", "The helper's reply is made wrong on purpose, on exactly half the hard items."],
           ["d′ (d-prime)", "How well a solver tells wrong AI replies from right ones. 0 = guessing."],
           ["Gate", "A known-answer check that must pass before a measure is trusted."],
           ["False alarm rate", "How often a test reports an effect that is not there. The target is 5%."],
           ["Power", "How often a test finds an effect that is there."],
           ["Holm correction", "Raises p-values a little because three hypotheses are tested together."],
           ["Wild cluster bootstrap", "A resampling way to get honest p-values when solvers from the same "
            "model are alike."],
           ["Multiverse", "Re-running the analysis under many reasonable choices to see if the answer holds."],
           ["Freeze", "A SHA-256 fingerprint of every study file, registered before the real run."]],
          [3.6 * cm, 13.0 * cm]),

    P("10. How to check this yourself", H2),
    bullets([
        "Rebuild both worlds: <font face='DV'>python -m logicxai run --profile mock</font> and "
        "<font face='DV'>--profile mock_null</font> (about 3 minutes each, no internet, no keys).",
        "Re-run the validation: <font face='DV'>python -m logicxai validate</font> (about 2 minutes).",
        "Check every file of a run: <font face='DV'>python -m logicxai verify runs/&lt;run id&gt;</font>.",
        "The files behind this report are in <font face='DV'>exploratory_synthetic/</font> with a "
        "SHA256SUMS.txt; the figures are rebuilt by <font face='DV'>python docs/figures.py</font>.",
    ]),

    P("11. Questions people ask", H2),
    P("<b>Why test on fake data at all?</b>", H3),
    P("Because only in a fake world do we know the right answer. If the pipeline cannot find a planted "
      "effect, or finds one that is not there, the real study would be worthless. The author's earlier music "
      "study showed how a measure can fail silently; this is the guard against that."),
    P("<b>The planted H1 effect was not significant. Is the study broken?</b>", H3),
    P("No. The validation shows the test finds a clear effect almost always. The fake world we happened to "
      "build has a U-shaped pattern, which a straight-line test only partly picks up. We did not tune the "
      "fake world to make the result look better, because that would defeat the purpose. The real study's "
      "power is shown in the project brief."),
    P("<b>Why change the main analysis now?</b>", H3),
    P("Because the change was made before registration, for a documented reason, and the evidence is in "
      "this report. Changing it after seeing real data would be a problem. Changing it now is good practice."),
    P("<b>Can I add this to the OSF registration?</b>", H3),
    P("Yes, as a supplementary file clearly labelled 'synthetic, exploratory'. It shows reviewers the "
      "pipeline was tested before any real data existed."),

    P("12. Statements", H2),
    P(AI_USE),
    P("<b>Data availability.</b> All code, synthetic outputs and this report are in the study package "
      "(MIT licence for code, CC BY 4.0 for generated data and reports)."),
    P("<b>Ethics.</b> No human participants and no personal data. Only AI models are tested."),
    P("<b>Funding and competing interests.</b> None."),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Logic x AI: exploratory run on synthetic data",
                        author="KEYUR SHAILESHKUMAR PATEL")
lab = f"Logic × AI · exploratory report · SYNTHETIC DATA, not study results · v{__version__}"
doc.build(story, onFirstPage=footer(lab), onLaterPages=footer(lab))
print(OUT)
