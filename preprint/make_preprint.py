# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Builds the illustrated protocol preprint (facts only: design, methods, pilot and re-pilot facts; no
confirmatory results exist yet). Numbers are read from the run folders where possible. Every section has a
plain-language box, and charts/diagrams come from preprint_figures.py.

Run: python preprint/make_preprint.py RUNS_DIR PILOT_RUN_DIR OUT.pdf [OSF_REGISTRATION_URL]
This folder is NOT part of the frozen study files (it is written after the freeze)."""
import json
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pdfkit import (AUTHOR_HTML, BODY, H2, H3, SMALL, SP, SUB, TITLE, P, ParagraphStyle, box, bullets,  # noqa: E402
                    figure, footer, table)
import preprint_figures as F  # noqa: E402
from repilot_summary import summarise  # noqa: E402

from logicxai import __version__  # noqa: E402
from logicxai.freeze import freeze_status  # noqa: E402

RUNS, PILOT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
REG = sys.argv[4] if len(sys.argv) > 4 else None
REG_TXT = f'<a href="{REG}" color="blue">{REG}</a>' if REG else "OSF registration (link on the OSF project page)"
FZ = freeze_status(ROOT)
assert FZ["status"] == "MATCHES_FREEZE", FZ["status"]
MAIN = RUNS / "repilot-v021-20261125"
summarise(MAIN)
TAB = pd.read_csv(MAIN / "analysis" / "repilot_by_level.csv")
PM = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
SYN = {w: json.loads((ROOT / "exploratory_synthetic" / w / "analysis" / "results.json").read_text())
       for w in ("amplify_world", "null_world")}
GH = "https://github.com/keyur140/logic-x-ai"
FIG = OUT.parent / (OUT.stem + "_figures")
FIG.mkdir(parents=True, exist_ok=True)
PLAIN = ParagraphStyle("PL", parent=BODY, fontSize=9.2, leading=13.4)


def plain(text):
    """Easy-language summary box shown under each section."""
    return box("<font color='#6a4ce2'><b>In plain words.</b></font> " + text, bg=colors.HexColor("#f1eefd"))


def tiles(items):
    """Row of coloured key-number tiles."""
    big = ParagraphStyle("BIG", parent=BODY, fontName="DVB", fontSize=17, leading=20, alignment=1,
                         textColor=colors.white, spaceAfter=0)
    lab = ParagraphStyle("LAB", parent=BODY, fontSize=7.6, leading=9.6, alignment=1, textColor=colors.white,
                         spaceAfter=0)
    cells = [[P(v, big) for v, _l, _c in items], [P(l, lab) for _v, l, _c in items]]
    t = Table(cells, colWidths=[16.6 * cm / len(items)] * len(items))
    st = [("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
          ("LINEAFTER", (0, 0), (-2, -1), 3, colors.white)]
    for i, (_v, _l, c) in enumerate(items):
        st.append(("BACKGROUND", (i, 0), (i, -1), colors.HexColor(c)))
    t.setStyle(TableStyle(st))
    return t


def half(path, caption):
    """Image + caption sized for one column of a two-up row (no KeepTogether inside table cells)."""
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image
    iw, ih = ImageReader(str(path)).getSize()
    w = 8.0 * cm
    return [Image(str(path), width=w, height=w * ih / iw), P(caption, SMALL)]


def two(a, b):
    t = Table([[a, b]], colWidths=[8.3 * cm, 8.3 * cm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


def pct(x):
    return "–" if pd.isna(x) else f"{100 * x:.0f}%"


def cell(solver, mode, cond, level, col="accuracy"):
    r = TAB[(TAB.solver == solver) & (TAB["mode"] == mode) & (TAB.cond == cond) & (TAB.level == level)]
    return r[col].iloc[0] if len(r) else float("nan")


def gate(run, alias):
    m = json.loads((RUNS / run / "manifest.json").read_text(encoding="utf-8"))
    return m["gates"]["G4_G5"][alias]


def p3(x):
    return "n/a" if x is None else (f"{x:.3f}".lstrip("0") if x >= 0.001 else "< .001")


g5_before = gate("repilot-v021-20261125", "gpt-oss-120b")["G5_worst_auc"]
g5_after = gate("g5check2-gptoss-20261126", "gpt-oss-120b")["G5_worst_auc"]
g5_gemma = gate("repilot-assist2-v021-20261125", "gemma-4-31b")["G5_worst_auc"]
ca, cn = SYN["amplify_world"]["confirmatory"], SYN["null_world"]["confirmatory"]
excl = PM["gates"]["G2_G3"]

story = [box("<font color='white' size='8'><b>STUDY PROTOCOL · PREREGISTERED · NO RESULTS YET</b></font>",
             bg=colors.HexColor("#257de0")), SP(1, 6),
         P("Who gains most from AI? Logical reasoning as an amplifier of LLM-assisted performance: "
           "protocol of a preregistered computational study with open-weight language models", TITLE),
         P(f"Study protocol preprint · package v{__version__} · no confirmatory data have been collected", SUB),
         SP(1, 6), P(AUTHOR_HTML, SMALL), SP(1, 8),
         tiles([("28", "open-weight AI models<br/>act as the solvers", "#257de0"),
                ("80", "solver units<br/>(models × reasoning modes)", "#6a4ce2"),
                ("50%", "of hard items get a<br/>planted AI mistake", "#e2452f"),
                ("USD 0", "total cost: free and<br/>open-source tools only", "#008a6b")]), SP(1, 6)]

story += [P("Abstract", H2), P(
    "Studies of people using AI assistants disagree on who benefits most: some find that AI help narrows the "
    "gap between weaker and stronger performers, others that it widens it. One proposed reconciliation is that "
    "the answer depends on whether the AI's output is easy to check. We describe a fully computational, "
    "preregistered test of this idea in which open-weight language models take the place of human solvers. "
    "Twenty-eight models (80 solver units across three reasoning modes) solve logic, easy arithmetic and "
    "multi-level deduction items, alone and with a stronger assistant model whose replies contain a planted "
    "logical error on exactly half of the hard items. The confirmatory hypotheses are that solvers with higher "
    "logical-reasoning scores gain more from the assistant when it can be wrong (H1), and that this advantage runs "
    "through detecting the assistant's errors (H4). Levelling on easy items (H2) is a registered secondary "
    "hypothesis. This preprint reports the design, the frozen analysis plan and the facts from the pilot and "
    f"re-pilot runs on development items. It reports no confirmatory results. Registration: {REG_TXT}. "
    f"Code and materials: {GH}."),
    plain("When people use AI helpers, does it close the gap between weaker and stronger thinkers, or widen it? "
          "My guess is that it depends on whether you can check the AI's answer. To test this without people, "
          "I use AI models as the 'students' and give them a stronger AI 'helper' that is sometimes secretly wrong. "
          "The plan was locked and published before any real data were collected, so the results cannot be "
          "bent to fit afterwards.")]

story += [P("1. Background and aims", H2), P(
    "Recent human studies report mixed patterns of who gains from large language model (LLM) assistance, and a "
    "recent theory paper predicts that gaps widen when AI output is hard to verify (background studies listed in "
    "the preregistration, §6.7). Testing this with people is slow and costly, and individual ability is hard to "
    "separate from general knowledge. Here the solvers are open-weight language models, so every solver sees every "
    "item in every condition, has no memory between items, and can be measured repeatedly at zero cost. The study "
    "asks whether logical reasoning skill, measured separately from knowledge and model size, predicts how much a "
    "solver gains from an AI assistant, and whether the answer differs between easy, checkable tasks and hard tasks "
    "where the assistant can be confidently wrong. The results concern reasoning agents as a controlled model of "
    "human-AI collaboration, not people."),
    P("Research questions", H3), bullets([
        "RQ1. Does logical reasoning predict how much a solver gains from an AI assistant on hard-to-verify items?",
        "RQ2. Does the pattern differ on easy, checkable items?",
        "RQ3 (exploratory). Does logical creativity predict gains in the originality of AI-assisted work?",
        "RQ4. Is the logic advantage explained by detecting the assistant's errors?"]),
    figure(F.design_flow(FIG / "fig1_design.png"),
           "<b>Figure 1.</b> One solver's path through the study. Every solver does the same tasks alone and with "
           "the assistant, so each solver is compared with itself."),
    plain("Each AI 'student' first takes a logic test on its own. Then it solves easy maths and hard logic "
          "puzzles, first alone and then with the helper. On half of the hard puzzles the helper's answer has one "
          "hidden mistake. We see who gets better with help, and who spots the mistake instead of copying it.")]

story += [P("2. Hypotheses", H2), table([
    ["", "Hypothesis", "Test", "Status"],
    ["H1", "On hard deduction items with a planted assistant error, higher-logic solvers have a larger AI gain",
     "One-sided slope of logic > 0", "Confirmatory"],
    ["H4", "The H1 effect runs through error detection (d′)", "Indirect effect logic → d′ → gain > 0",
     "Confirmatory"],
    ["H2", "On easy arithmetic, logic adds less than +5 percentage points of AI gain per SD",
     "Non-superiority against +5 pp", "Registered secondary"],
    ["H3", "Logical creativity predicts originality gain", "Slope > 0", "Exploratory"],
], [1.0 * cm, 7.6 * cm, 4.6 * cm, 3.4 * cm], zebra=True),
    P("Holm correction across H1 and H4 (familywise α = .05, one-sided). H2 is judged on its unadjusted p-value "
      "and 90% CI; a non-significant H2 is reported as inconclusive. H2 was moved out of the confirmatory family "
      "before registration because, with pilot variance, its power was about 2% (DEVIATIONS D16).", SMALL),
    figure(F.hypotheses_idea(FIG / "fig2_hypotheses.png"),
           "<b>Figure 2.</b> What each hypothesis predicts. The lines are drawings of the predictions, not data."),
    plain("H1: on hard puzzles where the helper can be wrong, better logical thinkers should gain more from help. "
          "H4: they gain more because they catch the helper's mistakes. H2: on easy maths that anyone can check, "
          "logic should matter much less.")]

story += [P("3. Methods", H2), P("3.1 Solvers and assistants", H3), P(
    "Twenty-eight open-weight models from about 0.36 to 27 billion parameters: 22 run locally with Ollama and 6 "
    "through free API tiers (Groq, Google AI Studio, Mistral). Each model runs in three reasoning modes (direct, "
    "brief, careful); the two always-thinking models (gpt-oss-20b, Gemma 4 26B) are pooled into one unit each, "
    "giving 80 solver units clustered in 28 models. The primary assistant is gpt-oss-120b; Gemma 4 31B is a "
    "registered replication. Temperature 0, fixed token limits, every request and reply hashed with SHA-256 and "
    "logged with the provider's reported model version. Total cost USD 0."),
    P("3.2 Items and conditions", H3), bullets([
        "Logic battery, alone (moderator): 120 belief-bias syllogisms, 80 conditional inferences, 30 "
        "cognitive-reflection items with fresh numbers.",
        "Benchmarks, alone: FOLIO 200 and LogiQA 300 (validity of the logic score), GSM8K 300, MMLU 400 "
        "(knowledge control, logic and mathematics subjects removed). Dataset revisions are pinned.",
        "Easy, checkable task: 200 multi-step arithmetic problems; the assistant answers naturally.",
        "Hard task: 300 rule-chain deduction puzzles at four levels (8, 12, 16 and 20 statements; levels 3 and 4 "
        "add one and two “either/or” steps). Answers TRUE/FALSE/UNKNOWN balanced within level. The assistant's "
        "reply is scripted: correct, or with exactly one subtle slip reaching a set wrong answer (50% of items, "
        "balanced per level × answer). Both reply kinds have the same shape (6 numbered steps, 450–800 characters).",
        "Logical creativity (exploratory, 40 items) and 24 catch items used only for a known-answer gate.",
        "Assisted prompts state that the assistant “is often helpful but can make mistakes” and require an "
        "ASSISTANT_ERROR: YES/NO flag and a FINAL answer."]),
    P("3.3 Measures and analysis", H3), P(
        "Logic score: mean solo accuracy on the battery, z-scored. AI gain: accuracy with the assistant minus solo "
        "accuracy on the same items, in percentage points. Error detection: d′ from flags on wrong versus correct "
        "assistant replies. Solver-level OLS with knowledge (MMLU), log parameter count and a thinking-model "
        "indicator as controls, cluster-robust errors by model, and wild cluster bootstrap p-values (9,999 "
        "replications). H4 uses a cluster bootstrap of the indirect effect. Registered secondary analyses include "
        "adjusted (ANCOVA) scores, a hard-minus-easy contrast and a within-model test across reasoning modes."),
    P("3.4 Safeguards", H3), bullets([
        "Seven preregistered known-answer gates (G1–G7) with fixed consequences, for example excluding solvers "
        "whose answers are mostly unreadable (G3) or marking H4 uninformative if error flags fail a known-answer "
        "check (G6).",
        "Registered items (seed 20261026) are locked: the code refuses to send them to any model unless every "
        "study file matches the freeze. Pilot and synthetic runs used a separate development seed (7110526).",
        f"Freeze file SHA-256 {FZ['freeze_sha256'][:16]}…, code SHA-256 {FZ['code_sha256_frozen'][:16]}… "
        "(full values on the registration)."]),
    figure(F.gates(FIG / "fig8_gates.png"),
           "<b>Figure 3.</b> The seven known-answer gates and what happens if one fails. They are checked before "
           "any hypothesis result counts."),
    plain("Before trusting any number, the code runs seven safety checks with known right answers, like testing "
          "a thermometer in ice water first. The real test puzzles stay locked until the plan is frozen, and every "
          "file has a digital fingerprint (SHA-256) so anyone can confirm nothing was changed later.")]

pilot_trials = PM["stages"]["trials"]["n"]
story += [P("4. Pilot and re-pilot (development items only)", H2), P(
    f"<b>Pilot.</b> Three models on the Groq free tier (allam-2-7b, gpt-oss-20b, qwen3.8-27b) × 3 modes, "
    f"{pilot_trials:,} trials, {PM['stages']['trials'].get('api_errors', 0)} API errors. The parser gate G1 "
    f"passed {PM['gates']['G1']['n_correct']}/{PM['gates']['G1']['n_cases']} cases. Gate G3 excluded "
    f"{excl['n_excluded']} of {excl['n_units']} solver units (all allam-2-7b, unreadable answers). Strong models "
    "were at or near ceiling on the original deduction puzzles, which led to the four-level ladder. Variance "
    "estimates: AI gain on hard planted-error items, between-model SD 7 pp and within-model SD 11 pp; easy-item "
    "gain, between-model SD 15 pp and within-model SD 37 pp. Simulated power for H1 with these values: 52% at "
    "3 pp per SD, 86% at 5 pp per SD, 99.7% at 8 pp per SD."),
    P("<b>Re-pilot of the difficulty levels.</b> 96 development puzzles (24 per level), solvers allam-2-7b, "
      "gpt-oss-20b, qwen3.8-27b and Gemma 4 26B, direct and careful modes, alone and with gpt-oss-120b. "
      "Accuracy alone:")]
rows = [["Solver", "Mode", "Level 1", "Level 2", "Level 3", "Level 4"]]
for s in ("allam-2-7b", "gemma-4-26b", "qwen3.8-27b", "gpt-oss-20b"):
    for mode in ("direct", "careful"):
        rows.append([s, mode] + [pct(cell(s, mode, "alone", lv)) for lv in (1, 2, 3, 4)])
story += [table(rows, [4.0 * cm, 2.2 * cm] + [2.6 * cm] * 4, zebra=True), P(
    "Each cell is 24 puzzles. Gemma 4 26B scores near 0% because almost all of its replies were cut off while it "
    "was still thinking, not because it chose wrong answers; it stays in the roster and the gates decide whether "
    "it is analysed (DEVIATIONS D14).", SMALL),
    figure(F.repilot_alone(TAB, FIG / "fig3_levels.png"),
           "<b>Figure 4.</b> Re-pilot accuracy alone at each difficulty level (same numbers as the table above). "
           "Harder levels pull the strong models down from the ceiling, as intended.")]
rows = [["Solver", "Mode", "Flagged the planted error (L1/L2/L3/L4)", "Copied the wrong answer (L1/L2/L3/L4)"]]
for s in ("allam-2-7b", "qwen3.8-27b", "gpt-oss-20b"):
    for mode in ("direct", "careful"):
        f = " / ".join(pct(cell(s, mode, "AI wrong (planted)", lv, "flagged_error")) for lv in (1, 2, 3, 4))
        c = " / ".join(pct(cell(s, mode, "AI wrong (planted)", lv, "copied_ai_wrong")) for lv in (1, 2, 3, 4))
        rows.append([s, mode, f, c])
story += [table(rows, [3.4 * cm, 1.8 * cm, 5.7 * cm, 5.7 * cm], zebra=True), P(
    "Planted-error items only, 11 or 12 per cell; descriptive, from three models, and not a test of any "
    "hypothesis.", SMALL),
    figure(F.planted(TAB, FIG / "fig4_planted.png"),
           "<b>Figure 5.</b> When the helper was secretly wrong: how often each solver flagged the mistake (blue) "
           "and how often it copied the wrong answer (orange). Careful mode, development items."),
    P(f"<b>Surface cues (gate G5).</b> In the re-pilot, planted assistant replies could be told apart from "
      f"correct ones by surface features (worst AUC {g5_before:.2f} for gpt-oss-120b and {g5_gemma:.2f} for Gemma 4 "
      f"31B; limit 0.65). After the fixed reply shape was introduced, gpt-oss-120b scored {g5_after:.2f} (pass). "
      "The check for Gemma 4 31B with the fixed shape did not finish because of free-tier overload; G5 is "
      "re-evaluated in the main study."),
    two(half(F.power(FIG / "fig6_power.png"), "<b>Figure 6.</b> Simulated power for H1 with pilot variance."),
        half(F.g5(g5_before, g5_gemma, g5_after, FIG / "fig7_g5.png"),
             "<b>Figure 7.</b> Surface-cue check G5. Lower is better; at or under .65 passes.")),
    plain("Small test runs on practice puzzles showed the original puzzles were too easy, so four difficulty levels "
          "were added. Weak models mostly copied the helper's mistakes; strong models mostly caught them. These "
          "practice runs only test the machinery. They are not the study's answer.")]

story += [P("5. Validation of the pipeline on synthetic data", H2), P(
    "Before registration the whole pipeline was run on 84 simulated solver units in a world with a planted "
    "effect and a world without one. In the no-effect world neither H1 nor H4 was supported. In the planted-effect "
    f"world H4 was supported (Holm p = {p3(ca['H4']['p_holm'])}) and H1 was positive but not significant "
    f"(+{ca['H1']['estimate_pp_per_sd']:.1f} pp per SD, p = {p3(ca['H1']['p_raw'])}). Separate simulations "
    "showed false-alarm rates near 5% for the registered gain-score tests and 62–99% for adjusted (ANCOVA) scores, "
    "which is why gain scores are primary (DEVIATIONS D4). These are simulated models, not study results."),
    figure(F.synthetic(ca, cn, FIG / "fig9_synthetic.png"),
           "<b>Figure 8.</b> The full analysis run on two simulated worlds where the truth is known."),
    plain("To check the analysis itself, I ran it on fake data twice: once where the effect was built in and once "
          "where it was not. It found nothing in the no-effect world, which is what a fair test should do.")]

story += [P("6. Status and timeline", H2), P(
    "The study materials were frozen and registered before any registered item was sent to a model. Data "
    "collection runs on free API tiers and a local computer and is expected to take several weeks because of "
    "daily limits; the run pauses and resumes from a hashed cache. All deviations will be reported with their "
    "reasons and effects. Results will be reported in a separate article whatever their direction."),
    figure(F.timeline(FIG / "fig10_timeline.png"), "<b>Figure 9.</b> Where the study is now (filled dots are done).")]

story += [P("7. Limitations", H2), bullets([
    "Language-model solvers are not people; the study models human-AI collaboration, it does not measure it.",
    "Model size, family and training data are confounded across models; the within-model test across reasoning "
    "modes addresses this only in part.",
    "Free API endpoints can change and cannot be version-pinned; the reported model version is logged per call.",
    "Benchmark items may be in training data; the main tasks therefore use freshly generated items."])]

story += [P("Words used", H2), table([
    ["Word", "Meaning"],
    ["Solver", "An AI model (in one reasoning mode) that answers the questions, standing in for a person."],
    ["Assistant", "The stronger AI helper whose answer the solver sees; sometimes it is secretly wrong."],
    ["Planted error", "One small deliberate slip in the helper's reasoning that leads to a set wrong answer."],
    ["AI gain", "Score with the helper minus score alone, on the same items, in percentage points (pp)."],
    ["d′ (d-prime)", "How well a solver tells wrong helper replies from right ones; 0 means guessing."],
    ["SD", "Standard deviation: a typical distance from the average. \"Per SD\" means per step of that size."],
    ["Preregistration", "Publishing the plan before collecting data, so it cannot be changed to fit the results."],
    ["SHA-256", "A digital fingerprint of a file; any change to the file changes the fingerprint."],
    ["Holm correction", "A rule that makes the bar stricter when more than one hypothesis is tested."],
    ["Power", "The chance the study detects an effect if the effect is really there."],
], [3.4 * cm, 13.2 * cm], zebra=True)]

story += [P("Statements", H2), bullets([
    f"<b>Registration:</b> {REG_TXT}. <b>Code, items generator, configuration and documentation:</b> {GH} "
    "(MIT licence); frozen release zip on OSF.",
    "<b>Data availability:</b> all items, replies, scored data and reports will be public on OSF (CC BY 4.0).",
    "<b>Ethics:</b> no human participants and no personal data.",
    "<b>Use of AI assistance:</b> the research idea, questions and design decisions are the author's own. Code, "
    "the preregistration draft, documentation and this preprint were drafted with help from an AI assistant "
    "(Claude, by Anthropic) under the author's direction; the author reviewed and approved all content and takes "
    "full responsibility for it. The AI assistant is not an author.",
    "<b>Author contributions (CRediT):</b> KEYUR SHAILESHKUMAR PATEL: conceptualization, methodology, software, "
    "validation, formal analysis, writing.",
    "<b>Funding:</b> none. <b>Competing interests:</b> none."])]

story += [P("References", H2), P(
    "Cameron, A. C., Gelbach, J. B., &amp; Miller, D. L. (2008). Bootstrap-based improvements for inference with "
    "clustered errors. <i>Review of Economics and Statistics, 90</i>(3), 414–427.<br/>"
    "Hautus, M. J. (1995). Corrections for extreme proportions and their biasing effects on estimated values of d′. "
    "<i>Behavior Research Methods, Instruments, &amp; Computers, 27</i>(1), 46–51.<br/>"
    "Holm, S. (1979). A simple sequentially rejective multiple test procedure. <i>Scandinavian Journal of "
    "Statistics, 6</i>(2), 65–70.<br/>"
    "Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. "
    "<i>Social Psychological and Personality Science, 8</i>(4), 355–362.<br/>"
    "Vaccaro, M., Almaatouq, A., &amp; Malone, T. (2024). When combinations of humans and AI are useful: A "
    "systematic review and meta-analysis. <i>Nature Human Behaviour, 8</i>, 2293–2303.<br/>"
    "Westfall, J., &amp; Yarkoni, T. (2016). Statistically controlling for confounding constructs is harder than "
    "you think. <i>PLOS ONE, 11</i>(3), e0152719.", SMALL)]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Who gains most from AI? Protocol preprint (Logic x AI)", author="KEYUR SHAILESHKUMAR PATEL")
lab = f"Logic × AI · protocol preprint · v{__version__}"
doc.build(story, onFirstPage=footer(lab), onLaterPages=footer(lab))
print(f"written {OUT}")
