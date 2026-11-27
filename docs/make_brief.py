# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)


"""Builds docs/Project_Brief.pdf: the project brief, with one easy-language section.
Run: python docs/figures.py && python docs/make_brief.py"""
import json
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdfkit import (AI_USE, AUTHOR_HTML, H2, H3, SMALL, SP, SUB, TITLE, WARM, P, box, bullets,  # noqa: E402
                    figure, footer, table)

from logicxai import __version__  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).parent / "figures"
OUT = Path(__file__).parent / "Project_Brief.pdf"
V = json.loads((ROOT / "exploratory_synthetic" / "validation" / "validation.json").read_text(encoding="utf-8"))


def rate(world, est):
    r = next(x for x in V["monte_carlo"] if x["check"] == "H1" and x["world"] == world and x["estimator"] == est)
    return f"{100 * r['reject_rate']:.0f}%"


story = [
    P("Logic × AI: who gains most from AI helpers?", TITLE),
    P(f"Project brief · a fully computational, preregistered study with free open models · v{__version__}", SUB),
    SP(1, 6), P(AUTHOR_HTML, SUB), SP(1, 10),

    P("Summary", H2),
    box("AI helpers raise average performance, but they are also confidently wrong at times. This study asks "
        "whether <b>logical reasoning decides who benefits</b>: do better reasoners gain more from an AI "
        "helper because they catch its mistakes (amplification), while on easy tasks the AI levels everyone "
        "up (levelling)? We test this on <b>84 AI 'solvers'</b> (28 free open models × 3 thinking modes), each "
        "working alone and then with a strong AI helper that is secretly wrong on exactly half of the hard "
        "items. No humans take part. Every item, reply and result is fingerprinted with SHA-256, and the "
        "whole plan is preregistered on OSF before any real data exist."),

    P("In plain words", H2),
    P("Imagine a student doing homework with a very confident tutor. Most of the time the tutor is right. "
      "Sometimes the tutor is wrong but sounds just as sure. A student who thinks carefully will notice "
      "('wait, that does not follow') and fix it. A student who does not will copy the mistake."),
    P("We do the same experiment, but every 'student' is an AI model. First we measure how good each one "
      "is at pure logic, with puzzles that a computer can prove right or wrong. Then each one solves two kinds "
      "of tasks, once alone and once with a big AI helper:"),
    bullets(["<b>Easy maths</b>, where the helper is nearly always right and its answers are easy to check. "
             "Here we expect the helper to lift the weaker solvers most, so logic should matter little.",
             "<b>Hard logic puzzles</b>, where the helper is wrong on exactly half of them. Here we expect the "
             "good reasoners to catch the mistakes and so gain more."]),
    P("Because there are no people involved, there are no privacy or ethics risks, and anyone can rerun the "
      "whole study on a normal computer with free tools. Before running it for real, we tested the whole "
      "machine on fake data where we know the right answer. That test caught a hidden flaw in our first "
      "analysis plan, which we fixed before registering."),

    P("The problem and the gap", H2),
    P("Field and lab studies find that generative AI helps most on tasks inside its competence and can hurt "
      "on tasks just outside it, where its errors are hard to see (Dell'Acqua et al., 2023; Noy &amp; Zhang, "
      "2023; Brynjolfsson et al., 2025). What decides who catches those errors is not settled. Human studies "
      "are slow, costly and hard to control. Here the AI's errors are planted on purpose, every answer key is "
      "proved by code, and the 'participants' can be rerun by anyone, so the same question can be asked "
      "cleanly and openly."),

    P("Design at a glance", H2),
    figure(FIG / "design.png", "Figure 1. Each solver is tested alone and with the AI helper on the same items. "
           "Hard items carry a planted error exactly half the time."),
    table([["Element", "Choice (each one the author's decision)"],
           ["Solvers", "28 free open models × 3 thinking modes (direct, brief, careful); 2 always-thinking models pooled = 80 solver units, clustered by model"],
           ["Logic score", "Syllogisms (120), conditional inferences (80), reflection puzzles (30); keys proved by code"],
           ["Tasks", "Easy maths (200 items), hard deduction (300 items, 150 with a planted AI error)"],
           ["Helper AI", "gpt-oss-120b (primary), Gemma 4 31B (replication)"],
           ["Controls", "Knowledge (MMLU-style test, 400 items) and log model size"],
           ["Validity links", "FOLIO, LogiQA, GSM8K benchmark subsets (medium size)"],
           ["Access", "Free APIs with Ollama as backup; total cost USD 0"]], [3.2 * cm, 13.4 * cm]),

    P("Hypotheses", H2),
    table([["", "Claim", "Test"],
           ["H1", "Amplification: when the helper is wrong, higher logic → bigger AI gain",
            "One-sided slope of AI gain on logic, wild cluster bootstrap"],
           ["H2", "Levelling: on easy maths, logic adds less than +5 points of AI gain per SD",
            "Registered secondary: non-superiority against +5 pp, unadjusted"],
           ["H4", "Mechanism: logic → spotting AI errors (d′) → AI gain", "Indirect effect, cluster bootstrap"],
           ["H3", "Logical creativity → more original conclusions with AI", "Exploratory only"]],
          [1.2 * cm, 8.4 * cm, 7.0 * cm]),
    P("H1 and H4 form the confirmatory family with Holm correction (α = .05); H2 became a registered secondary after the pilot showed it had too little power. AI gain = score with the helper minus score "
      "alone on the same items, adjusted for knowledge and size.", SMALL),

    P("Risks and safeguards", H2),
    table([["Risk", "Safeguard"],
           ["A measure fails silently (as a model checkpoint did in the author's earlier study)",
            "Known-answer gates G1 to G7 must pass before a result is read; G6 can mark H4 uninformative"],
           ["The analysis finds effects that are not there",
            f"Validated on synthetic worlds: false alarm rate {rate('ignore', 'gain_wild')} (target 5%); the "
            f"earlier adjusted-score plan gave {rate('ignore', 'ancova')} and was replaced"],
           ["Answer keys are wrong", "Every key recomputed by a second, independent method (SAT solver, syllogism rules)"],
           ["The helper's wrong replies look different from its right ones",
            "Gate G5: surface-cue AUC must be ≤ .65; templates matched in length and style"],
           ["A model answers the same thing every time", "Gate G2: excluded if > 90% the same answer (preregistered)"],
           ["Items leak into training data or are tuned after looking",
            "Fresh code-generated items; separate dev seed; registered items locked until the freeze"],
           ["Files change after registration", "Freeze: SHA-256 of every file and of the release zip, registered on OSF"],
           ["API models change silently", "Reported model id logged for every call; Ollama digests recorded; caveat in prereg"],
           ["Results depend on one analysis choice", "Multiverse of 72 variants per hypothesis (exploratory)"]],
          [6.2 * cm, 10.4 * cm]),

    P("Evidence that the pipeline works", H2),
    figure(FIG / "validation.png", "Figure 2. Synthetic validation. The registered test reports an effect about "
           "5% of the time when there is none, and nearly always when one is planted."),
    figure(FIG / "power.png", "Figure 3. Planned power for H1: about 80% for an effect of 4 points of AI gain "
           "per SD of logic."),
    figure(FIG / "forest.png", "Figure 4. Exploratory run on synthetic data (not study results): estimates in "
           "a planted-effect world and a no-effect world."),

    P("Status", H2),
    table([["Step", "Status"],
           ["Study code, items, gates, self-checks, reports", "Done (v" + __version__ + ")"],
           ["Unit tests and parser known-answer gate", "Done, all pass"],
           ["Synthetic exploratory run and pipeline validation", "Done (see exploratory report)"],
           ["Preregistration text", "Drafted; values to fill after the pilot"],
           ["Pilot on real models (dev items)", "Next: needs the author's free API keys"],
           ["Freeze and OSF registration", "After the pilot"],
           ["Full run on registered items", "After registration"]], [9.6 * cm, 7.0 * cm], status_col=None),

    P("What is new", H2),
    table([["Usually", "This study"],
           ["Human participants, small samples, costly", "84 AI solvers, rerunnable by anyone at no cost"],
           ["AI errors happen by chance", "Errors planted on exactly half the hard items, with checked keys"],
           ["Ability measured by general tests", "Logic measured with provably correct puzzles, linked to FOLIO and LogiQA"],
           ["Analysis checked after the fact", "Analysis validated on worlds with known truth before registration"],
           ["Files described in words", "Every file fingerprinted with SHA-256 and the freeze registered"]],
          [7.6 * cm, 9.0 * cm]),

    P("Impact", H2),
    table([["Who", "What they get"],
           ["Researchers", "An open, reusable test bed for 'who benefits from AI' questions"],
           ["Educators", "Evidence on whether teaching logic helps people use AI safely"],
           ["AI builders", "A way to measure if a model catches a helper's mistakes"],
           ["Open science", "A worked example of gates, freeze, validation and full hashing"]], [4.0 * cm, 12.6 * cm]),

    P("Cost and plan", H2),
    table([["Stage", "What happens", "Cost"],
           ["Pilot", "6 solvers, dev items: timing, parsing, floor and ceiling checks", "USD 0"],
           ["Freeze + OSF", "python -m logicxai freeze, package; register both SHA-256 values", "USD 0"],
           ["Full run", "80 solver units, registered items; resumable cache; multi-day on free limits", "USD 0"],
           ["Optional", "A rented GPU with Ollama if free limits are too slow", "Depends on provider"]],
          [2.6 * cm, 10.6 * cm, 3.4 * cm]),

    P("Limitations", H2),
    bullets(["Solvers are AI models, so results speak to AI-assisted AI, and only indirectly to people.",
             "H1 and H2 are correlational across solvers; thinking mode gives a within-model check.",
             "Free API versions can change; the reported model id is logged for every call.",
             "The synthetic results show the pipeline works; they are not evidence about real models."]),

    P("Statements", H2),
    P(AI_USE),
    P("<b>Ethics.</b> No human participants and no personal data. <b>Funding.</b> None. "
      "<b>Competing interests.</b> None. <b>Data and code.</b> Open (MIT for code, CC BY 4.0 for data and reports)."),

    P("References", H2),
    P("Brynjolfsson, E., Li, D., &amp; Raymond, L. (2025). Generative AI at work. <i>Quarterly Journal of "
      "Economics</i>, 140(2), 889–942.<br/>"
      "Cameron, A. C., Gelbach, J. B., &amp; Miller, D. L. (2008). Bootstrap-based improvements for inference "
      "with clustered errors. <i>Review of Economics and Statistics</i>, 90(3), 414–427.<br/>"
      "Dell'Acqua, F., et al. (2023). Navigating the jagged technological frontier. Harvard Business School "
      "Working Paper 24-013.<br/>"
      "Frederick, S. (2005). Cognitive reflection and decision making. <i>Journal of Economic Perspectives</i>, "
      "19(4), 25–42.<br/>"
      "Han, S., et al. (2022). FOLIO: Natural language reasoning with first-order logic. arXiv:2209.00840.<br/>"
      "Hautus, M. J. (1995). Corrections for extreme proportions and their biasing effects on estimated values "
      "of d′. <i>Behavior Research Methods, Instruments, &amp; Computers</i>, 27(1), 46–51.<br/>"
      "Holm, S. (1979). A simple sequentially rejective multiple test procedure. <i>Scandinavian Journal of "
      "Statistics</i>, 6(2), 65–70.<br/>"
      "Lakens, D. (2017). Equivalence tests. <i>Social Psychological and Personality Science</i>, 8(4), 355–362.<br/>"
      "Liu, J., et al. (2020). LogiQA: A challenge dataset for machine reading comprehension with logical "
      "reasoning. <i>Proceedings of IJCAI 2020</i>.<br/>"
      "Noy, S., &amp; Zhang, W. (2023). Experimental evidence on the productivity effects of generative "
      "artificial intelligence. <i>Science</i>, 381(6654), 187–192.<br/>"
      "Simonsohn, U., Simmons, J. P., &amp; Nelson, L. D. (2020). Specification curve analysis. <i>Nature Human "
      "Behaviour</i>, 4, 1208–1214.<br/>"
      "Westfall, J., &amp; Yarkoni, T. (2016). Statistically controlling for confounding constructs is harder "
      "than you think. <i>PLoS ONE</i>, 11(3), e0152719.", SMALL),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="Logic x AI: project brief",
                        author="KEYUR SHAILESHKUMAR PATEL")
lab = f"Logic × AI · project brief · v{__version__}"
doc.build(story, onFirstPage=footer(lab), onLaterPages=footer(lab))
print(OUT)
