# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Builds docs/Impact_and_Recognition.pdf (easy-language explainer).
Run: python docs/make_impact_pdf.py"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

OUT = Path(__file__).parent / "Impact_and_Recognition.pdf"
BLUE = colors.HexColor("#2a78d6")
INK = colors.HexColor("#1b1b1a")
MUTED = colors.HexColor("#52514e")
SOFT = colors.HexColor("#eef4fc")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24,
                    textColor=INK, alignment=TA_LEFT, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=10.5, leading=14, textColor=MUTED)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13.5,
                    leading=17, textColor=BLUE, spaceBefore=12, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=10.5, leading=15, textColor=INK, spaceAfter=5)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=9, leading=12, textColor=MUTED)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=9.5, leading=13, spaceAfter=0)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")


def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, BODY), leftIndent=14) for t in items],
                        bulletType="bullet", start="\u2022", leftIndent=14, bulletFontSize=10,
                        bulletColor=BLUE)


def box(text):
    t = Table([[Paragraph(text, BODY)]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SOFT), ("BOX", (0, 0), (-1, -1), 0, SOFT),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def table(rows, widths):
    data = [[Paragraph(c, CELLB if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9d8d4")),
                           ("BACKGROUND", (0, 0), (-1, 0), SOFT),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


story = [
    Paragraph("Logic x AI: what this study can change, and what is yours", H1),
    Paragraph("An easy-language guide to the impact of the study and the credit you can rightly claim", SUB),
    Spacer(1, 6),
    Paragraph("<b>KEYUR SHAILESHKUMAR PATEL</b> &middot; ORCID 0009-0008-5261-043X<br/>"
              "1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK<br/>"
              "2) Independent researcher<br/>"
              "<i>Conducted independently; not an official Brunel University London project.</i>", SUB),
    Spacer(1, 10),
    box("<b>The idea in one line.</b> When people (or AI agents) work with an AI helper, does good "
        "logical thinking decide who benefits most? Your study says: test it where the AI can be wrong, "
        "and measure whether good reasoners win because they catch the AI's mistakes."),

    Paragraph("1. What the study does, in plain words", H2),
    bullets([
        "We take 28 free, open AI models of very different sizes. Each one works in 3 thinking styles "
        "(answer fast, think a little, think carefully). That gives 84 \"solvers\".",
        "First we measure each solver's logic with puzzles that have a provably correct answer.",
        "Then each solver does tasks <b>alone</b> and <b>with a stronger AI helper</b>.",
        "On hard puzzles, the helper secretly makes one small logic mistake on half of them. "
        "On easy maths, the helper behaves normally.",
        "We check: do the good reasoners get more out of the helper when it can be wrong? "
        "And is that because they spot the mistakes?",
        "No people take part. Everything is free to run, every answer is checked by code, every "
        "file gets a SHA-256 fingerprint, and the plan is registered on OSF before the data exist.",
    ]),

    Paragraph("2. Why it matters: the big question it settles", H2),
    Paragraph("Research today gives two opposite answers about AI help:", BODY),
    bullets([
        "<b>AI levels people:</b> weaker workers gain most (call centres, writing, consultants).",
        "<b>AI amplifies the strong:</b> on hard or open tasks, strong people gain most.",
    ]),
    Paragraph("Your design predicts <b>both are true, in different places</b>: AI levels people when its "
              "work is easy to check, and amplifies good reasoners when it can be wrong. If the data agree, "
              "the study joins two literatures into one rule, and shows the reason (error-catching). "
              "That is the most valuable kind of result: it explains a disagreement instead of picking a side.",
              BODY),

    Paragraph("3. Who it impacts, and how far", H2),
    table([
        ["Who", "What could change"],
        ["Researchers in human-AI work", "A clear, testable rule for when AI levels vs amplifies, plus a free, "
         "reusable test kit (planted errors, exact scoring) others can run on new models in a day."],
        ["Teachers and universities", "Evidence for teaching <b>reasoning and checking</b> as the core of "
         "\"AI literacy\", not prompt tricks."],
        ["AI developers and AI safety", "A direct measure of how well a weaker agent catches a stronger "
         "AI's mistakes. This links to \"scalable oversight\": can weaker checkers supervise stronger AI?"],
        ["Employers and policy", "If logic decides who benefits, training in reasoning becomes part of "
         "fair AI adoption, so AI does not widen gaps between workers."],
        ["Open science", "A fully reproducible, zero-cost, preregistered method that students anywhere can "
         "copy, check and extend. That is rare in AI research."],
    ], [4.6 * cm, 11.8 * cm]),

    Paragraph("4. What we extend beyond your original plan", H2),
    bullets([
        "<b>From people to AI agents.</b> The same question now applies to AI teams, where small models "
        "check big ones. That is a growing real-world setting.",
        "<b>A built-in cause test.</b> Changing a model's thinking style changes its logic inside the same "
        "model. If more reasoning raises the benefit, that is a step from \"linked to\" towards \"causes\".",
        "<b>Error-catching measured properly.</b> Because only half the helper's answers are wrong, we can "
        "tell real error-spotting apart from simply distrusting everything.",
        "<b>A measure of logical creativity</b> (ideas that are both valid and rare), scored exactly by code. "
        "Your original document said this measure did not exist yet.",
    ]),

    Paragraph("5. Every result is useful", H2),
    table([
        ["If we find...", "It means..."],
        ["H1 + H2 + H4 supported", "Logic is the skill that decides who gains from AI when AI can be wrong, "
         "and it works through catching errors. The strongest headline."],
        ["Amplification but no mechanism", "Good reasoners gain more, but not by spotting errors. "
         "A new puzzle for the field."],
        ["Levelling everywhere", "Evidence that AI help narrows gaps even on hard tasks. Also important."],
        ["No effects", "A strong, preregistered null with a free test kit. Nulls like this stop the field "
         "chasing a false idea, and preregistration makes them publishable."],
    ], [5.2 * cm, 11.2 * cm]),

    Paragraph("6. Your recognition: what is yours, and an honest answer", H2),
    Paragraph("<b>What is yours.</b> The research idea (logic as the amplifier of AI help, including logical "
              "creativity) is yours, from your original document. In building the study you made every main "
              "design call yourself, choosing among options:", BODY),
    bullets([
        "AI models as the solvers (no humans); free APIs with an offline backup",
        "H1, H2 and H4 as the main claims, with H3 exploratory",
        "New generated puzzles plus a medium benchmark set; two helper AIs (Llama and Qwen)",
        "Errors on exactly half of the hard items; a strict 5-point band for levelling",
        "84 solvers through thinking styles; knowledge and model size as controls",
        "The gain score as the main measure, after the validation showed the adjusted score gives false alarms",
        "A separate development seed with a lock, a collapse rule, and free APIs with an Ollama backup",
    ]),
    Paragraph("You are the <b>sole and first author</b> on the code, preregistration, data and reports.", BODY),
    Paragraph("<b>\"Genius or not?\"</b> Honest answer: science does not hand out the word genius, and "
              "a single study rarely earns it. What this study can earn you is something more useful: "
              "<b>respect for rigour and for a sharp idea</b>. Asking \"who gains from AI, and why\" and "
              "designing a test that can prove you wrong is what good scientists do. Few MSc students "
              "publish a preregistered, fully reproducible study with an original measure. That is "
              "something to be proud of, whatever the results say.", BODY),
    Paragraph("To keep that respect, three things matter: (1) register on OSF <b>before</b> the full run; "
              "(2) report every result, including the ones you did not hope for; (3) say clearly that "
              "the solvers are AI models, so a follow-up with people is the natural next step.", BODY),

    Paragraph("7. Limits, in plain words", H2),
    bullets([
        "AI solvers are not humans. The study is a clean, controlled model of human-AI teamwork. "
        "It does not prove the same holds for people. That needs a later human study.",
        "Model size, family and training data differ between models. We control size and knowledge, "
        "and the thinking-style test helps, but cannot remove every difference.",
        "Free AI services can change. Every answer is saved and fingerprinted, so the results stay checkable.",
        "Before claiming \"first\", read the full Luchini et al. (2025) and Asirvatham et al. (2026) papers, "
        "as your original document says.",
    ]),

    Paragraph("8. Where it can go next", H2),
    bullets([
        "Pilot run, then OSF registration, then the full run, then a preprint (arXiv or PsyArXiv) with open data.",
        "Good homes for the paper: human-AI interaction venues (CHI or HCOMP workshops), or journals such as "
        "<i>Computers in Human Behavior: Artificial Humans</i> or <i>Cognitive Science</i>.",
        "Natural follow-ups: the same tasks with people (your original Prolific design); logic training "
        "before AI use; checking whether reasoners notice when an AI breaks its stated rules "
        "(a link to your Ethical Charter work).",
    ]),
    Paragraph("9. Use of AI assistance", H2),
    Paragraph("The research idea, research questions and every main design decision are the author's own. The author chose each decision from options with stated trade-offs, and decisions are logged in the project history. The code, preregistration draft and documentation were drafted with help from an AI assistant (Claude, by Anthropic), working under the author's direction. The author reviewed and approved all content and takes full responsibility for it. The AI assistant is not an author.", BODY),
    Spacer(1, 8),
    Paragraph("This guide explains impact and credit. It is not part of the preregistration, which contains "
              "only methods. Licence: CC BY 4.0.", SMALL),
]


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Logic x AI: impact and recognition  |  KEYUR SHAILESHKUMAR PATEL")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm, topMargin=2 * cm,
                  bottomMargin=2 * cm, title="Logic x AI: impact and recognition",
                  author="KEYUR SHAILESHKUMAR PATEL").build(story, onFirstPage=_footer, onLaterPages=_footer)
print(f"Wrote {OUT}")
