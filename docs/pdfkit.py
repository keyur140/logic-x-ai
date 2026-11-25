# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)


"""Shared look for the PDFs in docs/ (fonts, styles, tables, author block, statements)."""
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, Paragraph, Spacer, Table,
                                TableStyle)

_F = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_F / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(_F / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", str(_F / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DVBI", str(_F / "DejaVuSans-BoldOblique.ttf")))
from reportlab.lib.fonts import addMapping  # noqa: E402
addMapping("DV", 0, 0, "DV")
addMapping("DV", 1, 0, "DVB")
addMapping("DV", 0, 1, "DVI")
addMapping("DV", 1, 1, "DVBI")

BLUE = colors.HexColor("#2a78d6")
ORANGE = colors.HexColor("#eb6834")
INK = colors.HexColor("#1b1b1a")
MUTED = colors.HexColor("#52514e")
SOFT = colors.HexColor("#eef4fc")
WARM = colors.HexColor("#fdf0ea")
LINE = colors.HexColor("#d9d8d4")
GOOD = colors.HexColor("#e8f7f1")
W = 16.6 * cm

TITLE = ParagraphStyle("T", fontName="DVB", fontSize=19, leading=23, textColor=INK, alignment=TA_LEFT,
                       spaceAfter=4)
SUB = ParagraphStyle("S", fontName="DV", fontSize=10, leading=14, textColor=MUTED)
H2 = ParagraphStyle("H2", fontName="DVB", fontSize=13, leading=17, textColor=BLUE, spaceBefore=12,
                    spaceAfter=5, keepWithNext=1)
H3 = ParagraphStyle("H3", fontName="DVB", fontSize=10.5, leading=14, textColor=INK, spaceBefore=7,
                    spaceAfter=3, keepWithNext=1)
BODY = ParagraphStyle("B", fontName="DV", fontSize=9.6, leading=14, textColor=INK, spaceAfter=5)
SMALL = ParagraphStyle("SM", parent=BODY, fontSize=8.2, leading=11.5, textColor=MUTED)
CELL = ParagraphStyle("C", parent=BODY, fontSize=8.4, leading=11.2, spaceAfter=0)
CELLB = ParagraphStyle("CB", parent=CELL, fontName="DVB")
CAP = ParagraphStyle("CAP", parent=SMALL, spaceBefore=2, spaceAfter=8)

AUTHOR_HTML = ("<b>KEYUR SHAILESHKUMAR PATEL</b> · ORCID "
               "<link href='https://orcid.org/0009-0008-5261-043X' color='#2a78d6'>0009-0008-5261-043X</link><br/>"
               "1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK<br/>"
               "2) Independent researcher<br/>"
               "<i>Conducted independently; not an official Brunel University London project.</i>")

AI_USE = ("<b>Use of AI assistance.</b> The research idea, research questions and every main design decision "
          "are the author's own. The author chose each decision from options with stated trade-offs, and decisions "
          "are logged in the project history. The code, preregistration draft and documentation were drafted with "
          "help from an AI assistant (Claude, by Anthropic), working under the author's direction. The author "
          "reviewed and approved all content and takes full responsibility for it. The AI assistant is not an author.")


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BODY):
    return ListFlowable([ListItem(Paragraph(t, style), leftIndent=12) for t in items], bulletType="bullet",
                        start="•", leftIndent=12, bulletFontSize=9, bulletColor=BLUE, bulletFontName="DV")


def box(text, bg=SOFT, width=W):
    t = Table([[Paragraph(text, BODY)]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def table(rows, widths, zebra=False, status_col=None):
    data = [[Paragraph(str(c), CELLB if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
          ("BACKGROUND", (0, 0), (-1, 0), SOFT), ("TOPPADDING", (0, 0), (-1, -1), 4),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 5),
          ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if status_col is not None:
        for i, r in enumerate(rows[1:], 1):
            v = str(r[status_col]).upper()
            if v.startswith("PASS") or v.startswith("SUPPORTED") or v.startswith("YES"):
                st.append(("BACKGROUND", (status_col, i), (status_col, i), GOOD))
            elif v.startswith("FAIL") or v.startswith("NOT SUPPORTED"):
                st.append(("BACKGROUND", (status_col, i), (status_col, i), WARM))
    t.setStyle(TableStyle(st))
    return t


def figure(path, caption, width=W):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    return KeepTogether([Image(str(path), width=width, height=width * ih / iw), P(caption, CAP)])


def footer(label):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("DV", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(2.2 * cm, 1.2 * cm, label)
        canvas.drawRightString(A4W - 2.2 * cm, 1.2 * cm, f"page {doc.page}")
        canvas.restoreState()
    return draw


A4W = 21.0 * cm
SP = Spacer
