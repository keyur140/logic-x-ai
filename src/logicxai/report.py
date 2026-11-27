# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

"""Write the human-readable reports for a run: report.html, report.pdf,
data_card.md. All numbers come straight from analysis/results.json."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from . import AUTHOR


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    return str(x)


def _verdict(h: str, c: dict[str, Any]) -> str:
    if any("not testable" in x for x in c.get("labels", [])):
        return "Not testable (too few solvers)"
    if c.get("supported"):
        return "Supported"
    return "Not supported"


def _rows(results: dict[str, Any]) -> list[list[str]]:
    conf = results.get("confirmatory", {})
    desc = {
        "H1": "Amplification: when the AI makes planted errors, higher-logic solvers gain more "
              "(or lose less) from the AI",
        "H2": f"Levelling: on easy, checkable tasks logic adds less than "
              f"+{results.get('equivalence_bound_pp')} points per SD to the AI gain",
        "H4": "Mechanism: the logic advantage runs through catching AI errors (d′)",
    }
    out = []
    for h in ("H1", "H2", "H4"):
        c = conf.get(h, {})
        est = c.get("estimate_pp_per_sd") if h != "H4" else c.get("indirect")
        name = h if c.get("role", "confirmatory") == "confirmatory" else f"{h} (secondary)"
        out.append([name, desc[h], _f(est, 2), _f(c.get("p_raw"), 4),
                    _f(c.get("p_holm"), 4) if c.get("p_holm") is not None else "not in Holm family",
                    _verdict(h, c)])
    return out


def author_lines() -> list[str]:
    if not AUTHOR:
        return ["Author: (not set)"]
    lines = [AUTHOR["name"]]
    if AUTHOR.get("orcid"):
        lines.append(f"ORCID: {AUTHOR['orcid']}")
    for i, a in enumerate(AUTHOR.get("affiliations", []), 1):
        lines.append(f"{i}) {a}")
    if AUTHOR.get("note"):
        lines.append(AUTHOR["note"])
    return lines


def write_reports(run_dir: Path, manifest: dict[str, Any], results: dict[str, Any]) -> None:
    rep = run_dir / "reports"
    rep.mkdir(exist_ok=True)
    _html(rep / "report.html", run_dir, manifest, results)
    try:
        _pdf(rep / "report.pdf", run_dir, manifest, results)
    except Exception as e:  # PDF is a convenience; HTML is the full report
        (rep / "report_pdf_error.txt").write_text(str(e), encoding="utf-8")
    _data_card(rep / "data_card.md", manifest, results)


def _html(path: Path, run_dir: Path, m: dict[str, Any], r: dict[str, Any]) -> None:
    esc = html.escape
    rows = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in _rows(r))
    figs = "".join(f'<figure><img src="../analysis/{p.name}" alt="{esc(p.stem)}"></figure>'
                   for p in sorted((run_dir / "analysis").glob("fig*.png")))
    mock_banner = ('<p class="warn"><b>TEST RUN with simulated (mock) models.</b> These numbers only '
                   'check the pipeline. They are not study results.</p>') if m["profile"].startswith("mock") else ""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Logic x AI run report</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--accent:#2a78d6;--warn:#fff4d6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3a37;--accent:#3987e5;--warn:#3a3220}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3a37;--accent:#3987e5;--warn:#3a3220}}
body{{background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;max-width:920px;margin:0 auto;padding:24px 16px}}
h1{{font-size:1.5rem;margin:0 0 4px}} h2{{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid var(--line);padding-bottom:4px}}
.muted{{color:var(--muted)}} table{{border-collapse:collapse;width:100%;font-size:.9rem}}
td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
.warn{{background:var(--warn);padding:10px 12px;border-radius:6px}} img{{max-width:100%;height:auto}}
pre{{white-space:pre-wrap;font-size:.8rem;background:rgba(127,127,127,.08);padding:10px;border-radius:6px;overflow-x:auto}}
.wrap{{overflow-x:auto}}
</style></head><body>
<h1>Who gains most from AI? Logic × AI run report</h1>
<p class="muted">{'<br>'.join(esc(x) for x in author_lines())}</p>
{mock_banner}
<p>Run <b>{esc(m['run_id'])}</b> · profile <b>{esc(m['profile'])}</b> · started {esc(m['started_utc'])} ·
{r.get('n_solvers')} solvers ({r.get('n_models')} models × reasoning modes) · primary assistant
<b>{esc(str(r.get('primary_assistant')))}</b></p>
<h2>Confirmatory results (preregistered, Holm-corrected)</h2>
<div class="wrap"><table><tr><th>Hyp.</th><th>Claim</th><th>Estimate</th><th>p (raw)</th><th>p (Holm)</th><th>Verdict</th></tr>{rows}</table></div>
<p class="muted">Estimates for H1/H2 are percentage points of AI gain (score with the AI minus score alone,
same items) per 1 SD of logic score, adjusting for knowledge (MMLU) and model size. H4 is the indirect effect through error-detection d′. p-values: wild cluster
bootstrap (H1, H2) and cluster bootstrap (H4), clustered by model.</p>
<h2>Replication with the second assistant</h2>
<pre>{esc(json.dumps(r.get('replication', {}), indent=2))}</pre>
<h2>Figures</h2>{figs or '<p class="muted">No figures.</p>'}
<h2>Gates (G1 to G7) and self-checks (C1 to C12)</h2>
<pre>{esc(json.dumps({'gates_run': m.get('gates'), 'gates_analysis': r.get('gates'),
                 'self_checks': m.get('self_checks')}, indent=2, default=str))}</pre>
<h2>Measurement checks</h2>
<pre>{esc(json.dumps({'validity': r.get('validity'), 'manipulation_check': r.get('manipulation_check'),
                 'quality': r.get('quality')}, indent=2))}</pre>
<h2>Descriptives</h2><pre>{esc(json.dumps(r.get('descriptives'), indent=2))}</pre>
<h2>Run provenance</h2>
<pre>{esc(json.dumps({k: m.get(k) for k in ('run_id', 'package_version', 'config_sha256', 'prompts_sha256',
                                        'seed', 'started_utc', 'finished_utc', 'duration_s')}, indent=2))}
Stage hashes:
{esc(json.dumps(m.get('stages'), indent=2))}</pre>
<p class="muted">Every file in this run folder is listed with its SHA-256 in SHA256SUMS.txt.
Check it with: <code>python -m logicxai verify {esc(str(run_dir))}</code></p>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def _pdf(path: Path, run_dir: Path, m: dict[str, Any], r: dict[str, Any]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    story = [Paragraph("Who gains most from AI? Logic × AI run report", ss["Title"])]
    for ln in author_lines():
        story.append(Paragraph(html.escape(ln), ss["Normal"]))
    story.append(Spacer(1, 8))
    if m["profile"].startswith("mock"):
        story.append(Paragraph("<b>TEST RUN with simulated (mock) models. Not study results.</b>", ss["Normal"]))
    story.append(Paragraph(f"Run {m['run_id']} · profile {m['profile']} · {r.get('n_solvers')} solvers · "
                           f"primary assistant {r.get('primary_assistant')}", ss["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Confirmatory results (Holm-corrected)", ss["Heading2"]))
    data = [["Hyp.", "Claim", "Estimate", "p raw", "p Holm", "Verdict"]]
    for row in _rows(r):
        data.append([row[0], Paragraph(html.escape(row[1]), ss["BodyText"])] + row[2:])
    t = Table(data, colWidths=[1.2 * cm, 7.3 * cm, 2 * cm, 1.9 * cm, 1.9 * cm, 2.3 * cm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.grey),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
    story.append(t)
    for p in sorted((run_dir / "analysis").glob("fig*.png")):
        story.append(Spacer(1, 10))
        story.append(Image(str(p), width=16 * cm, height=9.6 * cm))
    story.append(Paragraph("Provenance", ss["Heading2"]))
    story.append(Paragraph(f"Config SHA-256: {m['config_sha256']}<br/>Prompts SHA-256: {m['prompts_sha256']}<br/>"
                           f"Seed: {m['seed']}<br/>All files are listed with SHA-256 in SHA256SUMS.txt.",
                           ss["BodyText"]))
    SimpleDocTemplate(str(path), pagesize=A4, title="Logic x AI run report",
                      author=AUTHOR.get("name", "") if AUTHOR else "").build(story)


def _data_card(path: Path, m: dict[str, Any], r: dict[str, Any]) -> None:
    counts = m.get("stages", {}).get("items", {}).get("counts", {})
    lines = ["# Data card: Logic × AI run " + m["run_id"], "",
             "## Author", *[f"- {x}" for x in author_lines()], "",
             "## What is in this folder",
             "- `items/` every test item (JSONL), each with its SHA-256",
             "- `assistant/` every assistant reply shown to solvers, with planted-error flags",
             "- `raw/trials.jsonl` every solver reply, full text, prompt hash and response hash",
             "- `scored/` tidy tables: `trials.csv` (one row per trial), `solver_level.csv` (one row per solver)",
             "- `analysis/` `results.json`, analysed table and figures",
             "- `logs/` `run.log` and `llm_calls.jsonl` (every API call: model reported, latency, retries)",
             "- `manifest.json` config hash, prompts hash, seed, software versions, models, stage hashes",
             "- `SHA256SUMS.txt` checksum of every file (check with `sha256sum -c SHA256SUMS.txt`)", "",
             "## Participants", "No humans. Solvers are open-weight language models; see `manifest.json`.", "",
             "## Item counts", *[f"- {k}: {v}" for k, v in counts.items()], "",
             "## Benchmarks used", "```", json.dumps(m.get("stages", {}).get("items", {}).get("benchmarks", {}), indent=2), "```",
             "", "## Licence", "Code: MIT. Generated data and reports: CC BY 4.0. Benchmark items keep their own licences.",
             ""]
    if m["profile"].startswith("mock"):
        lines.insert(2, "> TEST RUN with simulated models. Not study data.\n")
    path.write_text("\n".join(lines), encoding="utf-8")
