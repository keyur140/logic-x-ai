# Logic × AI: who gains most from AI?

**A preregistered computational study: does logical reasoning predict who gains most from an AI assistant?**

**Author:** KEYUR SHAILESHKUMAR PATEL · ORCID [0009-0008-5261-043X](https://orcid.org/0009-0008-5261-043X)
1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK
2) Independent researcher

*Conducted independently; not an official Brunel University London project.*

---

## What this study does

- **Solvers:** 28 free, open-weight language models × 3 reasoning modes (direct, brief, careful) = **84 solvers**. No human participants.
- **Moderator:** each solver's **logic score**, from belief-bias syllogisms, conditional inferences (MP/MT/AC/DA) and cognitive-reflection items.
- **Tasks:** done **alone** and **with an AI assistant** (Llama 3.3 70B; replication with Qwen 2.5 72B):
  - *easy, checkable* arithmetic, where the assistant answers normally (tests **H2 levelling**);
  - *hard* rule-chain deduction, where the assistant makes **one planted, subtle error on exactly half the items** (tests **H1 amplification** and **H4 error-catching**);
  - *logical creativity* (exploratory **H3**): valid and rare conclusions.
- **Exact scoring:** every answer key is computed by a logic engine (brute-force model checking), so no human or AI judges are needed.
- **Analysis:** solver-level **AI gain** (score with the AI minus score alone, same items), regressed on logic with knowledge (MMLU) and model size as controls. Clustered by model, with wild cluster bootstrap p-values and Holm correction across H1, H2 and H4. Full plan: [`preregistration/PREREGISTRATION.md`](preregistration/PREREGISTRATION.md).
- **Open science:** fixed seeds, temperature 0, every request and response hashed and cached, a run manifest, and `SHA256SUMS.txt` for every run.

## Safeguards (v0.2.0)

| Safeguard | What it does |
|---|---|
| Known-answer gates G1 to G7 | parser test, collapse and failure exclusions, planted-error manipulation check, surface-cue check (AUC ≤ .65), error-flag check on catch items, logic validity against FOLIO + LogiQA. Each has a consequence fixed in the preregistration |
| Self-checks C1 to C12 | key numbers recomputed a second, independent way in every run (SAT solver for keys, syllogism rules, statsmodels for the regression and Holm, hash checks, leak scan, lock check) |
| Dev / registered split | pilot and synthetic runs use the dev seed 7110526; registered items (seed 20261026) are locked until the freeze |
| Freeze and package | `freeze` hashes every study file; `status` says `MATCHES_FREEZE` or lists what changed; `package` builds a byte-reproducible zip whose SHA-256 goes on OSF |
| Pipeline validation | `validate` checks the false-alarm rate and power in worlds with known truth. It found that the earlier adjusted-score plan gave up to 99% false alarms, so the main outcome is now the gain score |
| Multiverse | 72 analysis variants per hypothesis, as an exploratory robustness check |

## Exploratory run on synthetic data

The whole pipeline was run on simulated models before any real data, in a world with a planted effect and a world with none. The outputs are in [`exploratory_synthetic/`](exploratory_synthetic/) and explained in easy language in [`docs/Exploratory_Report_SYNTHETIC.pdf`](docs/Exploratory_Report_SYNTHETIC.pdf). **These are not study results.** The project brief is [`docs/Project_Brief.pdf`](docs/Project_Brief.pdf).

## Quick start

You need Python 3.11+. For local models you also need [Ollama](https://ollama.com/download).

```bash
# 1. Unzip, open a terminal in the folder
cp .env.example .env          # then paste your free keys (Groq, OpenRouter, Hugging Face)

# 2. (optional, for local models) download the open models
python scripts/setup_ollama.py            # or --only-small for models up to 10B

# 3. Run everything: install, tests, offline test, model check, pilot
./run.sh                 # macOS / Linux
run.bat                  # Windows

# 4. Full preregistered study (after the pilot and the OSF registration)
./run.sh full
```

Or run the steps one at a time:

```bash
pip install -r requirements.txt && pip install -e .
python -m pytest -q                         # 22 tests: keys, independent checks, freeze, validation
python -m logicxai run --profile mock_quick # 1-minute offline smoke test with simulated models
python -m logicxai run --profile mock       # full-size synthetic run, planted-effect world (~3 min)
python -m logicxai run --profile mock_null  # full-size synthetic run, no-effect world (~3 min)
python -m logicxai validate                 # false-alarm rate and power in known worlds (~2 min)
python -m logicxai check                    # which models and routes answer
python -m logicxai run --profile pilot      # 6 models, small item counts
python -m logicxai power                    # power simulation used in the preregistration
python -m logicxai freeze                   # SHA-256 of every study file and the registered items
python -m logicxai status                   # MATCHES_FREEZE, or the list of changed files
python -m logicxai package                  # release zip + its SHA-256 (register both on OSF)
python -m logicxai run --profile full       # the registered study (refuses to run unless frozen)
python -m logicxai verify runs/<run id>     # check every file against its SHA-256
```

### Steps to preregister

1. `./run.sh` (tests, offline test, model check, pilot on dev items).
2. Fill the `[TO FILL]` items in the preregistration with the pilot numbers.
3. `python -m logicxai freeze`, then `python -m logicxai package`.
4. On OSF: upload the preregistration, the zip, and the exploratory report; paste the freeze SHA-256 and the zip SHA-256.
5. `./run.sh full`. Any later change to a study file makes `status` say `CHANGED_SINCE_FREEZE` and blocks the full run.

A crashed or interrupted run can be restarted with the same command. Answers already received come from the cache, so nothing is paid or asked twice and results are identical.

## What every run saves (`runs/<run id>/`)

| Path | Contents |
|---|---|
| `manifest.json` | author, run id, config and prompt hashes, seed, Python and package versions, models and routes, stage hashes, timing |
| `config/` | the exact `study.yaml` and `prompts.py` used |
| `items/` | every test item (JSONL), each with its SHA-256 |
| `assistant/` | every assistant reply shown to solvers, with planted-error flags and manipulation checks |
| `raw/trials.jsonl` | every solver reply in full, with prompt and response hashes |
| `scored/` | `trials.csv` (one row per trial), `solver_level.csv` (one row per solver) |
| `analysis/` | `results.json` (all tests), the analysed table, figures |
| `reports/` | `report.html`, `report.pdf`, `data_card.md` |
| `logs/` | `run.log`, `llm_calls.jsonl` (every API call: model reported, latency, retries) |
| `SHA256SUMS.txt` | checksum of every file (`sha256sum -c SHA256SUMS.txt` also works) |

Mock runs are clearly marked "TEST RUN" in every report and are never study data. Every run also records the gates (`manifest.json` → `gates`) and the self-checks (`manifest.json` → `self_checks`), and `scored/exclusions.csv` says which solvers were excluded and why.

## Project layout

```
configs/study.yaml          models, item counts, seeds, settings (frozen at registration)
src/logicxai/               the study code
  items.py logic_engine.py  seeded item generators and the exact logic engine
  prompts.py                every prompt (frozen)
  assistant.py              assistant replies, planted errors, manipulation checks
  llm.py                    one client for Groq / OpenRouter / Hugging Face / Ollama, with cache and logs
  scoring.py tables.py      exact scoring, d', logical creativity, solver-level table
  analysis.py               preregistered statistics
  report.py runner.py       reports, manifests, the pipeline
  power.py provenance.py    power simulation, SHA-256 tools
  checks.py independent.py  gates G1-G7, self-checks C1-C12, independent SAT solver and syllogism rules
  freeze.py validate.py     freeze / status / package, pipeline validation
  multiverse.py mock.py     specification curve, synthetic models (planted and null worlds)
preregistration/            OSF preregistration draft
docs/                       project brief, exploratory report, impact guide (PDFs) and their sources
exploratory_synthetic/      outputs of the synthetic exploratory run (not study data)
CHANGELOG.md DEVIATIONS.md ISSUES.md   what changed, every change to the plan, known issues
tests/                      unit tests
```

## Impact in easy words

*(Full version: [`docs/Impact_and_Recognition.pdf`](docs/Impact_and_Recognition.pdf).)*

- **The big question:** research disagrees on whether AI helps weaker people most (levelling) or stronger people most (amplifying). This study predicts **both, in different places**: AI levels when its work is easy to check, and amplifies good reasoners when it can be wrong, because they catch its mistakes. If the data agree, two literatures join into one rule with a measured reason.
- **Who it helps:** researchers (a clear rule and a free, reusable test kit); teachers (teach reasoning and checking as "AI literacy"); AI developers and AI safety (a direct measure of whether weaker agents can catch a stronger AI's errors, linked to *scalable oversight*); employers and policy (fair AI adoption); open science (a zero-cost, fully reproducible method).
- **What it adds:** the question moves from people to AI teams; a within-model cause test (changing how much a model reasons); error-catching measured properly with signal detection; and a new, exactly scored measure of **logical creativity**.
- **Every result is useful:** support, partial support, levelling everywhere, or a clean preregistered null.
- **Limits:** AI solvers are not humans. This is a controlled model of human-AI teamwork, and a human follow-up is the natural next step.

## Use of AI assistance

The research idea, research questions and every main design decision are the author's own. The author chose each decision from options with stated trade-offs, and decisions are logged in the project history. The code, preregistration draft and documentation were drafted with help from an AI assistant (Claude, by Anthropic), working under the author's direction. The author reviewed and approved all content and takes full responsibility for it. The AI assistant is not an author.

## How to cite

If you use this study, its code or its data, please cite:

> PATEL, KEYUR SHAILESHKUMAR (2026). *Who gains most from AI? Logical reasoning as an amplifier of LLM-assisted performance (Logic × AI)* (Version 0.2.0) [Computer software and preregistered study]. ORCID: https://orcid.org/0009-0008-5261-043X. Conducted independently; not an official Brunel University London project.

BibTeX:

```bibtex
@software{patel_2026_logicxai,
  author  = {PATEL, KEYUR SHAILESHKUMAR},
  title   = {Who gains most from AI? Logical reasoning as an amplifier of LLM-assisted performance (Logic x AI)},
  year    = {2026},
  version = {0.2.0},
  note    = {ORCID: 0009-0008-5261-043X. Conducted independently; not an official Brunel University London project.}
}
```

Machine-readable citation: [`CITATION.cff`](CITATION.cff). After you register on OSF and post the preprint, add the OSF DOI here.

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Generated items, run outputs, reports and the preregistration: CC BY 4.0. Public benchmark items (FOLIO, LogiQA, GSM8K, MMLU) keep their own licences.
