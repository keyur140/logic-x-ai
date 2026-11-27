# Data card: Logic × AI run 20261127T145112Z-c1a1e9f8-mock_null

> TEST RUN with simulated models. Not study data.

## Author
- KEYUR SHAILESHKUMAR PATEL
- ORCID: https://orcid.org/0009-0008-5261-043X
- 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK
- 2) Independent researcher
- Conducted independently; not an official Brunel University London project.

## What is in this folder
- `items/` every test item (JSONL), each with its SHA-256
- `assistant/` every assistant reply shown to solvers, with planted-error flags
- `raw/trials.jsonl` every solver reply, full text, prompt hash and response hash
- `scored/` tidy tables: `trials.csv` (one row per trial), `solver_level.csv` (one row per solver)
- `analysis/` `results.json`, analysed table and figures
- `logs/` `run.log` and `llm_calls.jsonl` (every API call: model reported, latency, retries)
- `manifest.json` config hash, prompts hash, seed, software versions, models, stage hashes
- `SHA256SUMS.txt` checksum of every file (check with `sha256sum -c SHA256SUMS.txt`)

## Participants
No humans. Solvers are open-weight language models; see `manifest.json`.

## Item counts
- syllogism: 120
- conditional: 80
- crt: 30
- easy_math: 200
- hard_deduction: 300
- logic_creative: 40
- catch: 24
- folio: 200
- logiqa: 300
- gsm8k: 300
- mmlu: 400

## Benchmarks used
```
{
  "folio": {
    "repo": "SYNTHETIC stand-in (offline run)",
    "n": 200,
    "status": "synthetic"
  },
  "logiqa": {
    "repo": "SYNTHETIC stand-in (offline run)",
    "n": 300,
    "status": "synthetic"
  },
  "gsm8k": {
    "repo": "SYNTHETIC stand-in (offline run)",
    "n": 300,
    "status": "synthetic"
  },
  "mmlu": {
    "repo": "SYNTHETIC stand-in (offline run)",
    "n": 400,
    "status": "synthetic"
  }
}
```

## Licence
Code: MIT. Generated data and reports: CC BY 4.0. Benchmark items keep their own licences.
