# Pilot deviations: run pilot-groq-20261107 (real models, development items)

**Author:** KEYUR SHAILESHKUMAR PATEL · ORCID https://orcid.org/0009-0008-5261-043X
**Affiliations:** 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; 2) Independent researcher
**Note:** Conducted independently; not an official Brunel University London project.

This file lists every way the first real pilot differed from the study plan (v0.2.0). The pilot used
development items only (seed 7110526). The registered items were not shown to any model, nothing was frozen,
and `configs/study.yaml`, the prompts and the item generators were not changed. The numbers match section 7 of the pilot report (P5 there covers P5a and P5b here). The pilot has its own config,
`configs/pilot_groq.yaml`.

| # | Date | Change | Reason | Effect |
|---|---|---|---|---|
| P1 | 2026-11-07 | Solvers: 3 Groq-hosted models (allam-2-7b, gpt-oss-20b, qwen3.8-27b) x 3 modes, instead of the first 6 models of the planned list | Ollama, OpenRouter and the Hugging Face router could not be reached from the cloud session. Author's decision. | 9 solver units from 3 models. Far too few to test H1, H2 or H4. |
| P2 | 2026-11-07 | Assistant: gpt-oss-120b standing in for Llama 3.3 70B. No second (replication) assistant. | Groq retired llama-3.3-70b-versatile. The Qwen 2.5 72B route (OpenRouter) was blocked. | Assistant results may not carry over to the planned assistant. |
| P3 | 2026-11-07 | gpt-oss models: `reasoning_effort: low` and 512 extra output tokens. qwen3.8-27b: `reasoning_effort: none`. | Hidden reasoning counted against the answer length cap and returned empty answers. | "Direct" mode is not truly direct for gpt-oss models. |
| P4 | 2026-11-07 | LogiQA loaded from its parquet revision (fa9f9918). All four benchmark revisions pinned in the pilot config. | The dataset's loading script is no longer supported. | Same items, different loading path. |
| P5a | 2026-11-07 | Code: per-model request settings and token allowance; retry count and longest wait settable (`LOGICXAI_MAX_RETRIES`, `LOGICXAI_MAX_SLEEP`); one worker pool per model; restart script `scripts/pilot_loop.sh`. | Free-tier limits (per minute and per day, per model). | Scheduling only. Answers are not affected. |
| P5b | 2026-11-12 | Code: the H4 mediation test now needs enough complete rows and at least 3 models; a test that cannot run is reported as "Not testable", not "Not supported". | With 3 solver units from 1 model the unguarded test printed "H4 supported, p = 0.0002", which was meaningless. Added after that result was seen. | H1, H2 and H4 are all reported as "Not testable" for this run. |
| P6 | 2026-11-12, filled 2026-11-20 to 11-21 | 449 of 4,002 solver answers were missing on 12 Nov (qwen3.8-27b careful mode: 369; qwen3.8-27b knowledge quiz: 80). They were filled in on 20-21 Nov with new Groq keys from two Groq accounts, using the same run id, config and answer cache; only missing calls were sent. The run is now complete (4,002 of 4,002, 0 API errors). | The first Groq API key expired on 2026-11-12 at 02:42 UTC; the replacement key then hit qwen's daily token cap. | The qwen answers come from a later date (8-14 days after the others). Groq does not let us pin model versions, so a silent model update in between cannot be ruled out; the reported model id and system fingerprint are kept per reply. 83 calls from 20 Nov were made in a cloud session whose call log was lost; their log lines were rebuilt from the answer cache and marked `reconstructed_from_cache`. |
| P7 | 2026-11-07 to 11-21 | Answers were collected over 5 days with pauses, plus the fill-in on 20-21 Nov. The run folder's start time is that of the last pass. | Daily token caps and cloud session pauses. | Each reply keeps its own timestamp in `logs/llm_calls.jsonl`. API model versions cannot be pinned; reported model ids are recorded. |
| P8 | 2026-11-07, 11-11 | Two interim looks at the data (after 1,380 and 3,197 answers) were shared with the author. | Progress checks during a multi-day run. | No design change was made from them. |

**Use of AI assistance.** The research idea, research questions and every main design decision are the
author's own. The code, this file and the pilot report were drafted with help from an AI assistant (Claude, by
Anthropic), working under the author's direction. The author reviewed and approved all content and takes full
responsibility for it. The AI assistant is not an author.
