# Preregistration (OSF Preregistration template)

**Title:** Who gains most from AI? Logical reasoning as an amplifier of LLM-assisted performance: a preregistered computational study with open-weight language models

**Author:** KEYUR SHAILESHKUMAR PATEL  
ORCID: https://orcid.org/0009-0008-5261-043X  
1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK  
2) Independent researcher  
  
*Conducted independently; not an official Brunel University London project.*

**Status:** FROZEN for registration (package v0.2.1). The pilot and re-pilot used development items only. This file is one of the hashed study files, so the freeze and package hashes are given on the OSF registration page and in `freeze/FREEZE.json`, not here. The code refuses to send the registered items to any model unless the freeze status is `MATCHES_FREEZE`.

---

## 1. Study information

### 1.1 Description

Workplace and laboratory studies disagree about who benefits most from large language models (LLMs). Some find that AI *levels* people (weaker performers gain most); others, on harder or open-ended tasks, find that AI *amplifies* the stronger. Logical reasoning ability has not been isolated as the moderator, and the proposed mechanism (stronger reasoners catch AI errors) has not been measured directly.

This study tests these questions computationally, with no human participants. The "solvers" are 28 open-weight language models, each run in three reasoning modes (direct answer, brief reasoning, careful step-by-step reasoning). Two models always reason at length whatever the mode prompt says ("thinking" models); their three modes are pooled into one solver, giving 80 solver units. Each solver's logical reasoning is measured on a logic battery. Each solver then completes matched tasks alone and with an AI assistant (gpt-oss-120b; replication with Gemma 4 31B). On hard-to-verify deduction tasks the assistant makes a planted, subtle error on exactly half of the items. On easy, checkable arithmetic tasks the assistant answers normally.

All correct answers are computed by an exact logic engine (brute-force model checking) or by arithmetic, so scoring needs no human or AI judge.

### 1.2 Research questions

- **RQ1.** Does a solver's logical reasoning predict how well it performs with an LLM assistant, beyond general knowledge and model size?
- **RQ2.** Does the answer differ between tasks where AI output is easy to check and tasks where it is hard to check or wrong?
- **RQ3 (exploratory).** Does logical creativity (generating conclusions that are both valid and rare) predict gains in the originality of AI-assisted work?
- **RQ4.** Is the logic advantage explained by detecting AI errors?

### 1.3 Hypotheses (confirmatory family: H1, H4; H2 registered secondary)

| # | Hypothesis | Directional prediction |
|---|---|---|
| **H1** Amplification | On hard deduction items where the assistant makes a planted error, higher-logic solvers have a larger **AI gain** (accuracy with the assistant minus accuracy alone, same items). | Slope of logic > 0 |
| **H2** Levelling *(registered secondary)* | On easy, checkable arithmetic items with a normal assistant, logic adds less than +5 percentage points of AI gain per 1 SD. | Slope of logic < +5 pp (non-superiority) |
| **H4** Mechanism | The logic advantage in H1 runs through error detection (signal-detection sensitivity d′ for flagging the assistant's errors). | Indirect effect logic → d′ → H1 outcome > 0 |

H1 and H2 together make the central claim: *logic matters most where the AI cannot be trusted blindly.*

**H2 is a registered secondary hypothesis, not part of the confirmatory family** (author's decision, 2026-11-27, before registration). With the variance seen in the pilot, the ±5 pp test has about 2% power (§3.4), so it cannot carry a confirmatory claim. H2 is still tested exactly as planned in §5.1 and always reported, with its unadjusted p-value and its 90% CI, but it does not enter the Holm correction and the central claim rests on H1 (and H4 for the mechanism). A non-significant H2 result will be described as *inconclusive*, never as evidence against levelling.

H3 (logical creativity → originality gain) is **exploratory**. H5 from the original plan (transfer to unassisted work) is dropped because language models do not learn between independent API calls.

---

## 2. Design plan

### 2.1 Study type
Computational experiment. No human participants, no personal data, no ethics approval required.

### 2.2 Blinding
Not applicable to model solvers. All scoring is automatic and rule-based; the analysis code is fixed before data collection.

### 2.3 Design
Within-solver design. Each solver (model × reasoning mode) completes:

1. **Logic battery, alone** (moderator): belief-bias syllogisms (valid/invalid × believable/unbelievable/neutral content), conditional inferences (MP, MT, AC, DA; real and nonsense content), and cognitive-reflection items with fresh numbers.
2. **Benchmarks, alone**: FOLIO and LogiQA (convergent validity of the logic score), GSM8K (validity of the easy task). MMLU (knowledge control) is run once per model in direct mode, with logic and mathematics subjects removed so the control measures knowledge rather than reasoning.
3. **Tasks, alone and with each assistant** (every item in every condition; language models have no memory between calls, so there are no order or carry-over effects):
   - *Easy, checkable:* multi-step arithmetic word problems with one integer answer. Assistant answers naturally.
   - *Hard to verify:* rule-chain deduction puzzles at four difficulty levels, 75 items each: level 1, 8 statements (4-step chain plus distractor rules); level 2, 12 statements (7-step chain); level 3, 16 statements (9-step chain with one "either/or" step, which can only be settled by checking both cases); level 4, 20 statements (11-step chain with two "either/or" steps). Answer TRUE / FALSE / UNKNOWN, balanced within each level. The levels were added after the pilot showed a ceiling (strong models ≥ 90% on level 1); the analysis averages over levels, which every solver sees equally. Assistant reply is scripted: a correct solution (clean) or a confident solution with exactly one subtle logical slip that reaches a specified wrong answer (planted error). Both kinds must have the same shape: exactly 6 numbered one-sentence steps and 450 to 800 characters, so that length and step count do not reveal the planted error.
   - *Logical creativity (exploratory):* list up to 10 distinct valid conclusions from 4–5 premises.
   - *Catch items (known-answer check, 24 items, assisted only):* the statements give the answer outright and the assistant contradicts it. Used only for gate G6 (§5.7), never in the hypothesis tests.

Assisted prompts state that the assistant "is often helpful but can make mistakes" and require the solver to end with `ASSISTANT_ERROR: YES/NO` and `FINAL: <answer>`.

### 2.4 Randomisation
- Planted errors: exactly 50% of hard items, chosen by a seeded hash ranking within each difficulty level × answer type (TRUE/FALSE/UNKNOWN) block. The same items carry errors for both assistants.
- The wrong target answer for each planted item is chosen uniformly at random (seeded) among the two incorrect options.
- **Registered items** are generated with the master seed `20261026`. **Development items** (pilot, synthetic runs) use a separate seed `7110526`. Every run checks that no development item equals a registered item (self-check C8).
- **Lock:** the code refuses to send registered items to any model unless the freeze status is `MATCHES_FREEZE` (every study file matches `freeze/FREEZE.json`).

### 2.5 Model settings
Temperature 0, top-p 1, fixed seed where the provider accepts one, fixed token limits per mode (direct 64, brief 384, careful 1536, assistant 1024; models whose hidden reasoning counts against the limit get a fixed extra allowance, listed in `configs/study.yaml`). Every request and response is hashed (SHA-256) and logged with the provider's reported model version. Hidden reasoning returned inside a reply (for example `<thought>` tags) is removed before scoring and before a solver sees an assistant reply, and is kept in the cache. Models up to about 14B parameters run locally with Ollama (exact digests recorded); larger models use free API tiers (Groq, Google AI Studio, Mistral), some with a local backup route; the route actually used is recorded per call. Cost: USD 0.

---

## 3. Sampling plan

### 3.1 Existing data
Registration takes place **before** any confirmatory data are collected. A pilot (3 models on the Groq free tier, small item counts, **development items only**; DEVIATIONS_PILOT P1) and a re-pilot of the difficulty levels (4 solver models, 96 hard items) were run before registration only to check timings, parsing, and ceiling/floor effects. Pilot data are not used in confirmatory analyses. Before registration the whole pipeline was also run on synthetic (mock) models in a planted-effect world and a no-effect world (§5.9); those outputs are attached as a labelled supplementary file.

### 3.2 Data collection procedure
`python -m logicxai run --profile full --until-done`. Free API tiers limit each key per day; when every key for a model has used its allowance the run pauses and continues later from the cache (several keys may rotate). A pause is not a failure and does not count toward the gate G3 limits. The full list of solver models, assistants, routes and parameter counts is in `configs/study.yaml`.

### 3.3 Sample size
- **Solvers:** 28 models. 26 models run in 3 reasoning modes (direct, brief, careful) and the 2 always-thinking models (gpt-oss-20b, Gemma 4 26B) are pooled into one unit each: 26 × 3 + 2 = **80 solver units**, clustered in 28 models.
- **Items:** syllogism 120, conditional 80, CRT 30, easy arithmetic 200, hard deduction 300 (150 planted, 150 clean), logical creativity 40, catch items 24; FOLIO 200, LogiQA 300, GSM8K 300, MMLU 400.

### 3.4 Sample size rationale
Precision for a solver-level moderator depends on the number of solvers, not items. A standardized equivalence bound of |β| < .05 would need about 3,400 solvers, so the H2 bound is set in raw units (5 percentage points per 1 SD of logic) and the solver count is tripled with reasoning modes. A simulation clustered by model (`python -m logicxai power`; assumptions: between-model SD of gain 8 pp, within-model SD 5 pp, reasoning-mode shift of 0.5 SD in logic) gives:
- H1: 80% power at about 4 pp per SD (92% at 5 pp per SD; one-sided α = .05);
- H2: 87% probability that the 90% CI falls within ±5 pp when the true slope is zero.

Each run also reports the minimum detectable H1 effect at 80% power from the observed standard error (`mde_H1_pp_per_sd` in `results.json`). In the synthetic planted-effect run it was 6.3 pp per SD.

**Updated with pilot variance (2026-11-26).** The pilot (3 models on Groq, development items) gave larger spreads than assumed: hard-item error gain between-model SD 7 pp, within-model SD 11 pp; easy-item gain between-model SD 15 pp, within-model SD 37 pp (direct-mode easy maths gains of 55 to 90 pp). Re-running the same simulation with these values gives:
- H1: 52% power at 3 pp per SD, 86% at 5 pp per SD, 99.7% at 8 pp per SD (one-sided α = .05). The study is therefore well powered for H1 effects of 5 pp per SD or more, and under-powered below about 4 pp per SD.
- H2: the probability that the 90% CI falls within ±5 pp when the true slope is zero is about 2% (60% at ±10 pp, 95% at ±15 pp). With the pilot variance the ±5 pp equivalence test is not informative; H2 is therefore a registered secondary hypothesis (§1.3).

The re-pilot of the difficulty ladder (`docs/Repilot_Report_v021.pdf`) confirmed that levels 3 and 4 pull the strongest pilot models off the ceiling (careful-mode accuracy alone 71% and 50% at level 4), which is what H1 needs.

### 3.5 Stopping rule
Data collection stops when every solver has completed every registered trial, or when a model is unavailable on all routes after 6 retries with exponential backoff (then handled as in §5.4). Daily-limit pauses do not stop data collection. If a model is retired by its provider during the run and has no backup route, its solver units are reported as missing (§5.4) and not replaced.

---

## 4. Variables

### 4.1 Manipulated variables
- Condition: alone vs assisted (within solver, within item).
- Assistant: gpt-oss-120b on Groq (primary), Gemma 4 31B on Google AI Studio (replication).
- Assistant correctness on hard items: clean vs planted error (within solver, between items).
- Reasoning mode: direct / brief / careful (defines solvers; also gives a within-model manipulation of reasoning). For the two thinking models the three modes are pooled into one solver unit and they do not enter the within-model analyses.
- Difficulty level of hard items: 1 to 4 (within solver, between items).

### 4.2 Measured variables
- Correctness per trial (exact match to the engine-computed key; unparsable answers are scored incorrect). Number answers are read leniently: the last number on the last `FINAL:` line (for example "FINAL: Tanya earned $365" is read as 365); with no `FINAL:` line, the last number in the reply.
- Error flag (`ASSISTANT_ERROR: YES/NO`); a missing flag counts as NO.
- For logical-creativity trials: the number of distinct valid conclusions (identity by truth table), invalid and trivial counts.

### 4.3 Indices
- **Logic score** (moderator): equal-weighted mean of solo accuracy on syllogisms, conditionals and CRT items, in the solver's own mode; z-scored across the solver units.
- **Knowledge control:** MMLU accuracy (per model), z-scored. **Size control:** log10 parameter count, z-scored. **Thinking control:** 1 for an always-thinking model, else 0 (included because both kinds are in the roster).
- **H1 outcome (AI gain, planted errors):** accuracy with the assistant minus solo accuracy on the same planted-error hard items, in percentage points.
- **H2 outcome (AI gain, easy):** accuracy with the assistant minus solo accuracy on the same easy items, in percentage points.
- **Error detection d′** (H4 mediator): z(hit rate) − z(false-alarm rate), where hits = flags on replies whose final answer is wrong and false alarms = flags on replies whose final answer is correct, with the log-linear correction (Hautus, 1995).
- **Logical creativity** (exploratory): mean over items of (number of valid distinct conclusions × their mean surprisal, −log₂ of the share of all replies to that item containing the conclusion).
- **Adjusted (ANCOVA) scores** (sensitivity only): accuracy with the assistant, in pp, adjusted for the same solver's solo accuracy on the same items (z). See §5.9 for why these are not primary.

---

## 5. Analysis plan

### 5.1 Statistical models
Unit = solver unit (n = 80: 26 models × 3 modes + 2 pooled thinking models); cluster = model (28). OLS with cluster-robust (CR1) standard errors. p-values from the wild cluster restricted bootstrap-t (Rademacher weights, 9,999 replications; Cameron, Gelbach & Miller, 2008).

- **H1:** `gain_hard_error ~ logic_z + mmlu_z + logparams_z + thinking`. One-sided test, slope of logic_z > 0.
- **H2 (registered secondary):** `gain_easy ~ logic_z + mmlu_z + logparams_z`. One-sided non-superiority test of H0: slope ≥ +5 against H1: slope < +5. The two one-sided tests against ±5 (TOST) are also reported, to state whether the effect is *equivalent to zero* (both bounds rejected) or *levelling* (CI entirely below 0).
- **H4:** a-path `dprime_z ~ logic_z + covariates`; b-path `gain_hard_error ~ dprime_z + logic_z + covariates` (covariates as in H1). Indirect effect a × b; one-sided p and percentile CI from a cluster (model) bootstrap with 4,999 resamples.

### 5.2 Inference criteria
Holm correction across H1 and H4 at familywise α = .05 (one-sided tests). A confirmatory hypothesis is supported when its Holm-adjusted p < .05. H2 (registered secondary) is judged on its unadjusted one-sided p < .05 and its 90% CI, outside the Holm family. Results with the primary assistant (gpt-oss-120b) decide the hypotheses. The Gemma 4 assistant is a registered replication: a hypothesis "replicates" if the Gemma 4 estimate has the same sign and its unadjusted p < .05.

### 5.3 Registered secondary analyses (not in the Holm family)
- S1–S2: H1 and H2 with adjusted (ANCOVA) scores. Reported as sensitivity analyses, labelled as biased upward when the solo score is noisy (§5.9).
- S3: contrast of gain slopes, hard-error vs easy (gain_hard_error − gain_easy ~ logic): the "flip" in RQ2.
- S4–S5: gains on all hard items and on clean hard items.
- S6–S7: **within-model** test: gains and logic centred within model (reasoning mode varies logic inside the same weights): does more reasoning cause bigger gains on planted-error items?
- S8: behavioural catch rate (correct final answer on planted-error items) ~ logic.
- Measurement validity: correlations of the logic score with FOLIO, LogiQA, MMLU, belief-bias index and size; standardized alpha of the three logic sub-scores; GSM8K vs easy-task solo accuracy.
- Manipulation check: assistant final answer wrong on 100% of planted items and correct on 100% of clean items (by construction; verified).

### 5.4 Data exclusion
- **Assistant replies:** a scripted reply must end with the target answer, contain none of the leak words listed in `prompts.py`, and have the fixed shape (6 numbered steps, 450 to 800 characters). After 6 failed attempts the item is excluded for that assistant only.
- **Solvers (gate G2, collapse):** a solver is excluded if it gives the same answer on more than 90% of the items in any multiple-choice family (it is not reasoning).
- **Solvers (gate G3):** a solver is excluded if, after retries, more than 10% of its trials fail at the API level, or more than 20% of its answers in any task family cannot be parsed. Excluded solvers are reported and not replaced. If fewer than 20 models remain, results are reported as underpowered.
- No outlier exclusion on scores.

### 5.5 Missing data
Unparsable answers are scored incorrect (sensitivity analysis: excluded). Missing error flags count as NO. Analyses use complete cases at solver level.

### 5.6 Exploratory analyses
- H3: originality gain ~ logical creativity + logic + covariates; valid-conclusion gain likewise.
- Mode-level descriptives; family-level (model family) patterns; belief-bias effects.
- AI gain and error detection by difficulty level (1 to 4); local vs API route.
- **Multiverse / specification curve** for H1 and H2: every combination of outcome (gain, ANCOVA), logic definition (battery; battery + FOLIO + LogiQA; syllogisms only), covariates (none; knowledge; knowledge + size), modes (all; without direct) and assistant (72 specifications per hypothesis), with cluster-robust t-tests. Reported as the share of specifications that agree with the registered one.

### 5.7 Known-answer gates (decided before data; consequences fixed)

| Gate | Check | If it fails |
|---|---|---|
| G1 | The answer parser reads 36 hand-written known replies correctly (31 answers + 5 error flags, including 7 lenient number cases) | The run stops |
| G2 | Collapse: > 90% same answer in any choice family | Solver excluded (§5.4) |
| G3 | > 20% unparsable answers in a family, or > 10% API failures | Solver excluded (§5.4) |
| G4 | Manipulation: every planted reply ends on its wrong target and every clean reply on the key; no leak words | Item excluded for that assistant |
| G5 | Surface cues: AUC of length, lines, steps, hedges and negations for telling planted from clean replies ≤ .65 | H1 and H4 carry a caution label |
| G6 | Error flag validity: flag rate on catch items above the false-alarm rate on clean replies (one-sided Wilcoxon, p < .05) | H4 is reported as UNINFORMATIVE |
| G7 | Logic validity: Spearman correlation of the logic score with FOLIO + LogiQA accuracy > 0 (p < .05) | H1 and H2 carry a caution label |

### 5.8 Self-checks (recomputed independently in every run)
C1 every reply matches its stored SHA-256; C2 deduction keys recomputed by an independent SAT solver; C3 syllogism keys recomputed by the traditional rules; C4 planted-error assignment re-derived from item ids and seed; C5 logic scores recomputed from raw replies; C6 main regression recomputed with statsmodels; C7 Holm correction recomputed with statsmodels; C8 development and registered items do not overlap; C9 no solver prompt contains a clue about the answer; C10 config and prompts match the manifest; C11 freeze status; C12 parser known-answer gate. A failed self-check is reported in the run report and the run's results are not used until it is resolved.

### 5.9 Pipeline validation (done before registration)
`python -m logicxai validate` simulates complete studies at the registered size (25 model clusters after gates, 150 planted-error items per solver) in three worlds with known truth. Results (2,000 simulations; 400 for the bootstrap test):

| World | True H1 effect | Registered test (gain, wild bootstrap) | Adjusted (ANCOVA) score |
|---|---|---|---|
| Solver ignores the assistant | none | 6.0% false alarms | 99.3% false alarms |
| Same catch rate for everyone | none | 0.0% | 67.9% |
| Logic raises error catching | yes | 100% power | 66.0% |

For H2, when the true slope equals the +5 pp bound, the non-superiority test made the false "levelling" claim 5.8% of the time (CR1 t-test). The ANCOVA's false alarms arise because the solo score is a noisy measure of an ability that the logic score measures more cleanly, so logic picks up the residual (Westfall & Yarkoni, 2016). For this reason the primary outcome was changed from the adjusted score to the gain score **before registration** (logged in `DEVIATIONS.md`).

The full pipeline was also run end to end on 84 synthetic solver units (28 mock models × 3 modes; mock models have no pooled thinking mode) (profiles `mock` and `mock_null`). In the no-effect world H1 and H4 were not supported (as expected). In the planted-effect world H4 was supported (Holm p = .005) and H1 was positive but not significant (+4.2 pp per SD, p = .095). All self-checks passed in both runs.

---

## 6. Other

### 6.1 Frozen materials (SHA-256)
`python -m logicxai freeze` writes `freeze/FREEZE.json` with the SHA-256 of every study file (code, config, prompts, tests, scripts, docs, this document, synthetic outputs) and of every registered item file. `python -m logicxai package` builds a byte-reproducible release zip.

The three hashes below cannot be written into this file, because this file is itself hashed. They are stated on the OSF registration page and can be recomputed with `python -m logicxai status`:
- Freeze file SHA-256 (`freeze/FREEZE.json`)
- Code SHA-256 (hash of all file hashes)
- Release zip SHA-256 (`logic-x-ai-study-v0.2.1.zip.sha256`)

Benchmark items are fixed by dataset revision (commit hash) and seed; revisions are pinned in `configs/study.yaml` before the full run.

### 6.2 Deviations
Every change is logged in `DEVIATIONS.md` with date, reason and effect. Changes made before registration (for example, the switch to gain scores in §5.9) are listed there too, so reviewers can see the history. Any deviation after registration will be listed in the final report with its reason and its effect on results.

### 6.3 Openness
Code: MIT licence. Generated items, raw replies, scored data and reports: CC BY 4.0, deposited on OSF with the SHA256SUMS file of each run.

### 6.4 Known limitations
- Language-model solvers are not humans; the study tests whether logic amplifies AI assistance *in reasoning agents*, as a controlled model of human–AI collaboration, not a claim about people.
- Model size, family and training data are confounded across models; the within-model reasoning-mode analysis (S6–S7) partly addresses this.
- Free API endpoints can change and their versions cannot be pinned; free daily limits make data collection take several weeks. Every reply is cached and hashed, the reported model version is logged per call, and Ollama model digests are recorded when the local route is used.
- Benchmark items may be in models' training data; for this reason the main tasks use freshly generated items.

### 6.5 Use of AI assistance
The research idea, research questions and every main design decision are the author's own. The author chose each decision from options with stated trade-offs, and decisions are logged in the project history. The code, preregistration draft and documentation were drafted with help from an AI assistant (Claude, by Anthropic), working under the author's direction. The author reviewed and approved all content and takes full responsibility for it. The AI assistant is not an author.

### 6.6 Statements
- **Data availability:** all code, items, raw replies, scored data and reports will be public on OSF.
- **Ethics:** no human participants and no personal data.
- **Author contributions (CRediT):** KEYUR SHAILESHKUMAR PATEL: conceptualization, methodology, software (with AI assistance as in §6.5), validation, formal analysis, writing.
- **Funding:** none. **Competing interests:** none.

### 6.7 References
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics, 90*(3), 414–427.
- Hautus, M. J. (1995). Corrections for extreme proportions and their biasing effects on estimated values of d′. *Behavior Research Methods, Instruments, & Computers, 27*(1), 46–51.
- Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science, 8*(4), 355–362.
- Vaccaro, M., Almaatouq, A., & Malone, T. (2024). When combinations of humans and AI are useful: A systematic review and meta-analysis. *Nature Human Behaviour, 8*, 2293–2303.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.
- Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour, 4*, 1208–1214.
- Westfall, J., & Yarkoni, T. (2016). Statistically controlling for confounding constructs is harder than you think. *PLoS ONE, 11*(3), e0152719.
- Doshi, A. R., & Hauser, O. P. (2024). Generative AI enhances individual creativity but reduces the collective diversity of novel content. *Science Advances, 10*(28), eadn5290.
- Background studies (cited only to motivate the questions; no design parameter depends on them): Luchini et al. (2025), PsyArXiv preprint; Asirvatham et al. (2026), CEPR Discussion Paper DP21882; Wu et al. (2026), HCOMP; Teng & Ren (2026), OSF preprint; Lee et al. (2025), CHI 2025; Kumar et al. (2025), CHI 2025. Full bibliographic details will be given in the final report.
