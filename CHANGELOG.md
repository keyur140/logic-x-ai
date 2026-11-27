# Changelog

## 0.2.1 (2026-11-25)

Built from the real Groq pilot (2026-11-07 to 11-21) and the author's decisions 15 to 21. See DEVIATIONS D6 to D12.

- **Harder puzzles:** hard deduction now has 4 difficulty levels (`items.HARD_LEVELS`). Level 1 is byte-identical to
  v0.2.0; levels 3 and 4 add "either/or" (proof by cases) steps. Answer keys use a fast exact truth table
  (`entailment_status_fast`) and are re-checked by the independent SAT solver (C2). Planted errors balanced per level.
- **Lenient number reading:** last number on the FINAL line. Gate G1 has 7 new known-answer cases (36 in total, with the 5 error-flag cases).
- **Thinking models:** `thinking: true` pools a model's 3 modes into one solver unit and adds a `thinking` control.
  Hidden reasoning (`<thought>` tags, e.g. Gemma 4) is stripped before scoring and before a solver sees an assistant
  reply; it is kept in the cache.
- **Free APIs over days:** several keys per provider rotate (`GROQ_API_KEY`, `_2`, `_3`, `_4`); a daily limit marks
  that key as spent (state in the cache folder) and the run pauses with exit code 3 instead of logging failures.
  `run --until-done` waits and continues by itself. New providers: Google AI Studio, Mistral, Cerebras.
- **New commands:** `plan` (calls, tokens and days under the free limits); `check` is parallel with short timeouts.
- **New profile:** `repilot` (hard items only, 4 free-API solvers, direct and careful modes); `scripts/repilot_summary.py`.
- **Model list:** 22 local (Ollama) + 6 free-API solvers; assistants gpt-oss-120b and Gemma 4 31B.
- **Fixes:** leak scan (C9) false alarm on item text; mediation guard; statsmodels pinned; benchmark revisions pinned.
- **H2 is now a registered secondary** (DEVIATIONS D16): Holm family = H1 and H4; H2 reported unadjusted.
- **Assistant reply shape (G5 fix):** scripted replies need 6 numbered steps and 450 to 800 characters
  (`prompts.SCRIPT_STEPS`, `SCRIPT_CHARS`), up to 6 tries. Synthetic assistants follow the same shape.
- **Re-pilot (2026-11-25/26)** on development items: `repilot`, `repilot_gemma` and `repilot_assist` profiles;
  report in `docs/Repilot_Report_v021.pdf`.
- Gemma 4 thinking allowance 4,096 tokens; request timeout settable (`LOGICXAI_TIMEOUT`).
- Tests: 43 (was 22).

## 0.2.0 (2026-11-05)

- **Primary outcome is now the gain score** for H1, H2 and H4 (DEVIATIONS D4). ANCOVA kept as sensitivity.
- New: known-answer gates G1 to G7 with pre-declared consequences; catch items for G6.
- New: self-checks C1 to C12 in every run, including an independent SAT solver (`independent.py`) and the
  traditional syllogism rules for every answer key, and statsmodels re-computation of the main regression and Holm.
- New: `freeze`, `status` and `package` commands; registered items are locked until `MATCHES_FREEZE`.
- New: development seed for pilot and synthetic runs, with an overlap check against registered items.
- New: `validate` command (false-alarm rate and power in worlds with known truth) and the `mock_null` profile.
- New: multiverse / specification curve (exploratory), minimum detectable effect in every run.
- New: easy-language exploratory report and project brief (`docs/`), with charts; synthetic outputs in
  `exploratory_synthetic/`.
- New: `route: api | local` switch; Ollama digests recorded by `check`.
- Fixed: synthetic (mock) replies were cached across worlds, so a no-effect run could reuse planted-world
  replies. Synthetic replies are never cached now.
- Fixed: synthetic models detected catch items by line count instead of item family.
- Tests: 22 (was 14).

## 0.1.0 (2026-10-26)

- First complete package: items and logic engine, assistant with planted errors, 28 solver models,
  ANCOVA analysis with wild cluster bootstrap and Holm, reports, manifests, SHA-256 for every file.
