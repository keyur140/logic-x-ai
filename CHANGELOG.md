# Changelog

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
