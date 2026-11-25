# Known issues and open items

| # | Issue | Status / plan |
|---|---|---|
| I1 | Free API model versions cannot be pinned | Reported model id logged per call; local Ollama route records exact digests. Stated in the preregistration (§6.4) |
| I2 | The CR1 t-test is slightly liberal with about 25 clusters (6.7% in the fast check) | The registered test is the wild cluster bootstrap (6.0%, within Monte Carlo error of 5%). The multiverse uses the t-test and is exploratory |
| I3 | In the synthetic planted-effect world, H1 was positive but not significant (+4.7 pp per SD, p = .058) | Reported as is, not tuned. The synthetic link is U-shaped (random responders at chance). Real-study power is in the brief |
| I4 | Benchmark revisions are empty in `configs/study.yaml` | Fill in the Hugging Face revisions after the pilot and before the freeze |
| I5 | Pilot variance estimates not yet available | Re-run `python -m logicxai power` with pilot values and fill §3.4 of the preregistration |
| I6 | Background references in the preregistration marked `[TO FILL]` | Add full references after reading the papers |
