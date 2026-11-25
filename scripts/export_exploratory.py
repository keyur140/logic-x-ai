# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)


"""Copy the small, reviewable outputs of the synthetic exploratory runs into
exploratory_synthetic/ (results, solver tables, manifests, figures, reports,
validation), with a SHA256SUMS.txt. The full run folders (raw replies, logs)
are too large for the package; anyone can recreate them with:

    python -m logicxai run --profile mock
    python -m logicxai run --profile mock_null
    python -m logicxai validate

Usage: python scripts/export_exploratory.py RUNS_DIR
(takes the latest finished mock, mock_null and validation folders in RUNS_DIR)
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from logicxai.provenance import write_sha256sums  # noqa: E402

KEEP = ["manifest.json", "SHA256SUMS.txt", "analysis/results.json", "analysis/multiverse.csv",
        "analysis/solver_level_analysed.csv", "scored/exclusions.csv", "reports/report.pdf",
        "reports/report.html", "reports/data_card.md"]


def latest(runs: Path, pattern: str) -> Path:
    c = sorted(p for p in runs.glob(pattern) if p.is_dir())
    if not c:
        raise SystemExit(f"nothing matching {pattern} in {runs}")
    return c[-1]


def main(runs: Path) -> Path:
    out = ROOT / "exploratory_synthetic"
    if out.exists():
        shutil.rmtree(out)
    pairs = {"amplify_world": latest(runs, "*-mock"), "null_world": latest(runs, "*-mock_null")}
    for name, src in pairs.items():
        dst = out / name
        for rel in KEEP + [p.relative_to(src).as_posix() for p in (src / "analysis").glob("fig*.png")]:
            if (src / rel).exists():
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / rel, dst / rel)
        # the run's own checksum list is kept as run_SHA256SUMS.txt (it covers the full run)
        (dst / "SHA256SUMS.txt").rename(dst / "run_SHA256SUMS.txt")
        (dst / "SOURCE_RUN.txt").write_text(src.name + "\n", encoding="utf-8")
    shutil.copytree(latest(runs, "validation-*"), out / "validation")
    (out / "validation" / "SHA256SUMS.txt").unlink(missing_ok=True)
    (out / "README.md").write_text(
        "# Exploratory run on SYNTHETIC data\n\n"
        "These files come from simulated (mock) models. They test the pipeline. "
        "They are NOT study results and say nothing about real AI models.\n\n"
        "- `amplify_world/`: a world where logic really helps catch AI errors (planted effect)\n"
        "- `null_world/`: a world where catching AI errors does not depend on logic (no effect)\n"
        "- `validation/`: Monte Carlo check of false-positive rate and power\n\n"
        "`run_SHA256SUMS.txt` in each world folder lists the checksums of the full original run.\n"
        "`SHA256SUMS.txt` here covers every file in this folder.\n", encoding="utf-8")
    write_sha256sums(out)
    return out


if __name__ == "__main__":
    print(main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs")))
