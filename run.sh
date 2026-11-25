#!/usr/bin/env bash
# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

# One-command run for macOS / Linux.
#   ./run.sh            -> install, self-test, quick offline mock run, model check, pilot run
#   ./run.sh full       -> the full preregistered study
set -euo pipefail
cd "$(dirname "$0")"
PROFILE="${1:-pilot}"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -e .
[ -f .env ] || { cp .env.example .env; echo "Created .env - paste your API keys into it, then run again."; exit 1; }
echo "== 1. Unit tests";            python -m pytest -q
echo "== 2. Offline pipeline test";  python -m logicxai run --profile mock_quick
echo "== 3. Which models answer";   python -m logicxai check || echo "Some models are not reachable (see above). They will be skipped or use their backup route."
echo "== 4. Time estimate";        python -m logicxai plan --profile "$PROFILE"
echo "== 5. Study run ($PROFILE)";  python -m logicxai run --profile "$PROFILE" --until-done
echo "Reports are in runs/<run id>/reports/"
