#!/usr/bin/env bash
# Logic x AI study (logicxai)
# Author: KEYUR SHAILESHKUMAR PATEL
# ORCID: https://orcid.org/0009-0008-5261-043X
# Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
#               2) Independent researcher
# Note: Conducted independently; not an official Brunel University London project.
# License: MIT (see LICENSE)

# v0.2.1 re-pilot driver. Answers are cached, so a restart continues where it stopped;
# --until-done waits out the free daily limits.  Usage: bash scripts/repilot_loop.sh [RUN_ID]
# CACHE and OUT can be set to keep the cache and runs somewhere else.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src LOGICXAI_MAX_RETRIES="${LOGICXAI_MAX_RETRIES:-10}" LOGICXAI_MAX_SLEEP="${LOGICXAI_MAX_SLEEP:-300}"
RUN_ID="${1:-repilot-v021-20261125}"
python3 -m logicxai run --profile repilot --cache "${CACHE:-.cache/llm}" \
  --out "${OUT:-runs}" --run-id "$RUN_ID" --until-done --wait-minutes 30
