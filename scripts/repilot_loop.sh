#!/usr/bin/env bash
# Logic x AI study (logicxai) - Author: KEYUR SHAILESHKUMAR PATEL (ORCID https://orcid.org/0009-0008-5261-043X)
# v0.2.1 re-pilot driver (cloud run). Answers are cached in the shared pilot cache, so a
# restart continues where it stopped; --until-done waits out the free daily limits.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src LOGICXAI_MAX_RETRIES=10 LOGICXAI_MAX_SLEEP=300
RUN_ID="${1:-repilot-v021-20261125}"
python3 -m logicxai run --profile repilot --cache /mnt/project-files/logic-x-ai/pilot/_cache \
  --out /mnt/project-files/logic-x-ai/v0.2.1/runs --run-id "$RUN_ID" --until-done --wait-minutes 30
