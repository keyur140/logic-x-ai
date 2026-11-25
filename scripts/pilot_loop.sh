#!/usr/bin/env bash
# Logic x AI study (logicxai) - Author: KEYUR SHAILESHKUMAR PATEL (ORCID https://orcid.org/0009-0008-5261-043X)
# Pilot driver: re-runs the Groq pilot with the same run id until every call succeeds.
# Answers are cached, so each pass only asks for what is still missing.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
RUN_ID="${1:-pilot-groq-20261107}"
CACHE=/mnt/project-files/logic-x-ai/pilot/_cache
export LOGICXAI_MAX_RETRIES=40 LOGICXAI_MAX_SLEEP=600
for pass in $(seq 1 30); do
  echo "== pass $pass $(date -u +%FT%TZ)"
  python -m logicxai run --config configs/pilot_groq.yaml --profile pilot --cache "$CACHE" --run-id "$RUN_ID" --out runs
  errs=$(python -c "import json;m=json.load(open('runs/$RUN_ID/manifest.json'));print(m['stages']['trials']['api_errors'])")
  echo "== pass $pass done, api_errors=$errs"
  [ "$errs" = "0" ] && break
  sleep 600
done
echo "== loop finished $(date -u +%FT%TZ)"
