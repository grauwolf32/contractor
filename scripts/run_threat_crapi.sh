#!/usr/bin/env bash
set -uo pipefail
cd /home/ruslan/src/contractor
export CONTRACTOR_RUN_EVAL=1 DEFAULT_MODEL_TIMEOUT=600
for slug in crapi-workshop crapi-identity; do
  echo "############# $slug $(date) #############"
  poetry run pytest \
    "tests/eval/test_threat_analysis_task_eval.py::test_threat_analysis_task[fixture[$slug]]" \
    -p no:cacheprovider -q --no-header -o addopts="" -s 2>&1 | tail -20
  echo "############# $slug done rc=$? $(date) #############"
done
echo "############# CRAPI THREAT RUNS COMPLETE $(date) #############"
