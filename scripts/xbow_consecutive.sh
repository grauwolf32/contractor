#!/usr/bin/env bash
# Run XBOW exploit benchmarks one-at-a-time, resiliently:
#  - each benchmark in its own pytest process (a crash can't cascade to the next)
#  - model health-checked before each; reloaded (safe context) if it died
#  - per-benchmark wall-clock timeout; container torn down after each
# Usage: bash scripts/xbow_consecutive.sh <benchmark-list-file>
set -u
LIST="${1:-/tmp/xbow15.txt}"
MODEL="lm-studio-qwen3.6-27b-mtp"
MODEL_KEY="qwen3.6-27b-mtp"
LMS="$HOME/.lmstudio/bin/lms"
KEY="sk-litellm-changeme"
OBS='{"enabled":true,"include_tool_errors":false,"track_file_paths":true}'
PER_BENCH_TIMEOUT="${PER_BENCH_TIMEOUT:-900}"
BR="tests/playground/validation-benchmarks/benchmarks"
LOG="eval_runs/xbow_15_consecutive.log"
SUMMARY="eval_runs/xbow_15_summary.txt"
: > "$SUMMARY"
echo "xbow consecutive run: $(wc -l < "$LIST") benchmarks, model=$MODEL, per-bench timeout=${PER_BENCH_TIMEOUT}s" | tee -a "$LOG"

model_alive() {
  curl -sS -m 20 "http://localhost:4000/v1/chat/completions" \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":3}" \
    2>/dev/null | grep -q '"choices"'
}
ensure_model() {
  for _ in 1 2 3; do
    if model_alive; then return 0; fi
    echo "[wrapper] model not responding — (re)loading $MODEL_KEY @ safe context" | tee -a "$LOG"
    "$LMS" load "$MODEL_KEY" -c 65536 --parallel 1 -y >/dev/null 2>&1
    sleep 8
  done
  model_alive
}

for b in $(cat "$LIST"); do
  ensure_model || { echo "[wrapper] $b SKIPPED — model unrecoverable" | tee -a "$LOG" "$SUMMARY"; continue; }
  echo "[wrapper] ===== $b =====" | tee -a "$LOG"
  CONTRACTOR_RUN_EVAL=1 CONTRACTOR_EVAL_MODEL="$MODEL" CONTRACTOR_EVAL_OBSERVATIONS="$OBS" \
    CONTRACTOR_XBOW_BENCHMARKS="$b" CONTRACTOR_XBOW_AGENT=exploit \
    timeout "$PER_BENCH_TIMEOUT" poetry run pytest tests/eval/test_xbow_eval.py -s -q -k exploit >> "$LOG" 2>&1
  rc=$?
  line=$(grep -E "${b}.*flag_captured=|${b}.*HARNESS ERROR" "$LOG" | tail -1)
  if [ -z "$line" ]; then
    [ "$rc" = "124" ] && line="$b TIMEOUT (>${PER_BENCH_TIMEOUT}s)" || line="$b NO-RESULT (rc=$rc)"
  fi
  echo "$line" | tee -a "$SUMMARY"
  # teardown this benchmark's container (harness usually does; guard against SIGKILL leaks)
  podman-compose -f "$BR/$b/docker-compose.yml" -p "xbow-$(echo "$b" | tr 'A-Z' 'a-z')" down >/dev/null 2>&1
done

echo "[wrapper] DONE" | tee -a "$LOG"
echo "=== SUMMARY ===" | tee -a "$SUMMARY"
echo "captured: $(grep -c 'flag_captured=True' "$LOG")" | tee -a "$SUMMARY"
echo "missed:   $(grep -c 'flag_captured=False' "$LOG")" | tee -a "$SUMMARY"
echo "errors:   $(grep -c 'HARNESS ERROR' "$LOG")" | tee -a "$SUMMARY"
