#!/usr/bin/env bash
# pass@3 A/B on the one evaluable trace case with large files in-repo.
# default 80KB byte cap vs new 50KB. 3 attempts/arm.
# NOTE: the new 50KB/2000-line caps are now the committed Settings defaults, so
# both arms pin the 2000-line cap and this isolates the BYTE cap. The pre-PR
# "no line cap" baseline is no longer env-reproducible (no None sentinel).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CONTRACTOR_RUN_EVAL=1 DEFAULT_MODEL_TIMEOUT=600
NODE="tests/eval/test_trace_agent_eval.py::test_trace_agent[cvebench-cve-2024-5084/hashform-file-upload-rce]"
OUT=eval_runs_pass3
rm -rf "$OUT"; mkdir -p "$OUT"

run_attempt () {
  local arm="$1" i="$2"
  # Default arm must EXPORT 80000 — unsetting now falls back to the new 50000
  # default and the arms would be identical.
  export FS_MAX_READ_LINES=2000
  if [ "$arm" = "new" ]; then export FS_MAX_OUTPUT=50000
  else export FS_MAX_OUTPUT=80000; fi
  rm -rf eval_runs
  echo "=== arm=$arm attempt=$i  FS_MAX_OUTPUT=${FS_MAX_OUTPUT} FS_MAX_READ_LINES=${FS_MAX_READ_LINES}  $(date +%H:%M:%S) ==="
  poetry run pytest "$NODE" -p no:cacheprovider -q --no-header -o addopts="" >"$OUT/$arm-$i.log" 2>&1
  local rc=$?
  echo "  pytest rc=$rc (0=pass, 1=fail)"
  mkdir -p "$OUT/$arm-$i"
  cp -r eval_runs/trace_agent/cases "$OUT/$arm-$i/cases" 2>/dev/null || true
  echo "$rc" > "$OUT/$arm-$i/rc.txt"
  # did the read cap actually fire this attempt?
  if grep -rqs "resume with read_file offset\|truncated at line" "$OUT/$arm-$i/cases" 2>/dev/null; then
    echo "  read-truncation FIRED this attempt"; echo fired > "$OUT/$arm-$i/truncated.txt"
  else
    echo "  no read-truncation this attempt"; echo none > "$OUT/$arm-$i/truncated.txt"
  fi
}

for i in 1 2 3; do run_attempt default "$i"; done
for i in 1 2 3; do run_attempt new "$i"; done

echo; echo "######## AGGREGATE ########"
for arm in default new; do
  pass=0; fired=0
  for i in 1 2 3; do
    [ "$(cat "$OUT/$arm-$i/rc.txt" 2>/dev/null)" = "0" ] && pass=$((pass+1))
    [ "$(cat "$OUT/$arm-$i/truncated.txt" 2>/dev/null)" = "fired" ] && fired=$((fired+1))
  done
  echo "$arm: pass@3 = $pass/3   (read-truncation fired in $fired/3 attempts)"
done
echo "######## DONE $(date) ########"
