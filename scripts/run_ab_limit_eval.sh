#!/usr/bin/env bash
# A/B eval: 80KB byte cap (pre-PR baseline) vs new 50KB cap.
# Arm A (default) = FS_MAX_OUTPUT=80000. Arm B (new) = FS_MAX_OUTPUT=50000.
# NOTE: the new 50KB/2000-line caps are now the committed Settings defaults, so
# this A/B isolates the BYTE cap only; both arms run at the 2000-line default.
# The original "no line cap" baseline is no longer reproducible via env (the
# setting has no None sentinel), so don't read this as an old-vs-new line-cap test.
# Same 15 trace cases + 4 OAS fixtures both arms. Project default model.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

export CONTRACTOR_RUN_EVAL=1
export DEFAULT_MODEL_TIMEOUT=600   # applied to BOTH arms; cannot bias the A/B

TRACE=(
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[crapi-workshop/shop-apply-coupon-sqli]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[crapi-workshop/shop-order-get-bola]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[crapi-workshop/merchant-contact-mechanic-ssrf]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[crapi-identity/auth-login-credential-flow]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[crapi-identity/jwt-validate-alg-none-bypass]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[crapi-identity/vehicle-get-location-no-owner-check]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[crapi-identity/change-email-no-password-verification]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[crapi-community/coupon-validate-nosqli]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[crapi-community/jwt-verify-tls-and-unverified-claims]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[vulnyapi/notes-search-sqli]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[vulnyapi/files-upload-path-traversal]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[vulnyapi/admin-users-access-control]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[vaultpay/create-transfer]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[fastapi/login]"
  "tests/eval/test_trace_agent_eval.py::test_trace_agent[spring/send-message]"
)
OAS=(
  "tests/eval/test_oas_builder_eval.py::test_oas_builder_endpoint_coverage[fixture[fastapi]]"
  "tests/eval/test_oas_builder_eval.py::test_oas_builder_endpoint_coverage[fixture[spring]]"
  "tests/eval/test_oas_builder_eval.py::test_oas_builder_endpoint_coverage[fixture[vaultpay]]"
  "tests/eval/test_oas_builder_eval.py::test_oas_builder_endpoint_coverage[fixture[vulnyapi]]"
)
NODES=( "${TRACE[@]}" "${OAS[@]}" )

run_arm () {
  local name="$1" newlimits="$2"
  # Both arms pin the line cap to the committed default so the byte cap is the
  # only variable. The default arm must EXPORT 80000 explicitly — unsetting it
  # now falls back to the new 50000 default, which would make the arms identical.
  export FS_MAX_READ_LINES=2000
  if [ "$newlimits" = "1" ]; then
    export FS_MAX_OUTPUT=50000
  else
    export FS_MAX_OUTPUT=80000
  fi
  echo "############# ARM=$name  FS_MAX_OUTPUT=${FS_MAX_OUTPUT} FS_MAX_READ_LINES=${FS_MAX_READ_LINES}  $(date) #############"
  rm -rf eval_runs
  poetry run pytest "${NODES[@]}" -p no:cacheprovider -q --no-header -o addopts="" 2>&1
  echo "############# ARM=$name pytest exit=$? $(date) #############"
  rm -rf "eval_runs_$name"
  mv eval_runs "eval_runs_$name" 2>/dev/null || mkdir -p "eval_runs_$name"
  echo "############# ARM=$name results -> eval_runs_$name #############"
}

run_arm default 0
run_arm new 1
echo "############# A/B COMPLETE $(date) #############"
