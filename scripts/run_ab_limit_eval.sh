#!/usr/bin/env bash
# A/B eval: default read limits vs new (50KB OR 2000 lines, whichever first).
# Arm A = current defaults. Arm B = FS_MAX_OUTPUT=50000 + FS_MAX_READ_LINES=2000.
# Same 15 trace cases + 4 OAS fixtures both arms. Project default model.
set -uo pipefail
cd /home/ruslan/src/contractor

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
  if [ "$newlimits" = "1" ]; then
    export FS_MAX_OUTPUT=50000 FS_MAX_READ_LINES=2000
  else
    unset FS_MAX_OUTPUT FS_MAX_READ_LINES
  fi
  echo "############# ARM=$name  FS_MAX_OUTPUT=${FS_MAX_OUTPUT:-<default 80000>} FS_MAX_READ_LINES=${FS_MAX_READ_LINES:-<none>}  $(date) #############"
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
