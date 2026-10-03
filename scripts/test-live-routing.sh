#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "$script_dir/.." && pwd)

: "${CONTRACTOR_LIVE_LLM_URL:=http://127.0.0.1:4000/v1}"
: "${CONTRACTOR_LIVE_LLM_MODEL:=planner-model}"
export CONTRACTOR_LIVE_LLM_URL CONTRACTOR_LIVE_LLM_MODEL

exec make -C "$repository_root" test-live-routing
