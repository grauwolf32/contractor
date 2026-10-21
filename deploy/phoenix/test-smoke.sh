#!/bin/sh
# Real-image OTLP/persistence check, isolated from the actual local viewer.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/../.." && pwd)
mkdir -p "$repo_dir/.local"
fixture_dir=$(mktemp -d "$repo_dir/.local/phoenix-smoke.XXXXXX")
export CONTRACTOR_PHOENIX_CONTAINER="contractor-phoenix-smoke-$$"
export CONTRACTOR_PHOENIX_PORT=0
cleanup() {
  podman rm --force "$CONTRACTOR_PHOENIX_CONTAINER" >/dev/null 2>&1 || true
  rm -rf "$fixture_dir"
}
trap cleanup 0
trap 'exit 130' 1 2 15
mkdir -p "$fixture_dir/deploy/phoenix"
cp "$script_dir/run.sh" "$fixture_dir/deploy/phoenix/run.sh"
run() { sh "$fixture_dir/deploy/phoenix/run.sh" "$@"; }
run up
podman exec --interactive "$CONTRACTOR_PHOENIX_CONTAINER" python - send < "$script_dir/smoke.py"
podman exec --interactive "$CONTRACTOR_PHOENIX_CONTAINER" python - check < "$script_dir/smoke.py"
run up >/dev/null
run down
run up
podman exec --interactive "$CONTRACTOR_PHOENIX_CONTAINER" python - check < "$script_dir/smoke.py"
echo 'Phoenix OTLP ingestion, repeated start and SQLite retention across replacement passed'
