#!/bin/sh
# A local OTLP viewer. No Compose provider or external database is required.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/../.." && pwd)
data_dir="$repo_dir/.local/phoenix"
container_name=${CONTRACTOR_PHOENIX_CONTAINER:-contractor-phoenix}
host_port=${CONTRACTOR_PHOENIX_PORT:-6006}
image=docker.io/arizephoenix/phoenix:latest@sha256:c542e7de1dc1fdd0f938de337d44ebefa7266e91391eed6e7978523d7f6e1f2f

if [ "$#" -gt 1 ]; then
  echo 'Usage: deploy/phoenix/run.sh [up|down|status]' >&2
  exit 2
fi

exists() {
  if podman container exists "$container_name"; then
    return 0
  else
    code=$?
    # Podman's absent-container status is 1; engine errors must not become absence.
    [ "$code" -eq 1 ] || exit "$code"
    return 1
  fi
}

case "${1:-up}" in
  up)
    case "$host_port" in
      ''|*[!0-9]*) echo 'CONTRACTOR_PHOENIX_PORT must be an integer in 0..65535' >&2; exit 1 ;;
    esac
    if [ "${#host_port}" -gt 5 ] || [ "$host_port" -gt 65535 ]; then
      echo 'CONTRACTOR_PHOENIX_PORT must be an integer in 0..65535' >&2
      exit 1
    fi
    if [ "$host_port" -eq 0 ]; then
      host_port=
    fi
    if [ -L "$repo_dir/.local" ] || [ -L "$data_dir" ]; then
      echo 'Phoenix local data paths must not be symlinks' >&2
      exit 1
    fi
    mkdir -p "$data_dir"
    if exists; then
      podman start "$container_name" >/dev/null
    else
      podman run --detach --name "$container_name" \
        --publish "127.0.0.1:$host_port:6006" \
        --volume "$data_dir:/data:Z" \
        --env PHOENIX_WORKING_DIR=/data \
        --env PHOENIX_HOST=0.0.0.0 \
        --env PHOENIX_PORT=6006 \
        --env PHOENIX_ENABLE_AUTH=false \
        --env PHOENIX_TELEMETRY_ENABLED=false \
        --env PHOENIX_ALLOW_EXTERNAL_RESOURCES=false \
        --env PHOENIX_DISABLE_AGENT_ASSISTANT=true \
        --env PHOENIX_ENABLE_MCP_SERVER=false \
        --log-opt max-size=10mb \
        --health-cmd '["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:6006/healthz\", timeout=3)"]' \
        --health-interval 10s --health-timeout 5s --health-retries 3 \
        "$image" >/dev/null
    fi
    ready=false
    for attempt in $(seq 1 30); do
      if podman healthcheck run "$container_name" >/dev/null 2>&1; then
        ready=true
        break
      fi
      sleep 1
    done
    if [ "$ready" != true ]; then
      echo "Phoenix did not become healthy; inspect: podman logs $container_name" >&2
      exit 1
    fi
    address=$(podman port "$container_name" 6006/tcp)
    printf 'Phoenix UI: http://%s\nOTLP/HTTP: http://%s/v1/traces\nData: %s\n' "$address" "$address" "$data_dir"
    ;;
  down)
    if exists; then
      podman stop --time 30 "$container_name" >/dev/null
      podman rm "$container_name" >/dev/null
    fi
    printf 'Phoenix stopped; data retained in %s\n' "$data_dir"
    ;;
  status)
    if exists; then
      podman inspect --format '{{.State.Status}}' "$container_name"
    else
      echo 'Phoenix is stopped'
    fi
    ;;
  *) echo 'Usage: deploy/phoenix/run.sh [up|down|status]' >&2; exit 2 ;;
esac
