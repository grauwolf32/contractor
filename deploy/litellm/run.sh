#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
image=${CONTRACTOR_LITELLM_IMAGE:-ghcr.io/berriai/litellm:main-stable@sha256:2e8517e2bed423c50ab7e40fb1ac0a9cbe62a764e9d65161d871fb6a9bf75a2d}
container_name=${CONTRACTOR_LITELLM_CONTAINER:-contractor-litellm}
upstream_url=${CONTRACTOR_LM_STUDIO_URL:-http://192.168.1.217:1234/v1}
upstream_token=${CONTRACTOR_LM_STUDIO_TOKEN:-lm-studio}

exec podman run --rm --name "$container_name" --network host \
  -e CONTRACTOR_LM_STUDIO_URL="$upstream_url" \
  -e CONTRACTOR_LM_STUDIO_TOKEN="$upstream_token" \
  -v "$script_dir/litellm_config.yaml:/app/config.yaml:ro,Z" \
  "$image" \
  --config /app/config.yaml --host 127.0.0.1 --port 4000
