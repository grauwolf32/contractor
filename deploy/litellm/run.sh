#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
litellm_image=${CONTRACTOR_LITELLM_IMAGE:-ghcr.io/berriai/litellm:main-stable@sha256:2e8517e2bed423c50ab7e40fb1ac0a9cbe62a764e9d65161d871fb6a9bf75a2d}
postgres_image=${CONTRACTOR_LITELLM_POSTGRES_IMAGE:-docker.io/library/postgres:17-alpine@sha256:18cfe3ef5e6815560c98237d6216d1e5119702fb0f3894c8785dd58b8bbe5d73}
proxy_name=${CONTRACTOR_LITELLM_CONTAINER:-contractor-litellm}
database_name=${CONTRACTOR_LITELLM_DATABASE_CONTAINER:-contractor-litellm-postgres}
database_volume=${CONTRACTOR_LITELLM_DATABASE_VOLUME:-contractor-litellm-postgres-data}
network_name=${CONTRACTOR_LITELLM_NETWORK:-contractor-litellm}
upstream_url=${CONTRACTOR_LM_STUDIO_URL:-http://192.168.1.217:1234/v1}
upstream_token=${CONTRACTOR_LM_STUDIO_TOKEN:-lm-studio}
master_key_file=${CONTRACTOR_LITELLM_MASTER_KEY_FILE:?CONTRACTOR_LITELLM_MASTER_KEY_FILE is required}
salt_key_file=${CONTRACTOR_LITELLM_SALT_KEY_FILE:?CONTRACTOR_LITELLM_SALT_KEY_FILE is required}

if [ ! -f "$master_key_file" ] || [ -L "$master_key_file" ] || [ ! -f "$salt_key_file" ] || [ -L "$salt_key_file" ]; then
  echo 'LiteLLM master and salt key paths must be regular non-symlink files' >&2
  exit 1
fi
master_key=$(sed -n '1p' "$master_key_file")
salt_key=$(sed -n '1p' "$salt_key_file")
case "$master_key" in sk-*) ;; *) echo 'LiteLLM master key must start with sk-' >&2; exit 1 ;; esac
case "$salt_key" in sk-*) ;; *) echo 'LiteLLM salt key must start with sk-' >&2; exit 1 ;; esac

if ! podman network exists "$network_name"; then
  podman network create "$network_name" >/dev/null
fi
podman volume create "$database_volume" >/dev/null
if podman container exists "$database_name"; then
  podman start "$database_name" >/dev/null
else
  podman run -d --name "$database_name" --network "$network_name" \
    -e POSTGRES_PASSWORD=contractor-litellm-local \
    -e POSTGRES_DB=litellm \
    -v "$database_volume:/var/lib/postgresql/data:Z" \
    "$postgres_image" >/dev/null
fi

database_ready=false
for attempt in $(seq 1 30); do
  if podman exec "$database_name" pg_isready -U postgres -d litellm >/dev/null 2>&1; then
    database_ready=true
    break
  fi
  sleep 1
done
if [ "$database_ready" != true ]; then
  echo 'persistent LiteLLM PostgreSQL did not become ready' >&2
  exit 1
fi

exec podman run --rm --name "$proxy_name" --network "$network_name" \
  -p 127.0.0.1:4000:4000 \
  -e LITELLM_MASTER_KEY="$master_key" \
  -e LITELLM_SALT_KEY="$salt_key" \
  -e DATABASE_URL="postgresql://postgres:contractor-litellm-local@$database_name:5432/litellm" \
  -e CONTRACTOR_LM_STUDIO_URL="$upstream_url" \
  -e CONTRACTOR_LM_STUDIO_TOKEN="$upstream_token" \
  -v "$script_dir/litellm_config.yaml:/app/config.yaml:ro,Z" \
  "$litellm_image" \
  --config /app/config.yaml --host 0.0.0.0 --port 4000
