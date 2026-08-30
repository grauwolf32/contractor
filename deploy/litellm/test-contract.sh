#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
run_id=$$
network_name="contractor-litellm-contract-$run_id"
database_name="contractor-litellm-contract-db-$run_id"
proxy_name="contractor-litellm-contract-proxy-$run_id"
volume_name="contractor-litellm-contract-data-$run_id"
litellm_image='ghcr.io/berriai/litellm:main-stable@sha256:2e8517e2bed423c50ab7e40fb1ac0a9cbe62a764e9d65161d871fb6a9bf75a2d'
postgres_image='docker.io/library/postgres:17-alpine@sha256:18cfe3ef5e6815560c98237d6216d1e5119702fb0f3894c8785dd58b8bbe5d73'
admin_key='sk-contractor-pinned-contract-admin'

cleanup() {
  podman rm -f "$proxy_name" >/dev/null 2>&1 || true
  podman rm -f "$database_name" >/dev/null 2>&1 || true
  podman volume rm "$volume_name" >/dev/null 2>&1 || true
  podman network rm "$network_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

podman network create "$network_name" >/dev/null
podman volume create "$volume_name" >/dev/null
podman run -d --name "$database_name" --network "$network_name" \
  -e POSTGRES_PASSWORD=contract-test \
  -e POSTGRES_DB=litellm \
  -v "$volume_name:/var/lib/postgresql/data:Z" \
  "$postgres_image" >/dev/null

database_ready=false
for attempt in $(seq 1 30); do
  if podman exec "$database_name" pg_isready -U postgres -d litellm >/dev/null 2>&1; then
    database_ready=true
    break
  fi
  sleep 1
done
if [ "$database_ready" != true ]; then
  echo 'pinned LiteLLM contract database did not become ready' >&2
  exit 1
fi

podman run -d --name "$proxy_name" --network "$network_name" \
  -p 127.0.0.1::4000 \
  -e LITELLM_MASTER_KEY="$admin_key" \
  -e LITELLM_SALT_KEY=sk-contractor-pinned-contract-salt \
  -e DATABASE_URL="postgresql://postgres:contract-test@$database_name:5432/litellm" \
  -e CONTRACTOR_LM_STUDIO_URL=http://192.168.1.217:1234/v1 \
  -e CONTRACTOR_LM_STUDIO_TOKEN=lm-studio \
  -v "$script_dir/litellm_config.yaml:/app/config.yaml:ro,Z" \
  "$litellm_image" \
  --config /app/config.yaml --host 0.0.0.0 --port 4000 >/dev/null

published_address=$(podman port "$proxy_name" 4000/tcp)
contract_url="http://$published_address"
proxy_ready=false
for attempt in $(seq 1 60); do
  if curl --silent --fail --max-time 2 "$contract_url/health/liveliness" >/dev/null; then
    proxy_ready=true
    break
  fi
  sleep 1
done
if [ "$proxy_ready" != true ]; then
  echo 'digest-pinned LiteLLM proxy did not become ready' >&2
  exit 1
fi

cd "$repository_root"
CONTRACTOR_LITELLM_CONTRACT_URL="$contract_url" \
CONTRACTOR_LITELLM_CONTRACT_ADMIN_KEY="$admin_key" \
  go test -race -count=1 ./internal/credentials/litellm -run '^TestPinnedLiteLLMContract$'
