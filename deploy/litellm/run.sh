#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
litellm_image=${CONTRACTOR_LITELLM_IMAGE:-ghcr.io/berriai/litellm:main-stable@sha256:2e8517e2bed423c50ab7e40fb1ac0a9cbe62a764e9d65161d871fb6a9bf75a2d}
postgres_image=${CONTRACTOR_LITELLM_POSTGRES_IMAGE:-docker.io/library/postgres:17-alpine@sha256:18cfe3ef5e6815560c98237d6216d1e5119702fb0f3894c8785dd58b8bbe5d73}
proxy_name=${CONTRACTOR_LITELLM_CONTAINER:-contractor-litellm}
database_name=${CONTRACTOR_LITELLM_DATABASE_CONTAINER:-contractor-litellm-postgres}
database_volume=${CONTRACTOR_LITELLM_DATABASE_VOLUME:-contractor-litellm-postgres-data}
network_name=${CONTRACTOR_LITELLM_NETWORK:-contractor-litellm}
upstream_key_file=${CONTRACTOR_OPENROUTER_API_KEY_FILE:?CONTRACTOR_OPENROUTER_API_KEY_FILE is required}
master_key_file=${CONTRACTOR_LITELLM_MASTER_KEY_FILE:?CONTRACTOR_LITELLM_MASTER_KEY_FILE is required}
salt_key_file=${CONTRACTOR_LITELLM_SALT_KEY_FILE:?CONTRACTOR_LITELLM_SALT_KEY_FILE is required}
password_file=${CONTRACTOR_LITELLM_POSTGRES_PASSWORD_FILE:?CONTRACTOR_LITELLM_POSTGRES_PASSWORD_FILE is required}

read_secret() {
  python3 - "$1" <<'PY'
import pathlib, stat, sys
try:
    path = pathlib.Path(sys.argv[1])
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or mode & 0o077:
        raise ValueError()
    with path.open() as stream:
        value = stream.read(4097).strip()
    if not value or len(value) > 4096 or any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError()
except (OSError, ValueError, UnicodeError):
    raise SystemExit('LiteLLM credentials require regular owner-only files with one nonempty value')
print(value)
PY
}
master_key=$(read_secret "$master_key_file")
salt_key=$(read_secret "$salt_key_file")
upstream_key=$(read_secret "$upstream_key_file")
database_password=$(read_secret "$password_file")
case "$master_key" in sk-*) ;; *) echo 'LiteLLM master key must start with sk-' >&2; exit 1 ;; esac
case "$salt_key" in sk-*) ;; *) echo 'LiteLLM salt key must start with sk-' >&2; exit 1 ;; esac
case "$upstream_key" in sk-or-v1-?*) ;; *) echo 'OpenRouter inference key must start with sk-or-v1-' >&2; exit 1 ;; esac
case "$database_password" in *[!A-Za-z0-9_-]*) echo 'PostgreSQL password must use URL-safe letters, digits, underscores or hyphens' >&2; exit 1 ;; esac

# Pass both bootstrap password and DSN by environment name, including DB creation.
export POSTGRES_PASSWORD="$database_password"
export DATABASE_URL="postgresql://postgres:$database_password@$database_name:5432/litellm"

if ! podman network exists "$network_name"; then
  podman network create "$network_name" >/dev/null
fi
podman volume create --ignore "$database_volume" >/dev/null
if podman container exists "$database_name"; then
  podman start "$database_name" >/dev/null
else
  podman run -d --name "$database_name" --network "$network_name" \
    -e POSTGRES_PASSWORD \
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

# Podman reads these from its environment; secret values must not appear in
# its long-lived command line while the foreground proxy is running.
export LITELLM_MASTER_KEY="$master_key"
export LITELLM_SALT_KEY="$salt_key"
export OPENROUTER_API_KEY="$upstream_key"
exec podman run --rm --name "$proxy_name" --network "$network_name" \
  -p 127.0.0.1:4000:4000 \
  -e LITELLM_MASTER_KEY \
  -e LITELLM_SALT_KEY \
  -e DATABASE_URL \
  -e OPENROUTER_API_KEY \
  -v "$script_dir/litellm_config.yaml:/app/config.yaml:ro,Z" \
  "$litellm_image" \
  --config /app/config.yaml --host 0.0.0.0 --port 4000
