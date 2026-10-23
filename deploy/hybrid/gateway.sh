#!/usr/bin/env bash
set -euo pipefail
test "$(id -u)" = 0
base=/etc/contractor-hybrid
read -r litellm_image postgres_image dns < <(python3 - <<'PY'
import json
d=json.load(open('/etc/contractor-hybrid/gateway.json'))
print(d['images']['litellm'],d['images']['postgres'],d['dns'][0])
PY
)
case "$1" in
  install)
    export DEBIAN_FRONTEND=noninteractive
    if ! command -v podman >/dev/null; then
      apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=2 install -y -qq podman
    fi
    podman pull "$postgres_image" >/dev/null
    podman pull "$litellm_image" >/dev/null
    podman volume create --ignore contractor-hybrid-litellm-pgdata >/dev/null
    # Reapply changed input without restarting unchanged healthy services.
    fingerprint=$(sha256sum "$base"/litellm.env "$base"/postgres.env "$base"/litellm_config.yaml "$base"/gateway.json "$0" /etc/systemd/system/contractor-hybrid-{postgres,litellm}.service | sha256sum | cut -d ' ' -f 1)
    previous=$(cat "$base/gateway.applied" 2>/dev/null || true)
    systemctl daemon-reload
    if test "$fingerprint" != "$previous"; then
      systemctl stop contractor-hybrid-litellm
      systemctl restart contractor-hybrid-postgres
    fi
    systemctl enable --now contractor-hybrid-postgres contractor-hybrid-litellm
    /opt/contractor-hybrid/bin/gateway.sh check
    printf '%s\n' "$fingerprint" > "$base/gateway.applied"
    ;;
  postgres)
    exec podman run --rm --replace --name contractor-hybrid-postgres --network host \
      --memory 512m --cpus 0.5 --env-file "$base/postgres.env" \
      -v contractor-hybrid-litellm-pgdata:/var/lib/postgresql/data \
      "$postgres_image" -c listen_addresses=172.30.91.1 -p 54329
    ;;
  litellm)
    ready=false
    for unused in $(seq 1 90); do
      if podman exec contractor-hybrid-postgres pg_isready -U litellm -p 54329 -h 172.30.91.1 -d litellm >/dev/null 2>&1; then ready=true; break; fi
      sleep 2
    done
    test "$ready" = true
    exec podman run --rm --replace --name contractor-hybrid-litellm \
      --network ns:/run/netns/contractor-llm --dns "$dns" \
      --memory 1536m --cpus 1 --env-file "$base/litellm.env" \
      -v "$base/litellm_config.yaml:/app/config.yaml:ro" \
      "$litellm_image" --config /app/config.yaml --host 0.0.0.0 --port 4000
    ;;
  check)
    python3 - <<'PY'
from pathlib import Path
import json,time,urllib.request,urllib.error
values=dict(line.split('=',1) for line in Path('/etc/contractor-hybrid/litellm.env').read_text().splitlines())
request=urllib.request.Request('http://172.30.91.2:4000/v1/models',headers={'Authorization':'Bearer '+values['LITELLM_MASTER_KEY']})
for attempt in range(90):
    try:
        with urllib.request.urlopen(request,timeout=3) as response:
            data=json.load(response)
        print('LiteLLM ready; model aliases:',len(data['data']))
        break
    except (urllib.error.URLError,TimeoutError,ConnectionError):
        if attempt==89: raise SystemExit('LiteLLM did not become ready; inspect its protected journal')
        time.sleep(2)
PY
    ;;
  *) printf '%s\n' 'usage: gateway.sh install|postgres|litellm|check' >&2; exit 2 ;;
esac
