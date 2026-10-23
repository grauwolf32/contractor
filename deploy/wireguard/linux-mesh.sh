#!/usr/bin/env bash
# Private network only. No default-route, forwarding or global firewall changes.
set -euo pipefail
test "$(id -u)" = 0
config=/etc/wireguard/wg-mesh.conf
grep -q '^# Managed by Contractor WireGuard mesh$' "$config"
case "$1" in
  up)
    if test "$#" -gt 1 && test -n "$2"; then
      python3 - "$2" <<'PY'
from pathlib import Path
import shutil
import sys
p = Path('/etc/apt/sources.list.d/ubuntu.sources')
original = p.read_text()
old = 'https://repo.mng.sbercloud.tech/repository/ubuntu'
new = sys.argv[1].rstrip('/')
if old in original:
    backup = p.with_name('ubuntu.sources.contractor-backup')
    if not backup.exists():
        shutil.copy2(p, backup)
    p.write_text(original.replace(old, new))
elif new not in original:
    raise SystemExit('refusing to replace an unrecognized package mirror')
PY
    fi
    if ! command -v wg-quick >/dev/null; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=2 update -qq
      apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=2 install -y -qq wireguard-tools
    fi
    if ip link show wg-mesh >/dev/null 2>&1; then
      wg syncconf wg-mesh <(wg-quick strip "$config")
    fi
    systemctl enable --now wg-quick@wg-mesh
    ;;
  status)
    systemctl is-active --quiet wg-quick@wg-mesh
    wg show wg-mesh latest-handshakes
    ;;
  *)
    printf '%s\n' 'usage: mesh.sh up|status' >&2
    exit 2
    ;;
esac
