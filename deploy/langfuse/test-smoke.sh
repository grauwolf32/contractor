#!/bin/sh
# Disposable real-backend interoperability check; this does not deploy a viewer.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/../.." && pwd)
cd "$repo_dir"
exec "$repo_dir/runtime/.venv/bin/python" "$script_dir/smoke.py"
