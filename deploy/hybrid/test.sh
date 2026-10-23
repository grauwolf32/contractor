#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_dir/runtime"
uv run --locked python -m unittest discover -s ../deploy/hybrid/tests -v
uv run --locked ruff check ../deploy/hybrid ../deploy/wireguard
uv run --locked ruff format --check ../deploy/hybrid ../deploy/wireguard
bash -n ../deploy/hybrid/gateway.sh ../deploy/wireguard/linux-mesh.sh ../deploy/wireguard/linux-egress.sh
