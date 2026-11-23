#!/bin/sh
# The helper's separate go.mod/go.sum pin the actual upstream implementation.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
go -C "$root/tests/interactsh-server" build -mod=readonly -o "$work/server" .
cd "$root/runtime"
CONTRACTOR_TEST_INTERACTSH_SERVER="$work/server" uv run pytest tests/test_interactsh_protocol.py
