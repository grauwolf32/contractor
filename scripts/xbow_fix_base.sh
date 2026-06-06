#!/usr/bin/env bash
# Make the buster-based XBOW benchmarks buildable.
#
# ~10 of the validation-benchmarks build FROM python:2.7.18-slim (Debian buster).
# buster is EOL: deb.debian.org/security.debian.org return 404 for it, so the
# benchmarks' `apt-get install` step fails with exit 100. This rebuilds a local
# python:2.7.18-slim whose apt sources point at archive.debian.org (buster main
# only; security/updates dropped) with the expired-Release check disabled — so
# `FROM python:2.7.18-slim` in the benchmarks resolves to the working image.
#
# Idempotent. Run once before an xbow batch. No fixture/submodule edits.
set -euo pipefail
ORIG="localhost/python27-orig:latest"
TARGET="docker.io/library/python:2.7.18-slim"

# Preserve a pristine copy of the upstream base the first time.
if ! podman image exists "$ORIG"; then
  podman image exists "$TARGET" || podman pull "$TARGET"
  podman tag "$TARGET" "$ORIG"
fi

tmp="$(mktemp -d)"
cat > "$tmp/Containerfile" <<'EOF'
FROM localhost/python27-orig:latest
RUN set -eux; \
  sed -i -e 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' \
         -e '/security\.debian\.org/d' \
         -e '/buster-updates/d' /etc/apt/sources.list; \
  printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99no-check-valid
EOF
podman build -t "$TARGET" "$tmp"
rm -rf "$tmp"
echo "patched $TARGET (buster -> archive.debian.org)"
