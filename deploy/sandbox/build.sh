#!/usr/bin/env bash
# Build the exploit-agent code-execution sandbox image.
# Usage: bash deploy/sandbox/build.sh [tag]
set -euo pipefail

TAG="${1:-contractor-sandbox:latest}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building ${TAG} from ${HERE}/Dockerfile ..."
podman build -t "${TAG}" "${HERE}"
echo "Done: ${TAG}"
