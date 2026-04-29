# syntax=docker/dockerfile:1.7
#
# Contractor runtime image.
#
# Excluded by design: the LiteLLM proxy (separate deployment, configured via
# LITELLM_API_BASE) and the local-only CLI (the CLI is left buildable for
# debugging but is not the production entrypoint — k8s Jobs/Deployments should
# override CMD with whatever worker/server entrypoint we add).

ARG PYTHON_VERSION=3.13
ARG NODE_VERSION=22.11.0

# ─── builder ─────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.1.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry-cache

# build-essential: tree-sitter / native deps may compile from sdist
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Resolve deps first for cache friendliness.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --without dev --no-root

# Then the source.
COPY contractor ./contractor
COPY cli ./cli
RUN poetry install --without dev

# ─── runtime ─────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
ARG NODE_VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:/usr/local/lib/node/bin:${PATH}" \
    CONTRACTOR_ARTIFACTS_DIR=/data/artifacts

# Node.js + likec4 (used by contractor/tools/likec4.py).
# Drop this block if the likec4 tool is removed from your build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils \
    && curl -fsSLO "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
    && mkdir -p /usr/local/lib/node \
    && tar -xJf "node-v${NODE_VERSION}-linux-x64.tar.xz" -C /usr/local/lib/node --strip-components=1 \
    && rm "node-v${NODE_VERSION}-linux-x64.tar.xz" \
    && npm install -g likec4 \
    && apt-get purge -y curl xz-utils \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.npm

# Non-root, gid=0 so OpenShift / arbitrary-uid policies still work.
RUN useradd --system --uid 1001 --gid 0 --create-home --shell /sbin/nologin contractor \
    && mkdir -p /data/artifacts \
    && chown -R 1001:0 /data \
    && chmod -R g=u /data

WORKDIR /app
COPY --from=builder --chown=1001:0 /app /app

USER 1001:0

# Placeholder. The k8s Job/Deployment spec overrides this once the
# worker/server entrypoint lands. For local sanity:
#   docker run --rm <image> --help
ENTRYPOINT ["python", "-m", "cli.main"]
CMD ["--help"]
