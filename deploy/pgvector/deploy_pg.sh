#!/usr/bin/env bash
# Bring up Postgres + pgvector for the artifact-pool RAG backend and print the
# DSN to drop into cli/.env (RAG_DB_DSN).
#
# Usage:
#   ./deploy_pg.sh up      # start (default)
#   ./deploy_pg.sh down    # stop, keep data
#   ./deploy_pg.sh nuke    # stop and delete the data volume
#   ./deploy_pg.sh dsn     # just print the DSN
#
# Honors PGVECTOR_USER / PGVECTOR_PASSWORD / PGVECTOR_DB / PGVECTOR_PORT
# (same defaults as docker-compose.yml). Works with either `docker compose`
# or `podman compose`.
set -euo pipefail

cd "$(dirname "$0")"

PGVECTOR_USER="${PGVECTOR_USER:-contractor}"
PGVECTOR_PASSWORD="${PGVECTOR_PASSWORD:-contractor}"
PGVECTOR_DB="${PGVECTOR_DB:-contractor_rag}"
PGVECTOR_PORT="${PGVECTOR_PORT:-5433}"
export PGVECTOR_USER PGVECTOR_PASSWORD PGVECTOR_DB PGVECTOR_PORT

if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v podman-compose >/dev/null 2>&1; then
  COMPOSE="podman-compose"
elif podman compose version >/dev/null 2>&1; then
  COMPOSE="podman compose"
else
  echo "error: need 'docker compose' or 'podman compose' on PATH" >&2
  exit 1
fi

dsn() {
  echo "postgresql://${PGVECTOR_USER}:${PGVECTOR_PASSWORD}@localhost:${PGVECTOR_PORT}/${PGVECTOR_DB}"
}

case "${1:-up}" in
  up)
    $COMPOSE up -d
    echo "pgvector is starting on port ${PGVECTOR_PORT}."
    echo "Add this to cli/.env:"
    echo "  RAG_DB_DSN=$(dsn)"
    ;;
  down)
    $COMPOSE down
    ;;
  nuke)
    $COMPOSE down -v
    ;;
  dsn)
    dsn
    ;;
  *)
    echo "usage: $0 {up|down|nuke|dsn}" >&2
    exit 1
    ;;
esac
