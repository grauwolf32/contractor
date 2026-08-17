#!/usr/bin/env bash
# serve.sh — serve a GGUF model via llama.cpp's llama-server (OpenAI-compatible).
#
# Auto-enables MTP self-speculative decoding for any model with "mtp" in its path.
# Keeps the whole model on the GPU (-ngl 99 --no-mmap) so SYSTEM RAM stays flat
# (the fix for the lm-studio OOM: it streamed MoE experts through system RAM).
#
# Usage:
#   ./serve.sh <search>           # e.g. ./serve.sh 27b-mtp   (substring match on GGUF path)
#   ./serve.sh -l | --list        # list available models
#   ./serve.sh --print <search>   # show the launch command, don't run it
#   ./serve.sh stop [--all]       # stop the server on $PORT (or all llama-server instances)
#   ./serve.sh <search> -- <extra llama-server args...>
#
# Config (all overridable via env):
#   MODELS_DIR    where GGUFs live           (default: /var/ai-models)
#   LLAMA_SERVER  the llama-server binary    (default: `llama-server` on PATH,
#                 else $LLAMA_HOME/build/bin/llama-server)
#   LLAMA_HOME    llama.cpp checkout          (default: ~/src/llama.cpp) — symlink-friendly:
#                 `ln -s /path/to/llama.cpp ~/src/llama.cpp` or put llama-server on PATH.
#   HOST PORT CTX NP NGL                      server tunables
#   BATCH UBATCH  prefill -b / -ub            (default: 512/512; per-slot buffers)
#   CTXCP         max context checkpoints     (default: 2; system-RAM KV cache)
#   TEMP TOP_P TOP_K MIN_P REPEAT PRESENCE    sampling (LM Studio-matched defaults)
#   ALIAS         model name on the API       (default: the model's folder name)
#   SPEC          force MTP                    (auto|on|off, default: auto by name)
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/var/ai-models}"
LLAMA_HOME="${LLAMA_HOME:-$HOME/src/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server 2>/dev/null || echo "$LLAMA_HOME/build/bin/llama-server")}"

# Accept human-friendly sizes: 128k -> 131072, 2m -> 2097152, plain int -> as-is.
to_int() { local v="${1,,}"; case "$v" in *k) echo $(( ${v%k} * 1024 ));; *m) echo $(( ${v%m} * 1024 * 1024 ));; *) echo "$v";; esac; }

HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-8081}"
CTX="$(to_int "${CTX:-128k}")"; NP="${NP:-2}"; NGL="${NGL:-99}"
# Context KV-snapshots cached in *system* RAM for context-shift reuse. The
# default (32) balloons host RSS on long agent runs with a large context (each
# snapshot is a partial KV of the growing context) and can exhaust RAM/swap.
# Cap low — context-shift is rare when the worker summarizes under the window.
CTXCP="${CTXCP:-2}"
# Prompt cache: DISABLED (0). llama-server snapshots slot KV states into *system*
# RAM for longest-common-prefix reuse across requests via `prompt_save`. The
# `--cache-ram N` limit caps only the *cumulative* cache, NOT an individual save:
# a single near-ceiling state (~175k tokens) is ~13GB and gets snapshotted in
# full regardless of the cap, OOM-killing the server (observed twice, even with
# cache-ram=4096). Disabling it (0) stops prompt_save entirely — we trade a
# prefill prefix-reuse speedup for not crashing multi-hour runs. (-1 = no limit.)
CACHE_RAM="${CACHE_RAM:-0}"
# Prefill batch sizes. `-b` (logical) / `-ub` (micro) size the compute and KV
# scratch buffers; with -np>1 these are allocated *per slot*, so large values
# (the old 2048/2048) balloon host+device buffers on long contexts. 512/512 is
# plenty for agent prefill and keeps the buffers small. Override via env.
BATCH="${BATCH:-512}"; UBATCH="${UBATCH:-512}"
# Sampling: matches LM Studio's defaults (which got config-B to a clean full
# cycle, F1 0.946, on the identical GGUF). The earlier narrow/greedy preset
# (temp 0.3, top_k 20, top_p 0.8, no repeat-penalty, presence 1.0) made the
# worker lock into non-terminating subtasks that ran context to the ceiling.
TEMP="${TEMP:-0.4}"; TOP_P="${TOP_P:-0.9}"; TOP_K="${TOP_K:-40}"
MIN_P="${MIN_P:-0.05}"; REPEAT="${REPEAT:-1.15}"; PRESENCE="${PRESENCE:-0.0}"
SPEC="${SPEC:-auto}"

list_models() { find "$MODELS_DIR" -type f -iname '*.gguf' ! -iname '*mmproj*' 2>/dev/null | sed "s#^$MODELS_DIR/##" | sort; }

case "${1:-}" in
  -l|--list) list_models; exit 0 ;;
  ""|-h|--help) sed -n '2,31p' "$0" | sed 's/^# \?//'; echo; echo "available models:"; list_models; exit 0 ;;
  stop)
    shift
    if [[ "${1:-}" == --all || "${1:-}" == -a ]]; then
      # match the resolved binary path, excluding this script's own PID.
      mapfile -t apids < <(pgrep -f "$LLAMA_SERVER" 2>/dev/null | grep -vx "$$" || true)
      if [[ ${#apids[@]} -gt 0 ]]; then kill "${apids[@]}" 2>/dev/null || true; echo "stopped llama-server (pids: ${apids[*]})"; else echo "no llama-server running"; fi
    else
      # find the PID actually LISTENING on $PORT — robust, no cmdline matching.
      pid=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)
      [[ -z "$pid" ]] && pid=$(fuser "$PORT/tcp" 2>/dev/null | tr -dc '0-9' || true)
      if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true; sleep 1
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        echo "stopped server on port $PORT (pid $pid)"
      else
        echo "nothing listening on port $PORT (try: $(basename "$0") stop --all)"
      fi
    fi
    exit 0 ;;
esac

PRINT_ONLY=0
[[ "$1" == "--print" ]] && { PRINT_ONLY=1; shift; }
QUERY="$1"; shift || true
EXTRA=(); [[ "${1:-}" == "--" ]] && { shift; EXTRA=("$@"); }

[[ -x "$LLAMA_SERVER" ]] || { echo "error: llama-server not found/executable at '$LLAMA_SERVER'" >&2; echo "set LLAMA_SERVER or LLAMA_HOME, or symlink llama-server into PATH." >&2; exit 1; }

# Resolve the GGUF: substring-match on path, skip mmproj. Prefer the first shard
# of a sharded model (…-00001-of-…), else the largest single match.
mapfile -t MATCHES < <(find "$MODELS_DIR" -type f -iname '*.gguf' ! -iname '*mmproj*' 2>/dev/null | grep -iF "$QUERY" | sort)
[[ ${#MATCHES[@]} -gt 0 ]] || { echo "error: no GGUF matching '$QUERY' under $MODELS_DIR" >&2; echo "try: $(basename "$0") --list" >&2; exit 1; }
MODEL=""
for m in "${MATCHES[@]}"; do case "$m" in *00001-of-*) MODEL="$m"; break ;; esac; done
[[ -n "$MODEL" ]] || MODEL=$(printf '%s\n' "${MATCHES[@]}" | xargs -d '\n' du -b 2>/dev/null | sort -rn | head -1 | cut -f2-)
[[ -n "$MODEL" ]] || MODEL="${MATCHES[0]}"

ALIAS="${ALIAS:-$(basename "$(dirname "$MODEL")")}"

# MTP: auto by name, or forced via SPEC=on/off.
SPEC_ARGS=()
case "$SPEC" in
  on) SPEC_ARGS=(--spec-type draft-mtp) ;;
  off) : ;;
  auto) [[ "$MODEL" == *[Mm][Tt][Pp]* ]] && SPEC_ARGS=(--spec-type draft-mtp) ;;
esac
[[ ${#SPEC_ARGS[@]} -gt 0 ]] && echo "[mtp] enabling self-speculative decoding (--spec-type draft-mtp)"

CMD=("$LLAMA_SERVER"
  -m "$MODEL"
  -ngl "$NGL" --no-mmap -fa on
  -c "$CTX" -np "$NP" -b "$BATCH" -ub "$UBATCH" --ctx-checkpoints "$CTXCP" --cache-ram "$CACHE_RAM"
  "${SPEC_ARGS[@]}"
  --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" --min-p "$MIN_P" --repeat-penalty "$REPEAT" --presence-penalty "$PRESENCE"
  --host "$HOST" --port "$PORT" -a "$ALIAS" --jinja
  "${EXTRA[@]}")

echo "[serve] model : $MODEL"
echo "[serve] alias=$ALIAS  api=http://$HOST:$PORT/v1  ctx=$CTX np=$NP ngl=$NGL b=$BATCH ub=$UBATCH ctxcp=$CTXCP"
echo "[serve] binary: $LLAMA_SERVER"
if [[ "$PRINT_ONLY" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
