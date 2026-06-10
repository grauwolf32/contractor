#!/usr/bin/env python3
"""Dump a Langfuse trace into a compact, agent-grouped JSON for LLM analysis.

Groups spans by agent invocation and emits, per invocation:
    - agent_name
    - system_prompt (extracted from the first LLM call's system message)
    - tools (catalog sent to the model — from `llm.tools.*` OpenInference attrs)
    - events: chronologically ordered LLM + tool spans
        * llm: model, usage, optional input/output, tool_calls requested
        * tool: name, arguments, result

Usage:
    poetry run python scripts/dump_langfuse_trace.py \\
        --trace-id <id> \\
        [--project <label>] [--agent <name>] \\
        [--format yaml|json] [--output trace.yaml] \\
        [--no-llm-content] [--max-tokens N] [--prompts-only]

Credentials are read from env (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
LANGFUSE_HOST). The script also auto-loads `<repo>/cli/.env` or `./.env`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DT_MIN = _dt.datetime.min.replace(tzinfo=_dt.UTC)


# ── Metadata helpers ────────────────────────────────────────────────────────


def _flatten(md: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(md, dict):
        return out
    for k, v in md.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _md(obs) -> dict[str, Any]:
    return _flatten(getattr(obs, "metadata", None) or {})


def _kind(obs) -> str:
    md = _md(obs)
    for key in (
        "openinference.span.kind",
        "attributes.openinference.span.kind",
        "scope.openinference.span.kind",
    ):
        if key in md and md[key]:
            return str(md[key]).upper()
    if getattr(obs, "type", None) == "GENERATION":
        return "LLM"
    name = (getattr(obs, "name", "") or "").lower()
    if "agent" in name:
        return "AGENT"
    if "tool" in name:
        return "TOOL"
    return "OTHER"


def _agent_name(obs) -> str | None:
    md = _md(obs)
    for key in ("agent.name", "openinference.agent.name", "attributes.agent.name"):
        if key in md and md[key]:
            return str(md[key])
    if _kind(obs) == "AGENT":
        return obs.name
    return None


def _safe_json(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


# Noise keys produced by ADK/Gemini that aren't useful for trace analysis.
# - usage_metadata: token counters duplicated outside the trace
# - tools / tool_choice: full tool catalog re-sent on every LLM call; we already
#   surface this once at the invocation level (inv.tools)
# - system_instruction: same — system prompt is hoisted to inv.system_prompt
_STRIP_KEYS = frozenset(
    {
        "usage_metadata",
        "usageMetadata",
        "tools",
        "tool_choice",
        "toolChoice",
        "system_instruction",
        "systemInstruction",
    }
)


def _strip_noise(obj: Any) -> Any:
    """Recursively drop bookkeeping keys (e.g. usage_metadata) from nested data."""
    if isinstance(obj, dict):
        return {k: _strip_noise(v) for k, v in obj.items() if k not in _STRIP_KEYS}
    if isinstance(obj, list):
        return [_strip_noise(x) for x in obj]
    return obj


def _start_key(obs):
    return obs.start_time or _DT_MIN


# ── Field extractors ────────────────────────────────────────────────────────


def _tools_catalog(obs) -> list[dict[str, Any]]:
    """Tools advertised to the LLM in this call (OpenInference: llm.tools.{i}.tool.json_schema)."""
    md = _md(obs)
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key, val in md.items():
        if not (key.startswith("llm.tools.") and key.endswith(".tool.json_schema")):
            continue
        schema = _safe_json(val)
        if not isinstance(schema, dict):
            continue
        fn = schema.get("function") if "function" in schema else schema
        name = fn.get("name") if isinstance(fn, dict) else None
        if not name or name in seen:
            continue
        seen.add(name)
        tools.append(
            {
                "name": name,
                "description": (fn.get("description") or "")[:400] if isinstance(fn, dict) else "",
            }
        )
    return tools


def _system_prompt(obs) -> str | None:
    if getattr(obs, "type", None) != "GENERATION":
        return None
    inp = getattr(obs, "input", None)
    # Gemini-style: top-level system_instruction outside the messages array.
    if isinstance(inp, dict):
        for k in ("system_instruction", "systemInstruction"):
            v = inp.get(k)
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                parts = v.get("parts") or []
                if isinstance(parts, list):
                    joined = "\n".join(
                        p.get("text", "") for p in parts if isinstance(p, dict)
                    )
                    if joined:
                        return joined
    msgs = inp if isinstance(inp, list) else (
        inp.get("messages") if isinstance(inp, dict) else None
    )
    if not msgs:
        return None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "system":
            content = m.get("content")
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            return content if isinstance(content, str) else None
    return None


def _tool_calls_from_output(out: Any) -> list[dict[str, Any]]:
    if not isinstance(out, dict):
        return []
    tcs = out.get("tool_calls") or []
    result = []
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        result.append(
            {
                "id": tc.get("id"),
                "name": fn.get("name") or tc.get("name"),
                "arguments": _safe_json(fn.get("arguments") or tc.get("arguments")),
            }
        )
    return result


def _llm_event(obs, *, include_content: bool) -> dict[str, Any]:
    ev: dict[str, Any] = {
        "type": "llm",
        "span_id": obs.id,
        "model": getattr(obs, "model", None),
        "start": obs.start_time.isoformat() if obs.start_time else None,
    }
    tool_calls = _tool_calls_from_output(getattr(obs, "output", None))
    if tool_calls:
        ev["tool_calls"] = tool_calls
    if include_content:
        inp = getattr(obs, "input", None)
        # Drop the system message — it's already at the invocation level.
        if isinstance(inp, list):
            ev["input"] = [m for m in inp if not (isinstance(m, dict) and m.get("role") == "system")]
        elif isinstance(inp, dict) and isinstance(inp.get("messages"), list):
            ev["input"] = [
                m for m in inp["messages"]
                if not (isinstance(m, dict) and m.get("role") == "system")
            ]
        else:
            ev["input"] = inp
        ev["output"] = getattr(obs, "output", None)
        ev["input"] = _strip_noise(ev["input"])
        ev["output"] = _strip_noise(ev["output"])
    return ev


def _tool_event(obs, *, include_content: bool) -> dict[str, Any]:
    md = _md(obs)
    name = md.get("tool.name") or obs.name
    ev: dict[str, Any] = {
        "type": "tool",
        "span_id": obs.id,
        "name": name,
        "start": obs.start_time.isoformat() if obs.start_time else None,
    }
    if (status := getattr(obs, "status_message", None)):
        ev["status"] = status
    if include_content:
        ev["arguments"] = _strip_noise(_safe_json(getattr(obs, "input", None)))
        ev["result"] = _strip_noise(_safe_json(getattr(obs, "output", None)))
    return ev


# ── Assembly ────────────────────────────────────────────────────────────────


@dataclass
class Invocation:
    agent_name: str
    span_id: str
    start: str | None
    end: str | None
    system_prompt: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


def assemble(observations, *, include_content: bool) -> list[Invocation]:
    """One Invocation per AGENT span; LLM/TOOL spans attach to nearest AGENT ancestor."""
    by_id = {o.id: o for o in observations}

    def nearest_agent(obs) -> str | None:
        cur = by_id.get(obs.parent_observation_id) if obs.parent_observation_id else None
        while cur is not None:
            if _kind(cur) == "AGENT":
                return cur.id
            cur = by_id.get(cur.parent_observation_id) if cur.parent_observation_id else None
        return None

    inv_by_span: dict[str, Invocation] = {}
    invocations: list[Invocation] = []

    for a in sorted([o for o in observations if _kind(o) == "AGENT"], key=_start_key):
        inv = Invocation(
            agent_name=_agent_name(a) or a.name or "unknown",
            span_id=a.id,
            start=a.start_time.isoformat() if a.start_time else None,
            end=a.end_time.isoformat() if a.end_time else None,
        )
        inv_by_span[a.id] = inv
        invocations.append(inv)

    # Synthetic root if no explicit AGENT spans surfaced.
    if not invocations:
        synth = Invocation(agent_name="<unknown>", span_id="__synth__", start=None, end=None)
        inv_by_span["__synth__"] = synth
        invocations.append(synth)

    for o in sorted(observations, key=_start_key):
        kind = _kind(o)
        if kind == "AGENT":
            continue
        anc = nearest_agent(o)
        if anc is None:
            anc = next(iter(inv_by_span))
        inv = inv_by_span.get(anc)
        if inv is None:
            continue
        if kind == "LLM":
            if not inv.system_prompt and (sp := _system_prompt(o)):
                inv.system_prompt = sp
            if not inv.tools and (tls := _tools_catalog(o)):
                inv.tools = tls
            inv.events.append(_llm_event(o, include_content=include_content))
        elif kind == "TOOL":
            inv.events.append(_tool_event(o, include_content=include_content))

    return invocations


# ── Output shaping ──────────────────────────────────────────────────────────


def _event_token_cost(ev: dict[str, Any]) -> int:
    """Rough token estimate for one event = chars of its JSON repr / 4."""
    s = json.dumps(ev, default=str, ensure_ascii=False)
    return max(1, len(s) // 4)


def apply_token_budget(invocations: list[Invocation], max_tokens: int) -> None:
    """Drop trailing events once the running token estimate exceeds budget.

    Walks events in order; the first event that would push us over budget is
    replaced (along with any later events) by an `{type: elided}` marker so
    the consumer knows truncation happened.
    """
    remaining = max_tokens
    stopped = False
    for inv in invocations:
        if stopped:
            inv.events = [{"type": "elided", "reason": "token_budget", "skipped": len(inv.events)}]
            continue
        kept: list[dict[str, Any]] = []
        for i, ev in enumerate(inv.events):
            cost = _event_token_cost(ev)
            if cost > remaining:
                kept.append(
                    {
                        "type": "elided",
                        "reason": "token_budget",
                        "skipped": len(inv.events) - i,
                    }
                )
                stopped = True
                break
            kept.append(ev)
            remaining -= cost
        inv.events = kept


# ── CLI ─────────────────────────────────────────────────────────────────────


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # `cli/.env` is the canonical config file (resolved from the repo root so
    # the script works from any CWD); plain `.env` in the CWD is a fallback.
    repo_root = Path(__file__).resolve().parents[1]
    for cand in (repo_root / "cli" / ".env", Path(".env")):
        if cand.exists():
            load_dotenv(cand)
            return


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace-id", required=True)
    p.add_argument("--project", default=None, help="Project label (informational; creds determine actual project).")
    p.add_argument("--agent", default=None, help="Keep only invocations of this agent name.")
    p.add_argument("--output", "-o", default="-", help="Output path or '-' for stdout.")
    p.add_argument("--no-llm-content", action="store_true", help="Strip LLM input/output bodies and tool args/results.")
    p.add_argument("--max-tokens", type=int, default=None, help="Soft cap on total content tokens; later events become 'elided'.")
    p.add_argument("--prompts-only", action="store_true", help="Output only agent_name + system_prompt per invocation.")
    p.add_argument("--format", choices=("yaml", "json"), default="yaml", help="Output format (default: yaml).")
    p.add_argument("--indent", type=int, default=2, help="JSON indent (ignored for yaml).")
    p.add_argument("--list-agents", action="store_true", help="Just list distinct agent names from the trace.")
    p.add_argument("--timeout", type=int, default=120, help="HTTP timeout (seconds) for the Langfuse API call.")
    args = p.parse_args()

    _load_env()
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        print("error: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set", file=sys.stderr)
        return 2

    from langfuse import Langfuse

    client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
        timeout=args.timeout,
    )

    trace = client.api.trace.get(args.trace_id, request_options={"timeout_in_seconds": args.timeout})
    observations = getattr(trace, "observations", None) or []

    invocations = assemble(observations, include_content=not args.no_llm_content)

    if args.list_agents:
        names = sorted({i.agent_name for i in invocations})
        for n in names:
            print(n)
        return 0

    if args.agent:
        invocations = [i for i in invocations if i.agent_name == args.agent]

    if args.prompts_only:
        payload = {
            "trace_id": args.trace_id,
            "project": args.project,
            "trace_name": getattr(trace, "name", None),
            "agent_count": len(invocations),
            "agents": [
                {"agent_name": inv.agent_name, "system_prompt": inv.system_prompt}
                for inv in invocations
            ],
        }
    else:
        if args.max_tokens is not None:
            apply_token_budget(invocations, args.max_tokens)
        payload = {
            "trace_id": args.trace_id,
            "project": args.project,
            "trace_name": getattr(trace, "name", None),
            "agent_count": len(invocations),
            "invocations": [inv.__dict__ for inv in invocations],
        }

    if args.format == "yaml":
        import yaml

        # Force literal block style ("|") for any string containing newlines,
        # otherwise PyYAML falls back to single-quoted folded scalars with
        # ugly `\n\` line-continuations. Trailing whitespace must be stripped
        # per-line — yaml refuses literal style otherwise.
        def _str_presenter(dumper, data):
            if "\n" in data:
                cleaned = "\n".join(line.rstrip() for line in data.split("\n"))
                return dumper.represent_scalar("tag:yaml.org,2002:str", cleaned, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        yaml.add_representer(str, _str_presenter, Dumper=yaml.SafeDumper)

        plain = json.loads(json.dumps(payload, default=str, ensure_ascii=False))
        text = yaml.safe_dump(
            plain,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=10**9,
        )
    else:
        text = json.dumps(payload, indent=args.indent, default=str, ensure_ascii=False) + "\n"

    if args.output == "-":
        sys.stdout.write(text)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
