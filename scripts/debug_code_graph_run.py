"""Dump every tool_call event from a single code_graph_agent run.

Forensic helper for the mystery where the harness reports 0 insert_line /
edit calls yet files end up modified with annotations.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


async def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / "cli" / ".env", override=False)

    from tests.eval.conftest import select_fixture
    from tests.eval.trace_harness import run_code_graph_agent

    fixture = select_fixture("vulnyapi")
    case = next(c for c in fixture.trace_cases if c["id"] == "notes-search-sqli")
    entry = case["entrypoint"]
    msg = (
        f"Trace the request flow that begins at `{entry['function']}` in "
        f"`{entry['file']}`. {case.get('intent', '')} "
        "Insert `# @trace target=... args=... calls=...` comments above each "
        "function definition you confidently identify as part of the path. "
        "Use the `insert_line` tool to mutate files. Stop once the path is "
        "covered."
    )
    from contractor.utils.settings import DEFAULT_MODEL

    override = os.environ.get("CONTRACTOR_EVAL_MODEL")
    if override:
        from google.adk.models.lite_llm import LiteLlm
        model = LiteLlm(model=override, timeout=600)
    else:
        model = DEFAULT_MODEL

    run = await run_code_graph_agent(
        fixture_root=fixture.source_root,
        user_message=msg,
        model=model,
        namespace="debug-code-graph",
        timeout_s=600.0,
    )

    events = run.agent_run.metrics_events
    print(f"Total metrics events: {len(events)}")
    tool_calls = [e for e in events if str(e.get("event_type")) == "tool_call"]
    tool_results = [e for e in events if str(e.get("event_type")) == "tool_result"]
    tool_excs = [e for e in events if str(e.get("event_type")) == "tool_exception"]
    print(f"  tool_call: {len(tool_calls)}")
    print(f"  tool_result: {len(tool_results)}")
    print(f"  tool_exception: {len(tool_excs)}")
    print()
    print("=== tool_call event sequence ===")
    for i, e in enumerate(tool_calls):
        name = e.get("tool_name")
        args = e.get("arguments", {})
        keys = list(args.keys()) if isinstance(args, dict) else type(args).__name__
        print(f"  {i+1:2d}. {name:20s} args_keys={keys}")
    print()
    print("=== tool_calls (harness reader) ===")
    for i, c in enumerate(run.agent_run.tool_calls):
        keys = list(c.args.keys()) if isinstance(c.args, dict) else type(c.args).__name__
        print(f"  {i+1:2d}. {c.name:20s} args_keys={keys}")
    print()
    print(f"Annotations extracted: {len(run.annotations)}")
    for a in sorted(run.annotations, key=lambda x: (x.file, x.function)):
        print(f"  {a.file} :: {a.function}")
    print(f"\nModified files (overlay _files keys): {sorted(run.modified_files)}")

    # Dump first modified file's content to confirm whether it really
    # contains @trace markers, or if the overlay just cached the original.
    if run.modified_files:
        # Walk overlay -> rerun via the trace harness's extractor.
        # We can't access the overlay here (it's local to run_code_graph_agent).
        # Instead, check whether ANY of the annotated files contains
        # the @trace marker by reading the (fresh) source from disk —
        # if it doesn't, the overlay must have written it.
        for path in sorted(run.modified_files):
            host_path = (fixture.source_root / path.lstrip("/")).resolve()
            if not host_path.exists():
                print(f"  {path}: host file MISSING (cannot compare)")
                continue
            content = host_path.read_text(errors="ignore")
            host_has = "@trace" in content
            print(f"  {path}: host has @trace = {host_has}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
