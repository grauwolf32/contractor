#!/usr/bin/env python3
"""Reproduction harness for the summarization-limit mechanism on llama.cpp.

Runs the REAL planner+worker path (via tests.eval.task_harness.run_task_pipeline)
with a low max_tokens so the summarization limit fires mid-run, against a bounded
real target, pointed at a llama.cpp-served model through the LiteLLM proxy.

Surfaces: per-LLM-call token progression, the summarization callback's
INFO/WARNING logs, whether tool_choice was injected, and the final task status —
so we can see whether summarization actually terminates the worker.

Env:
  SUMM_MAX_TOKENS  summarization trigger (default 8000)
  SUMM_MODEL       litellm alias (default llamacpp-qwen3.6-35b-a3b -> :8083)
  SUMM_TARGET      target dir to root the fs at (default contractor/runners)
  SUMM_TASK        task template name (default project_information)
  SUMM_TIMEOUT     overall timeout seconds (default 600)
  (force value comes from Settings.summarization_force_tool_choice / env)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from functools import partial

REPO = "/home/ruslan/src/contractor"
sys.path.insert(0, REPO)

# Only our two loggers at INFO; keep the rest quiet.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("contractor.callbacks.context").setLevel(logging.INFO)
logging.getLogger("contractor.utils.llm_compat").setLevel(logging.INFO)

from cli.fs import RootedLocalFileSystem  # noqa: E402
from contractor.agents.swe_agent.agent import build_swe_agent  # noqa: E402
from contractor.runners.agio import AgioEventType  # noqa: E402
from contractor.utils.settings import build_model, get_settings  # noqa: E402
from tests.eval.task_harness import run_task_pipeline  # noqa: E402

MAX_TOKENS = int(os.environ.get("SUMM_MAX_TOKENS", "8000"))
MODEL = os.environ.get("SUMM_MODEL", "llamacpp-qwen3.6-35b-a3b")
TARGET = os.environ.get("SUMM_TARGET", f"{REPO}/contractor/runners")
TASK = os.environ.get("SUMM_TASK", "project_information")
TIMEOUT = float(os.environ.get("SUMM_TIMEOUT", "600"))


async def routing_check(model) -> bool:
    """One tiny call to confirm the alias reaches llama.cpp through the proxy."""
    from google.adk.models import LlmRequest
    from google.genai import types

    req = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="say OK")])],
        config=types.GenerateContentConfig(),
    )
    try:
        async for resp in model.generate_content_async(req):
            txt = (resp.content.parts[0].text if resp.content and resp.content.parts else "")
            print(f"[routing] reached model, reply~={txt[:40]!r}")
            return True
    except Exception as e:  # noqa: BLE001
        print(f"[routing] FAILED: {type(e).__name__}: {e}")
        return False
    return False


def queue_fn(runner):
    fs = RootedLocalFileSystem(TARGET)
    model = build_model(MODEL, timeout=300)
    worker_builder = partial(
        build_swe_agent, name="swe_agent", _format="json", fs=fs,
        model=model, max_tokens=MAX_TOKENS,
    )
    runner.add_variable(name="project_path", value="/")
    runner.add_task(
        name=TASK,
        ref=TASK,
        worker_builder=worker_builder,
        iterations=1,
        max_attempts=1,
        max_steps=20,
        namespace="summ_repro",
        model=model,
    )


async def main():
    s = get_settings()
    print("=" * 72)
    print(f"model={MODEL}  max_tokens={MAX_TOKENS}  task={TASK}")
    print(f"target={TARGET}")
    print(f"force_tool_choice setting = {s.summarization_force_tool_choice!r}")
    print("=" * 72)

    if not await routing_check(build_model(MODEL, timeout=60)):
        print("ABORT: routing to llama.cpp failed; is :8083 up + proxy on :4000?")
        return

    try:
        run = await run_task_pipeline(
            queue_fn=queue_fn,
            artifact_keys=[f"{TASK}/result"],
            namespace="summ_repro",
            timeout_s=TIMEOUT,
            runner_name="summ-repro",
        )
    except Exception as e:  # noqa: BLE001
        print(f"\n>>> run_task_pipeline raised: {type(e).__name__}: {e}")
        return

    # Token progression across LLM calls.
    print("\n--- LLM call token progression (input/output/total) ---")
    running = 0
    for ev in run.events:
        if str(ev.type) == AgioEventType.LLM_USAGE:
            u = ev.payload.get("usage") or {}
            running += int(u.get("total", 0) or 0)
            print(f"  call: in={u.get('input')} out={u.get('output')} "
                  f"total={u.get('total')}  cumulative~={running}")

    # Summarization callback final state (from CALLBACK_SUMMARY events).
    print("\n--- SummarizationLimitCallback state ---")
    for ev in run.events:
        if str(ev.type) == AgioEventType.CALLBACK_SUMMARY:
            cbs = ev.payload.get("callbacks") or ev.payload
            print(f"  {str(cbs)[:400]}")
            break

    print("\n--- metrics ---")
    from tests.eval.task_harness import render_metrics_table
    print(render_metrics_table(run.metrics))

    print("\n--- results ---")
    for r in run.results:
        rd = dict(r) if isinstance(r, dict) else r
        print(f"  status={getattr(rd, 'status', rd)!r}")
    res_text = run.result_text(TASK)
    print(f"\n--- {TASK}/result ({len(res_text)} chars) ---")
    print(res_text[:800] if res_text else "(empty / not published)")


if __name__ == "__main__":
    asyncio.run(main())
