#!/usr/bin/env python3
"""No-network e2e proof of the summarization-limit CAP.

Drives a REAL contractor swe_agent worker (full callback stack: TokenUsage ->
SummarizationLimit -> elide -> guardrails) through a REAL google.adk Runner,
with the REAL SanitizingLiteLLMClient (so _apply_forced_params runs), but the
underlying litellm network call is replaced by a FakeLLM that:

  - each turn requests a tool call (read_file) and reports GROWING token usage,
  - so TokenUsageCallback pushes the worker over max_tokens after ~2 calls,
  - and CAPTURES the tool_choice kwarg actually present on each request
    (i.e. after _apply_forced_params has injected the ContextVar value).

We run twice:
  A) force_tool_choice="none" (production default)  -> expect cap
  B) force_tool_choice=None    (disabled)           -> expect runaway

Pass FORCE=none|off as argv[1].
"""
from __future__ import annotations

import asyncio
import logging
import sys

REPO = "/home/ruslan/src/contractor"
sys.path.insert(0, REPO)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("contractor.callbacks.context").setLevel(logging.INFO)
logging.getLogger("contractor.utils.llm_compat").setLevel(logging.INFO)

import google.adk.models.lite_llm as litellm_mod  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from contractor.agents.swe_agent.agent import build_swe_agent  # noqa: E402
from contractor.utils.llm_compat import SanitizingLiteLLMClient  # noqa: E402

MAX_TOKENS = 1500
HARD_CALL_CAP = 25  # safety net so a true runaway test still terminates

# --- shared observation state ---
OBSERVED: list[dict] = []


def make_fake_acompletion():
    """Return an async fake of litellm.acompletion.

    Captures kwargs['tool_choice'] (set by the real _apply_forced_params), and:
      - if tool_choice == 'none': returns a TEXT-ONLY final result (no tool call)
      - else: returns a read_file tool call + growing usage
    """
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        Choices,
        Function,
        Message,
        ModelResponse,
        Usage,
    )

    state = {"n": 0}

    async def fake(model, messages, tools, **kwargs):  # noqa: ANN001
        state["n"] += 1
        n = state["n"]
        tc = kwargs.get("tool_choice")
        rf = kwargs.get("response_format")
        OBSERVED.append({"call": n, "tool_choice": tc, "has_response_format": bool(rf)})
        print(f"[fake-llm] call#{n} tool_choice={tc!r} response_format={'yes' if rf else 'no'}")

        # Growing usage: each call ~900 tokens, so cumulative crosses 1500 at call 2.
        usage = Usage(prompt_tokens=800 + n * 50, completion_tokens=100, total_tokens=900 + n * 50)

        if tc == "none":
            # tool calls forbidden -> deliver final text result, no function_call
            msg = Message(
                role="assistant",
                content='{"task_id": "", "status": "done", "result": "final summary"}',
                tool_calls=None,
            )
            choice = Choices(finish_reason="stop", index=0, message=msg)
        else:
            # request another tool call (would loop forever if never capped)
            tcall = ChatCompletionMessageToolCall(
                id=f"call_{n}",
                type="function",
                function=Function(name="read_file", arguments='{"path": "/x"}'),
            )
            msg = Message(role="assistant", content=None, tool_calls=[tcall])
            choice = Choices(finish_reason="tool_calls", index=0, message=msg)

        if n > HARD_CALL_CAP:
            raise RuntimeError(f"HARD_CALL_CAP exceeded ({n}) -> RUNAWAY (not capped)")

        return ModelResponse(choices=[choice], model=model, usage=usage)

    return fake


async def run_once(force: str | None) -> dict:
    OBSERVED.clear()
    # Patch settings so build_worker picks up the desired force value.
    from contractor.utils import settings as settings_mod
    settings_mod.get_settings.cache_clear()
    s = settings_mod.get_settings()
    object.__setattr__(s, "summarization_force_tool_choice", force)

    # Real SanitizingLiteLLMClient -> exercises _apply_forced_params.
    model = LiteLlm(model="openai/fake", llm_client=SanitizingLiteLLMClient())

    # Force litellm's lazy import to populate module globals BEFORE we patch,
    # so _ensure_litellm_imported() (called inside acompletion) is a no-op that
    # won't overwrite our stub.
    litellm_mod._ensure_litellm_imported()

    # Stub the network: replace module-level acompletion used by LiteLLMClient.
    fake = make_fake_acompletion()
    orig = litellm_mod.acompletion
    litellm_mod.acompletion = fake

    # Build a real worker. Use a throwaway in-memory fs.
    from fsspec.implementations.memory import MemoryFileSystem
    fs = MemoryFileSystem()

    agent = build_swe_agent(
        name="swe_agent", fs=fs, namespace="summ_e2e",
        _format="json", max_tokens=MAX_TOKENS, model=model,
    )

    runner = InMemoryRunner(agent=agent, app_name="summ_e2e")
    session = await runner.session_service.create_session(
        app_name="summ_e2e", user_id="u1",
    )

    n_events = 0
    err = None
    try:
        async for _ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="do the subtask")]),
        ):
            n_events += 1
            if n_events > 200:
                err = "event-cap"
                break
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    finally:
        litellm_mod.acompletion = orig

    return {
        "force": force,
        "llm_calls": len(OBSERVED),
        "observed": OBSERVED.copy(),
        "error": err,
    }


async def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "none"
    force = None if which in ("off", "none-disabled", "disabled") else "none"
    if which == "off":
        force = None
    print("=" * 72)
    print(f"RUN force_tool_choice={force!r}  max_tokens={MAX_TOKENS}  HARD_CALL_CAP={HARD_CALL_CAP}")
    print("=" * 72)
    res = await run_once(force)
    print("\n--- RESULT ---")
    print(f"force={res['force']!r}  llm_calls={res['llm_calls']}  error={res['error']!r}")
    for o in res["observed"]:
        print(f"  {o}")
    capped = res["llm_calls"] <= HARD_CALL_CAP and res["error"] not in (
        "event-cap",
    ) and not (res["error"] and "RUNAWAY" in res["error"])
    saw_none = any(o["tool_choice"] == "none" for o in res["observed"])
    print(f"\nsaw tool_choice='none' injected: {saw_none}")
    print(f"terminated within cap: {capped}")


if __name__ == "__main__":
    asyncio.run(main())
