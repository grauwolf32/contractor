"""Drive the REAL SummarizationLimitCallback directly (no LLM).

Verifies the four enforcement scenarios with explicit PASS/FAIL prints.
"""

import sys

from contractor.callbacks.context import SummarizationLimitCallback
from contractor.utils.llm_compat import forced_response_format, forced_tool_choice
from tests.units.contractor_tests.helpers import mk_callback_context, mk_llm_request

TOKEN_STATE_KEY = "::TokenUsageCallback"

_failures = []


def _seed_token_state(ctx, total: int) -> None:
    """Mirror the layout TokenUsageCallback writes via save_to_state."""
    ctx.state.setdefault("callbacks", {})
    ctx.state["callbacks"][TOKEN_STATE_KEY] = {
        "counter": {"input": 0, "output": 0, "total": total},
        "invocation_id": ctx.invocation_id,
    }


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    if not cond:
        _failures.append(label)
    print(f"  [{status}] {label}")


RESP_FMT = {
    "type": "json_schema",
    "json_schema": {"name": "SubtaskExecutionResult", "schema": {"type": "object"}},
}


def reset_ctxvars():
    forced_tool_choice.set(None)
    forced_response_format.set(None)


def scenario_1_under_limit():
    print("\n=== Scenario 1: under limit -> no message, ctxvar cleared (None), last_forced None ===")
    reset_ctxvars()
    # Pre-seed a stale 'none' to prove the under-limit path actively clears it.
    forced_tool_choice.set("none")
    forced_response_format.set(RESP_FMT)

    ctx = mk_callback_context()
    _seed_token_state(ctx, total=500)  # < 1000

    cb = SummarizationLimitCallback(
        message="please summarize",
        max_tokens=1000,
        force_tool_choice="none",
        force_response_format=RESP_FMT,
    )
    req = mk_llm_request()
    cb(ctx, req)

    print(f"  token_count={cb.token_count} max_tokens={cb.max_tokens}")
    print(f"  request.contents={req.contents}")
    print(f"  forced_tool_choice.get()={forced_tool_choice.get()!r}")
    print(f"  forced_response_format.get()={forced_response_format.get()!r}")
    print(f"  last_forced={cb.last_forced!r}")

    check("no summarize message injected", req.contents == [])
    check("forced_tool_choice published as None (cleared)", forced_tool_choice.get() is None)
    check("forced_response_format cleared to None", forced_response_format.get() is None)
    check("last_forced is None", cb.last_forced is None)


def scenario_2_over_limit():
    print("\n=== Scenario 2: at/over limit -> message injected once, none forced, rf set, last_forced none ===")
    reset_ctxvars()
    ctx = mk_callback_context()
    _seed_token_state(ctx, total=1000)  # == max_tokens (>= triggers)

    cb = SummarizationLimitCallback(
        message="summarize now",
        max_tokens=1000,
        force_tool_choice="none",
        force_response_format=RESP_FMT,
    )
    req = mk_llm_request()
    cb(ctx, req)

    print(f"  token_count={cb.token_count} max_tokens={cb.max_tokens}")
    print(f"  len(request.contents)={len(req.contents)}")
    injected_text = None
    injected_role = None
    if req.contents:
        c = req.contents[0]
        injected_role = c.role
        injected_text = c.parts[0].text
    print(f"  injected role={injected_role!r} text={injected_text!r}")
    print(f"  forced_tool_choice.get()={forced_tool_choice.get()!r}")
    print(f"  forced_response_format.get()={forced_response_format.get()!r}")
    print(f"  last_forced={cb.last_forced!r}")
    print(f"  history(len)={len(cb.history)}")

    check("exactly one content injected", len(req.contents) == 1)
    check("injected role == user", injected_role == "user")
    check("injected text == message", injected_text == "summarize now")
    check("forced_tool_choice.get() == 'none'", forced_tool_choice.get() == "none")
    check("forced_response_format set (paired)", forced_response_format.get() == RESP_FMT)
    check("last_forced == 'none'", cb.last_forced == "none")
    check("history recorded one firing", len(cb.history) == 1)
    return cb, ctx


def scenario_3_same_invocation_latch(cb, ctx):
    print("\n=== Scenario 3: over-limit again SAME invocation_id -> NOT injected again (latch holds) ===")
    # counter is still over-limit (same seeded state, same ctx)
    req2 = mk_llm_request()
    cb(ctx, req2)

    print(f"  fired={cb.fired} fired_invocation_id==ctx.invocation_id -> {cb.fired_invocation_id == ctx.invocation_id}")
    print(f"  len(req2.contents)={len(req2.contents)}")
    print(f"  history(len)={len(cb.history)}")
    print(f"  forced_tool_choice.get()={forced_tool_choice.get()!r} (still refreshed)")

    check("second over-limit call does NOT inject message", req2.contents == [])
    check("history still length 1 (no second firing)", len(cb.history) == 1)
    check("enforcement still refreshed (none)", forced_tool_choice.get() == "none")

    # third call, still same invocation
    req3 = mk_llm_request()
    cb(ctx, req3)
    check("third over-limit call also not injected", req3.contents == [])
    check("history still length 1 after third call", len(cb.history) == 1)


def scenario_4_rearm_new_invocation(cb):
    print("\n=== Scenario 4: NEW invocation_id -> latch re-arms (message injected again) ===")
    ctx2 = mk_callback_context()  # fresh invocation_id
    _seed_token_state(ctx2, total=2000)  # over limit
    req = mk_llm_request()
    cb(ctx2, req)

    print(f"  new invocation_id={ctx2.invocation_id}")
    print(f"  fired_invocation_id={cb.fired_invocation_id}")
    print(f"  len(request.contents)={len(req.contents)}")
    print(f"  history(len)={len(cb.history)}")

    injected_text = req.contents[0].parts[0].text if req.contents else None
    check("message injected again on new invocation", len(req.contents) == 1)
    check("injected text == message", injected_text == "summarize now")
    check("history grew to 2 (re-armed firing)", len(cb.history) == 2)
    check("fired_invocation_id tracks new invocation", cb.fired_invocation_id == ctx2.invocation_id)


def main():
    scenario_1_under_limit()
    cb, ctx = scenario_2_over_limit()
    scenario_3_same_invocation_latch(cb, ctx)
    scenario_4_rearm_new_invocation(cb)

    print("\n=================== SUMMARY ===================")
    if _failures:
        print(f"OVERALL: FAIL ({len(_failures)} checks failed):")
        for f in _failures:
            print(f"   - {f}")
        sys.exit(1)
    else:
        print("OVERALL: PASS (all checks passed)")
        sys.exit(0)


if __name__ == "__main__":
    main()
