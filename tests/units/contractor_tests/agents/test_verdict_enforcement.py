from types import SimpleNamespace

from fsspec.implementations.memory import MemoryFileSystem

from contractor.agents.trace_verifier_agent.agent import (
    build_trace_verifier_agent,
)
from tests.units.contractor_tests.helpers import (
    MockContent,
    mk_callback_context,
    mk_text_part,
)


def test_trace_verifier_nudges_text_only_finish_without_persisting_tool_call():
    agent = build_trace_verifier_agent(
        name="trace_verifier",
        fs=MemoryFileSystem(),
        namespace="verify",
        source_namespace="source",
    )
    ctx = mk_callback_context()
    response = SimpleNamespace(
        content=MockContent(role="model", parts=[mk_text_part("done")]),
        usage_metadata=None,
    )

    nudge = agent.after_model_callback(
        callback_context=ctx,
        llm_response=response,
    )

    assert nudge is not None
    assert "report_verification" in nudge.content.parts[0].text

