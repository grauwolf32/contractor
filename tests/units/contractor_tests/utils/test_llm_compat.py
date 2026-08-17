"""Tests for the llama.cpp request-adaptation client.

Covers the forced-tool_choice injection that backs hard summarization
enforcement, alongside the pre-existing ``$ref`` schema sanitizing (they share
one client, so we also check they compose).
"""

import pytest
from google.adk.models.lite_llm import LiteLLMClient

from contractor.utils.llm_compat import (
    SanitizingLiteLLMClient,
    forced_response_format,
    forced_tool_choice,
)


def _reset():
    forced_tool_choice.set(None)
    forced_response_format.set(None)


@pytest.mark.asyncio
async def test_acompletion_injects_response_format_over_preset_none(monkeypatch):
    """response_format must be injected even though ADK pre-sets it to None.

    Regression: a key-presence guard skipped the override because ADK's
    lite_llm.py always puts ``response_format=None`` in completion_args, so the
    forced schema never reached the wire.
    """
    captured: dict = {}

    async def fake_super(self, model, messages, tools, **kwargs):
        captured["kwargs"] = kwargs
        return "resp"

    monkeypatch.setattr(LiteLLMClient, "acompletion", fake_super)
    client = SanitizingLiteLLMClient()
    rf = {"type": "json_schema", "json_schema": {"name": "r", "schema": {}}}

    forced_tool_choice.set("none")
    forced_response_format.set(rf)
    try:
        # ADK passes response_format=None explicitly — must be overridden.
        await client.acompletion(model="m", messages=[], tools=[], response_format=None)
    finally:
        _reset()

    assert captured["kwargs"]["tool_choice"] == "none"
    assert captured["kwargs"]["response_format"] == rf


@pytest.mark.asyncio
async def test_acompletion_injects_forced_tool_choice(monkeypatch):
    """When the ContextVar is set, tool_choice is injected; sanitizing still runs."""
    captured: dict = {}

    async def fake_super(self, model, messages, tools, **kwargs):
        captured["kwargs"] = kwargs
        captured["tools"] = tools
        return "resp"

    monkeypatch.setattr(LiteLLMClient, "acompletion", fake_super)
    client = SanitizingLiteLLMClient()
    tools = [{"function": {"parameters": {"properties": {"$ref": {"type": "string"}}}}}]

    forced_tool_choice.set("none")
    try:
        out = await client.acompletion(model="m", messages=[], tools=tools)
    finally:
        _reset()

    assert out == "resp"
    # forced tool_choice flowed into the downstream kwargs
    assert captured["kwargs"]["tool_choice"] == "none"
    # and the $ref property name was still sanitized on the way out
    props = captured["tools"][0]["function"]["parameters"]["properties"]
    assert "ref" in props and "$ref" not in props


@pytest.mark.asyncio
async def test_acompletion_no_injection_when_unset(monkeypatch):
    captured: dict = {}

    async def fake_super(self, model, messages, tools, **kwargs):
        captured["kwargs"] = kwargs
        return "resp"

    monkeypatch.setattr(LiteLLMClient, "acompletion", fake_super)
    client = SanitizingLiteLLMClient()

    _reset()  # explicit clean baseline
    await client.acompletion(model="m", messages=[], tools=[])

    assert "tool_choice" not in captured["kwargs"]


@pytest.mark.asyncio
async def test_completion_injects_forced_tool_choice(monkeypatch):
    """The sync ``completion`` path (used for streaming) injects too."""
    captured: dict = {}

    def fake_super(self, model, messages, tools, stream=False, **kwargs):
        captured["kwargs"] = kwargs
        return "resp"

    monkeypatch.setattr(LiteLLMClient, "completion", fake_super)
    client = SanitizingLiteLLMClient()

    forced_tool_choice.set("none")
    try:
        out = client.completion(model="m", messages=[], tools=[])
    finally:
        _reset()

    assert out == "resp"
    assert captured["kwargs"]["tool_choice"] == "none"
