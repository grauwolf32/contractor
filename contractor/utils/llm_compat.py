"""llama.cpp tool-schema compatibility shim.

llama.cpp's tool-call parser path (json-schema-to-grammar + the differential
auto-parser) crashes when a tool *parameter* schema contains a property whose
name is literally ``$ref``: it treats the key as the JSON-Schema ``$ref``
*reference keyword* and expects a string value, throwing
``[json.exception.type_error.302] type must be string, but is object``.

Our ``PathItem`` model (contractor/tools/openapi/models.py) has a ``ref`` field
aliased to ``$ref`` (legal in OpenAPI), which surfaces as exactly such a
property in the ``upsert_path`` tool schema. LM Studio tolerated it (it parses
tool calls free-form, without deriving a grammar); llama.cpp does not.

Fix: rename ``$ref`` property *names* to ``ref`` in the *outbound* tool schemas
only. The response needs no reverse step — the backing pydantic models use
``validate_by_name=True`` together with ``alias="$ref"``, so they accept either
``ref`` or ``$ref`` natively. We never rewrite model output, so there is no
ambiguity about "what to restore".

Only property names are renamed (keys directly under a ``properties`` map, or
entries in a ``required`` list). Genuine JSON-Schema ``$ref`` *references*
(string-valued, e.g. ``"#/$defs/Operation"``) are left untouched.

Harmless on non-llama.cpp backends: the receiving models accept both spellings.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

from google.adk.models.lite_llm import LiteLLMClient

logger = logging.getLogger(__name__)

# Request-scoped tool_choice override for the *next* model call, set by
# ``SummarizationLimitCallback`` (contractor/callbacks/context.py) and read here.
#
# Why a ContextVar and not client/model attributes: ``DEFAULT_MODEL`` (hence its
# ``llm_client``) is a single shared instance, so mutating it to force a
# tool_choice would race across concurrent invocations that fan out over the
# same model. A ContextVar is copied per asyncio task, so each invocation sees
# only the value its own callback set; here we mutate only the per-call
# ``kwargs`` dict, never shared state.
#
# Value is an OpenAI-style tool_choice *string* ("none" | "auto" | "required");
# ``None`` means "no override" (normal behaviour). llama.cpp honours only the
# string forms — the object/named-function form is silently dropped to ``auto``
# there — so we deliberately carry a string. ``"none"`` forbids tool calls,
# which is how we force a worker to emit its final structured result instead of
# continuing to call tools once it crosses the context limit.
forced_tool_choice: ContextVar[str | None] = ContextVar(
    "forced_tool_choice", default=None
)

# Companion to forced_tool_choice: an OpenAI ``response_format`` (JSON-schema)
# to inject on the same forced call. ``tool_choice="none"`` forbids tools, but
# contractor workers carry no output_schema, so without a response_format the
# forced text is unconstrained and frequently fails to parse as the expected
# result. Pairing the two grammar-constrains the forced output to valid JSON.
forced_response_format: ContextVar[dict | None] = ContextVar(
    "forced_response_format", default=None
)


def _apply_forced_params(kwargs: dict[str, Any]) -> None:
    """Inject context-scoped forced request params into ``kwargs`` (in place).

    No-op when unset. When set, they win: the callback only sets them when it
    deliberately wants to force the model's hand (e.g. ``"none"`` + a result
    schema at the context limit).
    """
    tc = forced_tool_choice.get()
    if tc is not None:
        kwargs["tool_choice"] = tc
        rf = forced_response_format.get()
        # ADK pre-populates completion_args with ``response_format=None``
        # (lite_llm.py), so test the *value*, not key presence — otherwise the
        # explicit None would block our override.
        if rf is not None and not kwargs.get("response_format"):
            kwargs["response_format"] = rf
        logger.info(
            "forced tool_choice=%r (response_format=%s) injected into request",
            tc, "yes" if kwargs.get("response_format") else "no",
        )


def sanitize_schema(node: Any) -> None:
    """Make a JSON-Schema safe for llama.cpp's tool-call parser (in place).

    Two independent fixes, both observed to crash the differential auto-parser on
    the ``upsert_path`` tool's ``PathItem`` schema:

    1. Rename ``$ref`` *property names* to ``ref`` — only when ``$ref`` names a
       property (a key inside a ``properties`` map, or an entry in a ``required``
       list). Genuine JSON-Schema ``$ref`` *references* (string-valued) are kept.
    2. Drop ``examples`` — the auto-parser walks nested example *objects* as if
       they were schemas and throws ``type must be string, but is object``.
       ``examples`` is metadata (sample values), so removing it is loss-free for
       validation; the model still has field descriptions.
    """
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict) and "$ref" in props:
            props["ref"] = props.pop("$ref")
        required = node.get("required")
        if isinstance(required, list) and "$ref" in required:
            node["required"] = ["ref" if r == "$ref" else r for r in required]
        node.pop("examples", None)
        for value in node.values():
            sanitize_schema(value)
    elif isinstance(node, list):
        for item in node:
            sanitize_schema(item)


def sanitize_tools(tools: Any) -> Any:
    """Rename ``$ref`` property names in each tool's parameter schema (in place).

    ``tools`` is the OpenAI-format list passed to litellm. Idempotent: once a
    schema is renamed there is no ``$ref`` property left to rename.
    """
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                params = tool.get("function", {}).get("parameters")
                if params is not None:
                    sanitize_schema(params)
    return tools


class SanitizingLiteLLMClient(LiteLLMClient):
    """``LiteLLMClient`` that adapts outbound requests for llama.cpp.

    Two in-place adaptations on every call, both harmless on other backends:

    1. Sanitize ``$ref`` tool-schema property names (see module docstring).
    2. Inject the context-scoped :data:`forced_tool_choice` override, so a
       callback can force ``tool_choice`` (e.g. ``"none"`` at the context
       limit) without mutating this shared client instance.

    Drop-in replacement injected via ``build_model``.
    """

    async def acompletion(self, model, messages, tools, **kwargs):
        _apply_forced_params(kwargs)
        return await super().acompletion(
            model=model, messages=messages, tools=sanitize_tools(tools), **kwargs
        )

    def completion(self, model, messages, tools, stream=False, **kwargs):
        _apply_forced_params(kwargs)
        return super().completion(
            model=model,
            messages=messages,
            tools=sanitize_tools(tools),
            stream=stream,
            **kwargs,
        )
