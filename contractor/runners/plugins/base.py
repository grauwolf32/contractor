# contractor/runners/_base_adk_plugin.py
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from google.adk.plugins.base_plugin import BasePlugin


def snapshot_state(state_obj: Any) -> dict[str, Any]:
    """Best-effort conversion of an ADK state object to a plain dict."""
    if state_obj is None:
        return {}
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(state_obj, method_name, None)
        if callable(method):
            try:
                value = method()
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
    if isinstance(state_obj, dict):
        return dict(state_obj)
    return {}


def resolve_tool_args(
    tool_args: dict[str, Any] | None,
    args: dict[str, Any] | None,
) -> dict[str, Any]:
    """ADK passes tool arguments under varying kwarg names — normalise them."""
    return tool_args if tool_args is not None else (args or {})


def resolve_tool_response(
    tool_response: dict[str, Any] | None,
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """ADK passes tool results under varying kwarg names — normalise them."""
    return tool_response if tool_response is not None else result


_REDACTED: str = "***REDACTED***"

# Exact (lowercased) key names whose values are credentials/secrets.
_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "cookies",
        "auth",
        "password",
        "passwd",
        "passphrase",
        "secret",
        "private_key",
        "private-key",
        "credential",
        "client_secret",
        "client-secret",
        "token",
        "access_token",
        "access-token",
        "refresh_token",
        "refresh-token",
        "id_token",
        "id-token",
        "session_token",
        "api_key",
        "api-key",
        "apikey",
        "x-api-key",
        "credentials",
        "bearer",
    }
)

# Substrings that unambiguously mark a secret (chosen to NOT match benign
# token-count fields like ``token_count`` / ``max_tokens`` / ``prompt_tokens``).
_SENSITIVE_SUBSTRINGS: tuple[str, ...] = (
    "password",
    "secret",
    "api_key",
    "apikey",
    "api-key",
)


def _is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    k = key.strip().lower()
    if k in _SENSITIVE_KEYS:
        return True
    if any(sub in k for sub in _SENSITIVE_SUBSTRINGS):
        return True
    # access_token / refresh_token / auth-token — but NOT token_count / *_tokens.
    return k.endswith("_token") or k.endswith("-token")


def redact_sensitive(obj: Any, _depth: int = 0) -> Any:
    """Return a copy of *obj* with the values of secret-bearing keys masked.

    Walks nested dicts/lists so HTTP ``headers`` (Authorization, Set-Cookie),
    ``auth`` bundles, cookie jars, and bearer/API tokens are masked wherever they
    appear in tool arguments or results before they are persisted to
    ``metrics.jsonl``. The input is never mutated; scalar leaves (incl. large
    strings) are returned as-is.

    Note: this does NOT cover the Langfuse span path — when ``USE_LANGFUSE`` is
    set, tool args and LLM messages are exported by OpenInference OTel
    auto-instrumentation (``contractor/utils/observability.py``), which bypasses
    the plugin/sink pipeline entirely and is not redacted here.
    """
    if _depth > 12:
        return obj
    if isinstance(obj, dict):
        return {
            k: (_REDACTED if _is_sensitive_key(k) else redact_sensitive(v, _depth + 1))
            for k, v in obj.items()
        }
    if isinstance(obj, list | tuple):
        return [redact_sensitive(v, _depth + 1) for v in obj]
    return obj


class PluginContext:
    """Immutable bundle of identifiers every plugin callback needs."""

    __slots__ = ("task_name", "task_id", "iteration", "session_id")

    def __init__(
        self,
        *,
        task_name: str,
        task_id: int,
        iteration: int,
        session_id: str,
    ) -> None:
        self.task_name = task_name
        self.task_id = task_id
        self.iteration = iteration
        self.session_id = session_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "task_id": self.task_id,
            "iteration": self.iteration,
            "session_id": self.session_id,
        }


class BaseAdkPlugin(BasePlugin):
    """
    Shared base for all TaskRunner ADK plugins.

    Provides:
      - A ``PluginContext`` with the common identification fields.
      - A thin ``_emit`` wrapper that auto-injects those fields.
      - Helpers for extracting invocation / agent identity from ADK contexts.
    """

    def __init__(
        self,
        *,
        plugin_prefix: str,
        ctx: PluginContext,
        emit: Callable[..., Awaitable[None]],
    ) -> None:
        name = f"{plugin_prefix}_{ctx.task_name}_{ctx.task_id}_{ctx.iteration}"
        super().__init__(name=name)
        self._ctx = ctx
        self._raw_emit = emit

    # Payload keys whose (possibly nested) values may carry live-target
    # credentials and must be scrubbed before persistence/telemetry.
    _REDACT_PAYLOAD_KEYS: tuple[str, ...] = ("arguments", "result", "tool_response")

    async def _emit(self, event_type: str, **payload: Any) -> None:
        for key in self._REDACT_PAYLOAD_KEYS:
            value = payload.get(key)
            if value is not None:
                payload[key] = redact_sensitive(value)
        await self._raw_emit(event_type, **self._ctx.as_dict(), **payload)

    @staticmethod
    def _identity(context: Any) -> tuple[str | None, str | None]:
        """Extract (invocation_id, agent_name) from a tool/callback context."""
        return (
            getattr(context, "invocation_id", None),
            getattr(context, "agent_name", None),
        )
