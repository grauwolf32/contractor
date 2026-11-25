"""Release google-adk's process-wide caches keyed on Worker tool callables (RTA-R116).

The Worker wraps every allocation-bound tool instance in ``FunctionTool``. In the
pinned google-adk 2.15.0 three process-wide LRU caches retain callable keys:

- ``google.adk.utils.context_utils.find_context_parameter``, called by
  ``FunctionTool.__init__``;
- ``google.adk.tools.function_tool._build_declaration_cached``, called by
  ``FunctionTool._get_declaration`` on every model request;
- ``google.adk.utils._callable_utils._get_type_hints_for_unwrapped``, which
  retains the underlying function, including an allocation captured by a closure.

Without a release step each released tool, and through it its ArtifactClient,
Toolset session, Worker State and MetricsState, would stay reachable until 1024
newer callables evicted it. These caches memoize pure introspection.
Clearing may rebuild a live neighbor's
declaration, without changing its callable or invocation state. The released
allocation joins all users before clearing, so none can repopulate its keys.

All names are resolved at import, so an ADK upgrade that renames or removes
one fails Runtime startup instead of silently retaining tools again.
"""

from __future__ import annotations

from google.adk.tools import function_tool as _adk_function_tool
from google.adk.utils import _callable_utils as _adk_callable_utils
from google.adk.utils import context_utils as _adk_context_utils

_TOOL_CALLABLE_CACHES = (
    _adk_context_utils.find_context_parameter,
    _adk_function_tool._build_declaration_cached,
    _adk_callable_utils._get_type_hints_for_unwrapped,
)


def release_adk_tool_caches() -> None:
    """Drop every ADK cache entry that keeps a Worker tool callable alive."""

    for cache in _TOOL_CALLABLE_CACHES:
        cache.cache_clear()
