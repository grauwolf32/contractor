"""Static contract for structured taint annotations over a project workspace."""

from __future__ import annotations

from types import MappingProxyType

from contractor_runtime.toolsets.code_analysis import SHALLOW_PINNED_DEPENDENCIES

TAINT_ANNOTATIONS_REF = "taint-annotations@1"
EXPORTED_TOOLS = frozenset({"annotate_trace", "annotate_validate", "annotate_sink"})
PINNED_DEPENDENCIES = MappingProxyType(dict(SHALLOW_PINNED_DEPENDENCIES))
