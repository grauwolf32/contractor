from datetime import datetime, timezone

from .dictutils import DictDiff, deep_merge, dict_diff
from .prompt import (all_active_prompt_versions, load_prompt,
                     load_prompt_with_version)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "dict_diff",
    "deep_merge",
    "DictDiff",
    "all_active_prompt_versions",
    "load_prompt",
    "load_prompt_with_version",
    "utc_now_iso",
]
