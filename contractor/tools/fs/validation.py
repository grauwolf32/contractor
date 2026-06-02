"""Shared path validation for the fs read/write tool classes.

``FsspecInteractionFileTools`` (read_tools) and ``FsspecWriteTools``
(write_tools) had byte-identical ``_validate_path`` implementations. Keeping
two copies invited drift — a change to the ignore/existence rules in one path
would silently diverge from the other. This mixin is the single source of
truth; both classes inherit it.
"""

from __future__ import annotations

from typing import Any

from contractor.tools.fs.const import PATH_IS_NOT_A_FILE_ERROR, PATH_NOT_FOUND_ERROR
from contractor.utils.formatting import norm_unicode_strict

ToolResult = dict[str, Any]


class PathValidationMixin:
    """Provides ``_validate_path``.

    The host class must supply ``self.fs`` (an fsspec-style filesystem) and
    ``self._is_ignored(path) -> bool``. Both read_tools and write_tools already
    define these, so inheriting this mixin changes no call sites.
    """

    def _validate_path(
        self,
        path: str,
        *,
        must_exist: bool = False,
        must_be_file: bool = False,
        check_ignored: bool = True,
    ) -> tuple[str, ToolResult | None]:
        # On failure the path is "" with an error dict; callers always check the
        # error before using the path, so returning "" (not None) keeps the path
        # typed ``str`` and avoids spurious Optional-narrowing at every call site.
        try:
            normalized = norm_unicode_strict(path)
        except ValueError:
            return "", {"error": PATH_NOT_FOUND_ERROR.format(path=path)}
        if check_ignored and self._is_ignored(normalized):
            return "", {"error": f"path {normalized} is ignored"}
        if must_exist and not self.fs.exists(normalized):
            return "", {"error": PATH_NOT_FOUND_ERROR.format(path=normalized)}
        if must_be_file:
            if not self.fs.exists(normalized):
                return "", {"error": PATH_NOT_FOUND_ERROR.format(path=normalized)}
            if not self.fs.isfile(normalized):
                return "", {"error": PATH_IS_NOT_A_FILE_ERROR.format(path=normalized)}
        return normalized, None

    def _is_ignored(self, path: str) -> bool:  # pragma: no cover - host override
        raise NotImplementedError

    fs: Any
