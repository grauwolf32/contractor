import os
import fnmatch
from typing import Any, Iterator

from fsspec.implementations.local import LocalFileSystem, stringify_path

from contractor.utils.formatting import norm_unicode


class RootedLocalFileSystem(LocalFileSystem):
    """
    Local filesystem sandboxed to root_path.

    All public methods accept *virtual* paths (rooted at ``/``).
    Any path that would escape the sandbox is silently treated as
    non-existent, and symlinks are never followed.
    """

    def __init__(self, root_path: str, *args: Any, **kwargs: Any) -> None:
        self.root_path = os.path.realpath(stringify_path(root_path))
        if not os.path.isdir(self.root_path):
            raise ValueError(f"root_path is not a directory: {root_path}")

        # Sentinel path returned when a lookup must be silently blocked.
        self._blocked_path = os.path.join(self.root_path, ".__blocked__")
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_within_sandbox(self, real_path: str) -> bool:
        """Return True if *real_path* is the root or a descendant of it."""
        return real_path == self.root_path or real_path.startswith(
            self.root_path + os.sep
        )

    def _to_virtual(self, real_path: str) -> str:
        """Convert an absolute, resolved host path to a virtual path."""
        rel = os.path.relpath(real_path, self.root_path)
        if rel == ".":
            return "/"
        return "/" + rel.replace(os.sep, "/")

    def _normalize_virtual(self, path: str | None) -> str:
        """Coerce caller-supplied path to a canonical virtual form."""
        if path in (None, "", "/"):
            return ""
        return path

    def _strip_protocol(self, path: str) -> str:
        """
        Map a virtual (or already-absolute) path to a host path.

        Returns ``self._blocked_path`` for any path that escapes the sandbox.
        """
        path = norm_unicode(stringify_path(path))

        if path.startswith("file://"):
            path = path[7:]

        # Already an absolute host path inside the sandbox.
        real = os.path.realpath(path)
        if self._is_within_sandbox(real):
            return real

        # Treat as a virtual path relative to the root.
        if path in ("", "/"):
            return self.root_path

        candidate = os.path.normpath(os.path.join(self.root_path, path.lstrip("/")))
        resolved = os.path.realpath(candidate)

        if self._is_within_sandbox(resolved):
            return candidate

        return self._blocked_path

    def _is_blocked(self, host_path: str) -> bool:
        return host_path == self._blocked_path or not os.path.exists(host_path)

    @staticmethod
    def _is_safe_entry(base: str, name: str) -> bool:
        """Return True if *name* inside *base* is not a symlink."""
        return not os.path.islink(os.path.join(base, name))

    # ------------------------------------------------------------------
    # Public API overrides
    # ------------------------------------------------------------------

    def walk(
        self, path: str = "", **kwargs: Any
    ) -> Iterator[tuple[str, list[str], list[str]]]:
        host_root = self._strip_protocol(self._normalize_virtual(path))

        if self._is_blocked(host_root):
            return

        for current_root, dirs, files in os.walk(host_root, followlinks=False):
            real_root = os.path.realpath(current_root)

            if not self._is_within_sandbox(real_root):
                dirs.clear()
                continue

            # Prune symlinked directories so os.walk never descends into them.
            dirs[:] = [d for d in dirs if self._is_safe_entry(current_root, d)]

            yield self._to_virtual(real_root), dirs, files

    def ls(
        self, path: str = "", detail: bool = False, **kwargs: Any
    ) -> list[dict[str, Any] | str]:
        host_path = self._strip_protocol(self._normalize_virtual(path))

        if host_path == self._blocked_path:
            return []

        try:
            entries = super().ls(host_path, detail=True, **kwargs)
        except FileNotFoundError:
            return []

        result: list[dict[str, Any] | str] = []
        for entry in entries:
            real = os.path.realpath(entry["name"])

            if not self._is_within_sandbox(real):
                continue

            virtual = self._to_virtual(real)

            if detail:
                result.append({**entry, "name": virtual})
            else:
                result.append(virtual)

        return result

    def glob(self, pattern: str, **kwargs: Any) -> list[str]:
        """
        Sandbox-safe glob with Python-like semantics.

        Returns virtual paths such as ``/file.txt`` or ``/dir/inner.txt``.
        """
        if not pattern:
            return []

        pattern = norm_unicode(pattern.lstrip("/")) or ""

        # Reject obvious traversal attempts.
        if ".." in pattern.split("/"):
            return []

        recursive = "**" in pattern
        matches: set[str] = set()

        if recursive:
            walker = os.walk(self.root_path, followlinks=False)
        else:
            try:
                top_entries = os.listdir(self.root_path)
            except FileNotFoundError:
                return []
            walker = iter([(self.root_path, [], top_entries)])

        tail_pattern = pattern.rsplit("/", 1)[-1] if recursive else ""

        for host_root, dirs, files in walker:
            # Prune symlinked directories.
            dirs[:] = [d for d in dirs if self._is_safe_entry(host_root, d)]

            rel_root = os.path.relpath(host_root, self.root_path)
            if rel_root == ".":
                rel_root = ""

            for name in files:
                normalized_name = norm_unicode(name) or name
                host_path = os.path.join(host_root, normalized_name)

                if os.path.islink(host_path):
                    continue

                rel_path = (
                    f"{rel_root}/{normalized_name}" if rel_root else normalized_name
                )
                rel_path = norm_unicode(rel_path.replace(os.sep, "/")) or rel_path

                if fnmatch.fnmatch(rel_path, pattern):
                    matches.add("/" + rel_path)
                elif (
                    recursive
                    and "/" not in rel_path
                    and fnmatch.fnmatch(normalized_name, tail_pattern)
                ):
                    matches.add("/" + normalized_name)

        return sorted(matches)
