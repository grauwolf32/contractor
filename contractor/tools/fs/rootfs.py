import os
import fnmatch

from typing import Any
from contractor.utils.formatting import norm_unicode
from fsspec.implementations.local import LocalFileSystem, stringify_path


class RootedLocalFileSystem(LocalFileSystem):
    """
    Local filesystem sandboxed to root_path.
    Forbidden paths are treated as non-existent.
    """

    def __init__(self, root_path: str, *args: Any, **kwargs: Any) -> None:
        self.root_path = os.path.realpath(stringify_path(root_path))
        if not os.path.isdir(self.root_path):
            raise ValueError(f"root_path is not a directory: {root_path}")

        self._blocked_path = os.path.join(self.root_path, ".__blocked__")
        super().__init__(*args, **kwargs)

    def walk(self, path: str = "", **kwargs: Any):
        path = "" if path in (None, "/", "") else path
        host_root = self._strip_protocol(path)

        if host_root == self._blocked_path or not os.path.exists(host_root):
            return

        for current_root, dirs, files in os.walk(host_root, followlinks=False):
            real_root = os.path.realpath(current_root)

            if not (
                real_root == self.root_path
                or real_root.startswith(self.root_path + os.sep)
            ):
                continue

            dirs[:] = [
                d for d in dirs if not os.path.islink(os.path.join(current_root, d))
            ]

            rel_root = os.path.relpath(real_root, self.root_path)
            virtual_root = (
                "/" if rel_root == "." else "/" + rel_root.replace(os.sep, "/")
            )

            yield virtual_root, dirs, files

    def ls(self, path: str = "", detail: bool = False, **kwargs: Any):
        path = "" if path in (None, "/", "") else path
        host_path = self._strip_protocol(path)

        if host_path == self._blocked_path:
            return []

        try:
            entries = super().ls(host_path, detail=True, **kwargs)
        except FileNotFoundError:
            return []

        result = []
        for entry in entries:
            host_name = entry["name"]
            real = os.path.realpath(host_name)

            if not (real == self.root_path or real.startswith(self.root_path + os.sep)):
                continue

            relative = os.path.relpath(real, self.root_path)
            virtual = "/" if relative == "." else "/" + relative.replace(os.sep, "/")

            if detail:
                normalized_entry = entry.copy()
                normalized_entry["name"] = virtual
                result.append(normalized_entry)
            else:
                result.append(virtual)

        return result

    def glob(self, pattern: str, **kwargs: Any):
        """
        Sandbox-safe glob with Python-like semantics.
        Returns virtual paths like /file.txt, /dir/inner.txt.
        """
        if not pattern:
            return []

        pattern = norm_unicode(pattern.lstrip("/")) or ""

        if ".." in pattern.split("/"):
            return []

        recursive = "**" in pattern
        matches: list[str] = []

        if recursive:
            walker = os.walk(self.root_path, followlinks=False)
        else:
            try:
                files = os.listdir(self.root_path)
            except FileNotFoundError:
                return []
            walker = [(self.root_path, [], files)]

        for host_root, dirs, files in walker:
            dirs[:] = [
                directory
                for directory in dirs
                if not os.path.islink(os.path.join(host_root, directory))
            ]

            rel_root = os.path.relpath(host_root, self.root_path)
            if rel_root == ".":
                rel_root = ""

            for name in files:
                normalized_name = norm_unicode(name) or name
                host_path = os.path.join(host_root, normalized_name)

                if os.path.islink(host_path):
                    continue

                rel_path = (
                    os.path.join(rel_root, normalized_name)
                    if rel_root
                    else normalized_name
                )
                rel_path = norm_unicode(rel_path.replace(os.sep, "/")) or rel_path

                if fnmatch.fnmatch(rel_path, pattern):
                    matches.append("/" + rel_path)
                    continue

                if recursive and "/" not in rel_path:
                    tail = pattern.split("/")[-1]
                    if fnmatch.fnmatch(normalized_name, tail):
                        matches.append("/" + normalized_name)

        return sorted(set(matches))

    def _strip_protocol(self, path):
        path = norm_unicode(stringify_path(path))

        if path.startswith("file://"):
            path = path[7:]

        # If path already points inside root_path → accept as-is
        real = os.path.realpath(path)
        if real == self.root_path or real.startswith(self.root_path + os.sep):
            return real

        # Virtual FS paths: "/", "/file.txt", "dir/a.txt"
        if path in ("", "/"):
            candidate = self.root_path
        else:
            candidate = os.path.join(self.root_path, path.lstrip("/"))

        candidate = os.path.abspath(os.path.normpath(candidate))
        resolved = os.path.realpath(candidate)

        if resolved == self.root_path or resolved.startswith(self.root_path + os.sep):
            return candidate

        # Escape attempt → silent block
        return self._blocked_path
