import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from contractor.tools.fs.models import FileLoc, FsEntry
from contractor.utils.formatting import xml_escape


@dataclass(slots=True)
class FileFormat:
    with_types: bool = True
    with_file_info: bool = True
    # Agents thread a shared output-format knob here. "str"/"xml" have dedicated
    # rendering; "json" and other accepted values ("yaml"/"markdown") render as
    # JSON — FileFormat has no yaml/markdown renderer, so they fall back to json.
    _format: Literal["str", "json", "xml", "yaml", "markdown"] = "json"
    loc: Literal["lines", "bytes"] = "lines"

    def _format_loc(self, loc: FileLoc) -> str | dict[str, Any]:
        if self.loc == "bytes":
            payload: dict[str, Any] = {
                "byte_start": loc.byte_start,
                "byte_end": loc.byte_end,
            }
        else:
            payload = {
                "line_start": loc.line_start,
                "line_end": loc.line_end,
            }

        if loc.content is not None:
            payload["content"] = loc.content

        if self._format == "str":
            return json.dumps(payload, ensure_ascii=False)

        if self._format == "xml":
            parts = ["<loc>"]
            for key, value in payload.items():
                parts.append(f"<{key}>{xml_escape(str(value))}</{key}>")
            parts.append("</loc>")
            return "".join(parts)

        return payload

    def format_fs_entry(self, entry: FsEntry) -> str | dict[str, Any]:
        kind = "dir" if entry.is_dir else "file"
        payload: dict[str, Any] = {}

        if self.with_file_info:
            payload.update(
                {
                    "kind": kind,
                    "name": entry.name,
                    "path": entry.path,
                    "size": entry.size,
                }
            )

        if self.with_types and entry.filetype is not None:
            try:
                payload["filetype"] = asdict(entry.filetype)
            except Exception:
                payload["filetype"] = str(entry.filetype)

        if entry.loc is not None:
            payload["loc"] = self._format_loc(entry.loc)

        if self._format == "str":
            return json.dumps(payload, ensure_ascii=False)

        if self._format == "xml":
            parts = [f"<{kind}>"]
            xml_payload = dict(payload)
            xml_payload.pop("kind", None)

            for key, value in xml_payload.items():
                if isinstance(value, (dict, list)):
                    serialized = json.dumps(value, ensure_ascii=False)
                    parts.append(f"<{key}>{xml_escape(serialized)}</{key}>")
                else:
                    parts.append(f"<{key}>{xml_escape(str(value))}</{key}>")

            parts.append(f"</{kind}>")
            return "".join(parts)

        return payload

    def format_file_list(
        self,
        files: Sequence[FsEntry | None],
    ) -> str | list[dict[str, Any]]:
        cleaned = [file for file in files if file is not None]

        if self._format == "str":
            return "\n".join(str(self.format_fs_entry(file)) for file in cleaned)

        if self._format == "xml":
            inner = "".join(str(self.format_fs_entry(file)) for file in cleaned)
            return f"<files>{inner}</files>"

        return [self.format_fs_entry(file) for file in cleaned]  # type: ignore[return-value]

    @staticmethod
    def format_output(
        content: str,
        max_output: int,
        *,
        base_offset: int | None = None,
        max_lines: int | None = None,
    ) -> str:
        """Truncate *content* to ``max_output`` bytes on a line boundary.

        Truncation fires on whichever cap binds first: the ``max_output`` byte
        budget or (when given) the ``max_lines`` line cap. Routing the line cap
        through here — rather than pre-slicing the content — ensures the
        truncation footer (and its resume offset) is emitted in *both* cases;
        a caller that pre-trimmed to ``max_lines`` and handed in a short-line
        slice that fits the byte budget would otherwise return silently.

        When ``base_offset`` is given (the 0-based line offset of *content*
        within the original file), the truncation footer also advertises a
        ready-to-use ``offset`` so the agent can resume with
        ``read_file(offset=<offset>)``. It is left ``None`` for non-paginated
        callers (e.g. diff output) so no misleading offset is emitted.
        """
        lines = content.splitlines(True)
        out_parts: list[str] = []
        out_bytes = 0
        cut_at_line: int | None = None

        for index, line in enumerate(lines):
            if max_lines is not None and index >= max_lines:
                cut_at_line = index
                break
            line_bytes = len(line.encode("utf-8", errors="ignore"))
            if out_bytes + line_bytes > max_output:
                cut_at_line = index
                break

            out_parts.append(line)
            out_bytes += line_bytes

        if cut_at_line is None:
            return "".join(out_parts)

        def _footer(emitted: int) -> str:
            # All three fields derive from `emitted` (the lines actually kept
            # after any footer-fit trim), so they stay mutually consistent —
            # `cut_at_line` is the pre-trim cut point and would over-state what
            # was emitted / under-state what remains.
            remaining = max(0, len(lines) - emitted)
            segments = [
                f"### truncated at line: {emitted} ###",
                f"lines left in the file: {remaining} ###",
            ]
            # Only advertise a resume offset when at least one line was
            # emitted; otherwise the offset would equal the requested one
            # (a single line wider than the budget) and re-reading would loop.
            if base_offset is not None and emitted > 0:
                segments.append(f"resume with read_file offset={base_offset + emitted} ###")
            return "\n\n" + " ".join(segments)

        footer = _footer(len(out_parts))
        footer_bytes = len(footer.encode("utf-8", errors="ignore"))

        if footer_bytes > max_output:
            return footer[:max_output]

        while out_parts and (out_bytes + footer_bytes) > max_output:
            removed = out_parts.pop()
            out_bytes -= len(removed.encode("utf-8", errors="ignore"))

        # Recompute so the resume offset reflects the lines actually kept
        # after trimming for the footer (prevents skipping a popped line).
        footer = _footer(len(out_parts))
        return "".join(out_parts) + footer
