import json

from typing import Union, Optional, Any, Literal
from contractor.tools.fs.models import (
    FileLoc,
    FsEntry,
)
from contractor.utils.formatting import xml_escape
from dataclasses import dataclass, asdict


@dataclass(slots=True)
class FileFormat:
    with_types: bool = True
    with_file_info: bool = True
    _format: Literal["str", "json", "xml"] = "json"
    loc: Literal["lines", "bytes"] = "lines"

    def _format_loc(self, loc: FileLoc) -> Union[str, dict[str, Any]]:
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

    def format_fs_entry(self, entry: FsEntry) -> Union[str, dict[str, Any]]:
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
        files: list[Optional[FsEntry]],
    ) -> Union[str, list[dict[str, Any]]]:
        cleaned = [file for file in files if file is not None]

        if self._format == "str":
            return "\n".join(str(self.format_fs_entry(file)) for file in cleaned)

        if self._format == "xml":
            inner = "".join(str(self.format_fs_entry(file)) for file in cleaned)
            return f"<files>{inner}</files>"

        return [self.format_fs_entry(file) for file in cleaned]  # type: ignore[return-value]

    @staticmethod
    def format_output(content: str, max_output: int) -> str:
        lines = content.splitlines(True)
        out_parts: list[str] = []
        out_bytes = 0
        cut_at_line: Optional[int] = None

        for index, line in enumerate(lines):
            line_bytes = len(line.encode("utf-8", errors="ignore"))
            if out_bytes + line_bytes > max_output:
                cut_at_line = index
                break

            out_parts.append(line)
            out_bytes += line_bytes

        if cut_at_line is None:
            return "".join(out_parts)

        remaining = max(0, len(lines) - cut_at_line)
        footer = (
            f"\n\n### truncated at line: {cut_at_line} ### "
            f"lines left in the file: {remaining} ###"
        )
        footer_bytes = len(footer.encode("utf-8", errors="ignore"))

        if footer_bytes > max_output:
            return footer[:max_output]

        while out_parts and (out_bytes + footer_bytes) > max_output:
            removed = out_parts.pop()
            out_bytes -= len(removed.encode("utf-8", errors="ignore"))

        return "".join(out_parts) + footer
