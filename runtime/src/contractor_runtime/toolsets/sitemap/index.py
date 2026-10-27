"""An encoded-backed immutable view, indexed once without a document object graph.

The generated schema validates wire shape only. The Server owns URL/context
identity, source filtering and reduction. Index records contain byte spans and
opaque Server IDs, never another route grouping policy.
"""

from __future__ import annotations

import json
import re
import sys
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contractor_runtime.artifacts import ArtifactTransportError
from contractor_runtime.sitemap import MAX_SITEMAP_BYTES

MAX_VIEW_BYTES = 128 << 20
MAX_PAGE_BYTES = 256 << 10
MAX_PAGE_ITEMS = 100
_SCHEMAS = json.loads(Path(__file__).with_name("document_schema.json").read_bytes())["schemas"]
_NO_MEMBER = 0xFFFFFFFF
_PARTS = ("member", "summary", "annotation", "example", "evidence", "declaration")
_PART_SCHEMAS = {
    "SitemapSummary": 1,
    "SitemapAnnotation": 2,
    "SitemapEvidence": 4,
    "SitemapDeclaration": 5,
}
Span = tuple[int, int]


@dataclass(slots=True)
class Entry:
    fields: dict[str, Span] = field(default_factory=dict)
    members: list[str] = field(default_factory=list)
    parts: array = field(default_factory=lambda: array("I"))
    markers: array = field(default_factory=lambda: array("I"))
    member_omitted: array = field(default_factory=lambda: array("Q"))


class ViewIndex:
    def __init__(self, data: bytes) -> None:
        if len(data) > MAX_SITEMAP_BYTES:
            raise ArtifactTransportError("sitemap view exceeds its encoded bound")
        self.data = data
        self.entries: dict[str, Entry] = {}
        self.member_ids: dict[str, Entry] = {}
        self.suppressed: set[str] = set()
        self.sources: dict[str, Span] = {}
        self.rules: list[Span] = []
        self.markers: list[Span] = []
        self.gaps: list[Span] = []
        self._entry: Entry | None = None
        self._member = _NO_MEMBER
        self._position = 0
        self._value({"$ref": "#/components/schemas/SitemapDocument"}, 0)
        self._space()
        if self._position != len(data) or len(self.member_ids) > 50000:
            self._invalid()
        if not self.suppressed <= self.member_ids.keys():
            self._invalid()
        for span in self.markers:
            touched = set()
            for member in self.load(span)["members"]:
                entry = self.member_ids[member]
                if id(entry) not in touched:
                    entry.markers.extend(span)
                    touched.add(id(entry))
        self.entry_ids = tuple(self.entries)
        self.metadata = tuple(
            [("source", span) for span in self.sources.values()]
            + [("rule", span) for span in self.rules]
            + [("gap", span) for span in self.gaps]
        )
        self.retained_bytes = retained_size(self)
        if self.retained_bytes + MAX_PAGE_BYTES > MAX_VIEW_BYTES:
            raise ArtifactTransportError("sitemap view exceeds its total memory bound")

    def load(self, span: Span) -> Any:
        return json.loads(self.data[span[0] : span[1]])

    def _space(self) -> None:
        data, pos = self.data, self._position
        while pos < len(data) and data[pos] in b" \t\r\n":
            pos += 1
        self._position = pos

    def _token(self, expected: int) -> None:
        self._space()
        if self._position >= len(self.data) or self.data[self._position] != expected:
            self._invalid()
        self._position += 1

    def _string(self, maximum: int = 8192) -> str:
        self._space()
        start = self._position
        self._token(34)
        pos = self._position
        while True:
            end = self.data.find(b'"', pos)
            if end < 0 or end - start > maximum * 6 + 2:
                self._invalid()
            back = end - 1
            while back > start and self.data[back] == 92:
                back -= 1
            if (end - back - 1) % 2 == 0:
                break
            pos = end + 1
        self._position = end + 1
        try:
            value = json.loads(self.data[start : self._position])
            value.encode("utf-8")
        except (ValueError, UnicodeError):
            self._invalid()
        if len(value) > maximum:
            self._invalid()
        return value

    def _value(self, schema: dict, depth: int, example: bool = False) -> Span:
        if depth > 32:
            self._invalid()
        name = ""
        if "$ref" in schema:
            name = schema["$ref"].rsplit("/", 1)[-1]
            schema = _SCHEMAS[name]
        self._space()
        start = self._position
        kind = schema.get("type")
        if "oneOf" in schema:
            options = [
                _SCHEMAS[option["$ref"].rsplit("/", 1)[-1]] if "$ref" in option else option
                for option in schema["oneOf"]
            ]
            value = self._string(
                max(option.get("maxLength", len(option.get("const", ""))) for option in options)
            )
            if sum(self._scalar_valid(value, option) for option in options) != 1:
                self._invalid()
        elif kind == "object":
            if name == "SitemapGroup":
                self._entry = Entry()
            if name == "SitemapMember":
                if self._entry is None:
                    self._invalid()
                self._member = len(self._entry.members)
            self._token(123)
            fields: dict[str, Span] = {}
            self._space()
            if self._position < len(self.data) and self.data[self._position] != 125:
                while True:
                    key = self._string(128)
                    if key in fields or key not in schema["properties"]:
                        self._invalid()
                    self._token(58)
                    fields[key] = self._value(
                        schema["properties"][key], depth + 1, key == "examples"
                    )
                    self._space()
                    if self._position >= len(self.data):
                        self._invalid()
                    if self.data[self._position] == 125:
                        break
                    self._token(44)
            self._token(125)
            if not set(schema.get("required", ())) <= fields.keys():
                self._invalid()
            self._object(name, fields, (start, self._position))
        elif kind == "array":
            self._token(91)
            count = 0
            self._space()
            if self._position < len(self.data) and self.data[self._position] != 93:
                while True:
                    count += 1
                    if count > schema.get("maxItems", 50000):
                        self._invalid()
                    span = self._value(schema["items"], depth + 1)
                    if example:
                        self._part(3, span)
                    self._space()
                    if self._position >= len(self.data):
                        self._invalid()
                    if self.data[self._position] == 93:
                        break
                    self._token(44)
            self._token(93)
            if count < schema.get("minItems", 0):
                self._invalid()
        else:
            if kind == "string":
                value = self._string(schema.get("maxLength", 8192))
            else:
                end = self._position
                while end < len(self.data) and self.data[end] not in b",]} \t\r\n":
                    end += 1
                if end - start > 64:
                    self._invalid()
                try:
                    value = json.loads(self.data[start:end])
                except (ValueError, UnicodeError):
                    self._invalid()
                self._position = end
            if not self._scalar_valid(value, schema):
                self._invalid()
        return start, self._position

    @staticmethod
    def _scalar_valid(value: Any, schema: dict) -> bool:
        kind = schema.get("type")
        if kind == "string":
            return (
                isinstance(value, str)
                and schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 8192)
                and len(value.encode()) <= schema.get("x-max-utf8-bytes", 1 << 20)
                and ("pattern" not in schema or re.search(schema["pattern"], value) is not None)
                and ("const" not in schema or value == schema["const"])
                and ("enum" not in schema or value in schema["enum"])
            )
        if kind == "boolean":
            return type(value) is bool
        if kind == "integer":
            return type(value) is int and schema.get("minimum", 0) <= value <= schema.get(
                "maximum", 2**53 - 1
            )
        return False

    def _object(self, name: str, fields: dict[str, Span], span: Span) -> None:
        if name in _PART_SCHEMAS:
            self._part(_PART_SCHEMAS[name], span)
        elif name == "SitemapMember":
            if self._entry is None:
                self._invalid()
            member = self.load(fields["id"])
            if member in self.member_ids or len(self.member_ids) >= 50000:
                self._invalid()
            self.member_ids[member] = self._entry
            self._entry.member_omitted.append(self.load(fields["omitted"]))
            self._entry.members.append(member)
            # Header fields exclude every potentially large member collection.
            self._entry.parts.extend((0, self._member, fields["path"][0], fields["path"][1]))
            self._member = _NO_MEMBER
        elif name == "SitemapGroup":
            if self._entry is None:
                self._invalid()
            identity = self.load(fields["id"])
            if identity in self.entries:
                self._invalid()
            self._entry.fields = {
                key: value
                for key, value in fields.items()
                if key not in {"members", "declarations"}
            }
            self.entries[identity] = self._entry
            self._entry = None
        elif name == "SitemapSource":
            source = self.load(fields["provenance"])["sourceId"]
            if source in self.sources:
                self._invalid()
            self.sources[source] = span
        elif name == "SitemapMarker":
            self.suppressed.update(self.load(fields["members"]))
            self.markers.append(span)
        elif name == "SitemapRule":
            self.rules.append(span)
        elif name == "SitemapGap":
            self.gaps.append(span)

    def _part(self, kind: int, span: Span) -> None:
        if self._entry is None:
            self._invalid()
        self._entry.parts.extend((kind, self._member, *span))

    @staticmethod
    def _invalid() -> Any:
        raise ArtifactTransportError("sitemap API returned an invalid document")


def retained_size(root: object) -> int:
    """Measure the actual retained graph, including bytes and typed-array buffers."""
    seen: set[int] = set()

    def size(value: object) -> int:
        if id(value) in seen:
            return 0
        seen.add(id(value))
        result = sys.getsizeof(value)
        if isinstance(value, dict):
            return result + sum(size(key) + size(item) for key, item in value.items())
        if isinstance(value, (tuple, list, set, frozenset)):
            return result + sum(size(item) for item in value)
        if isinstance(value, Entry):
            return result + sum(size(getattr(value, key)) for key in value.__slots__)
        if isinstance(value, ViewIndex):
            return result + size(value.__dict__)
        return result

    return size(root)
