"""Counters for the work that paging one exact Artifact revision does.

``CountingPayload`` counts decodes of the whole payload, which any reader that
decodes per page performs. ``count_derivations`` counts the cache's full-payload
UTF-8 validations and line-index builds and the size of every window a text page
decodes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

import contractor_runtime.toolsets.common.artifact_read_cache as read_cache
from contractor_runtime.core.lines import split_lines


class CountingPayload(bytes):
    """Payload bytes that count how often the whole payload is decoded."""

    decodes: int

    def __new__(cls, data: bytes) -> CountingPayload:
        payload = super().__new__(cls, data)
        payload.decodes = 0
        return payload

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        self.decodes += 1
        return super().decode(encoding, errors)


@dataclass
class Derivations:
    validations: int = 0
    indexes: int = 0
    splits: int = 0
    window_bytes: list[int] = field(default_factory=list)


def count_derivations(monkeypatch: pytest.MonkeyPatch, *tool_modules: Any) -> Derivations:
    """Count full-payload work in the cache and line splits in ``tool_modules``."""

    counted = Derivations()
    is_utf8 = read_cache._is_utf8
    line_index = read_cache.LineIndex
    text = line_index.text

    def counted_is_utf8(data: bytes) -> bool:
        counted.validations += 1
        return is_utf8(data)

    def counted_index(data: bytes) -> read_cache.LineIndex:
        counted.indexes += 1
        return line_index(data)

    def counted_text(index: read_cache.LineIndex, start: int, end: int) -> str:
        counted.window_bytes.append(end - start)
        return text(index, start, end)

    def counted_split(content: str, keepends: bool = False) -> list[str]:
        counted.splits += 1
        return split_lines(content, keepends)

    monkeypatch.setattr(read_cache, "_is_utf8", counted_is_utf8)
    monkeypatch.setattr(read_cache, "LineIndex", counted_index)
    monkeypatch.setattr(line_index, "text", counted_text)
    for module in tool_modules:
        # A reader that splits the whole text per page names split_lines.
        monkeypatch.setattr(module, "split_lines", counted_split, raising=False)
    return counted


@dataclass
class HeldPreparation:
    """Holds the cache's off-loop preparation until the test releases it."""

    calls: int = 0
    started: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)


def hold_preparation(monkeypatch: pytest.MonkeyPatch) -> HeldPreparation:
    held = HeldPreparation()
    prepare = read_cache._prepare

    def held_prepare(data: bytes, utf8: bool | None, lines: bool) -> Any:
        held.calls += 1
        held.started.set()
        if not held.release.wait(timeout=60):
            raise RuntimeError("test never released the preparation")
        return prepare(data, utf8, lines)

    monkeypatch.setattr(read_cache, "_prepare", held_prepare)
    return held
