"""Readiness consumers cannot observe a partially written fixture payload."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from fakes.process_markers import atomic_marker_source


@pytest.mark.parametrize("contents", ["12345", "12345 67890"])
def test_ready_marker_is_absent_until_the_complete_payload_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: str
) -> None:
    marker = tmp_path / "ready"
    partial_write = threading.Event()
    finish_write = threading.Event()
    failures: list[BaseException] = []
    split = len(contents) // 2

    def write_in_two_parts(path: Path, data: str, *, encoding: str) -> int:
        with path.open("w", encoding=encoding) as stream:
            stream.write(data[:split])
            stream.flush()
            partial_write.set()
            if not finish_write.wait(3):
                raise TimeoutError("fixture marker writer was not released")
            stream.write(data[split:])
        return len(data)

    monkeypatch.setattr(Path, "write_text", write_in_two_parts)

    def publish() -> None:
        try:
            exec(atomic_marker_source(marker, repr(contents)), {})
        except BaseException as error:
            failures.append(error)

    writer = threading.Thread(target=publish)
    writer.start()
    try:
        assert partial_write.wait(3), "fixture did not reach the partial-write barrier"
        assert not marker.exists(), "an incomplete payload was advertised as ready"
        staging = marker.with_name(marker.name + ".tmp")
        assert staging.read_text() == contents[:split]
    finally:
        finish_write.set()
        writer.join(3)
    assert not writer.is_alive(), "fixture marker writer outlived its owner"
    assert failures == []
    assert marker.read_text() == contents
    assert not staging.exists()
