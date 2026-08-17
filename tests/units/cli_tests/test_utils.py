from __future__ import annotations

from pathlib import Path

import pytest
from google.genai import types

from cli.utils import save_artifact


class _ArtifactService:
    def __init__(self, artifacts: dict[str, types.Part]) -> None:
        self.artifacts = artifacts

    async def list_artifact_keys(self, **kwargs) -> list[str]:
        return list(self.artifacts)

    async def load_artifact(self, *, filename: str, **kwargs) -> types.Part | None:
        return self.artifacts.get(filename)


@pytest.mark.asyncio
async def test_save_artifact_preserves_text_and_inline_bytes(tmp_path: Path):
    binary = b"\x00\xffPNG\x80"
    service = _ArtifactService(
        {
            "report.txt": types.Part.from_text(text="hello \u03c0\n"),
            "files/blob.bin": types.Part.from_bytes(
                data=binary,
                mime_type="application/octet-stream",
            ),
        }
    )

    saved = await save_artifact("app", "user", tmp_path, service)  # type: ignore[arg-type]

    assert saved == [tmp_path / "report.txt", tmp_path / "files/blob.bin"]
    assert (tmp_path / "report.txt").read_text(encoding="utf-8") == "hello \u03c0\n"
    assert (tmp_path / "files/blob.bin").read_bytes() == binary
