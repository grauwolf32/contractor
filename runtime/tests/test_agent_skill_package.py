from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

import pytest

from contractor_runtime.agent_skills import SkillPackageError, validate_package
from contractor_runtime.agent_skills.package import (
    CODE_LIMIT_EXCEEDED,
    CODE_PATH_INVALID,
    MAXIMUM_ARCHIVE_BYTES,
    MAXIMUM_DESCRIPTION_BYTES,
    MAXIMUM_FRONTMATTER_BYTES,
    MAXIMUM_MANIFEST_BYTES,
    MAXIMUM_PATH_BYTES,
    MAXIMUM_RESOURCE_BYTES,
)

FIXTURES = Path(__file__).parents[2] / "testdata" / "agent-skills" / "cases.json"


def test_shared_package_corpus() -> None:
    fixtures = json.loads(FIXTURES.read_text())
    assert fixtures["schemaVersion"] == "1.0"
    assert len(fixtures["cases"]) >= 20
    for fixture in fixtures["cases"]:
        payload = base64.b64decode(fixture["archiveBase64"])
        if fixture["expectedCode"]:
            with pytest.raises(SkillPackageError) as captured:
                validate_package(payload, fixture["expectedName"])
            assert captured.value.code == fixture["expectedCode"], fixture["id"]
            assert "\n" not in str(captured.value)
            assert len(str(captured.value)) <= 576
            continue

        package = validate_package(payload, fixture["expectedName"])
        expected = fixture["expected"]
        assert package.digest == expected["digest"], fixture["id"]
        assert {
            "name": package.manifest.name,
            "description": package.manifest.description,
            **({"license": package.manifest.license} if package.manifest.license else {}),
            **(
                {"compatibility": package.manifest.compatibility}
                if package.manifest.compatibility
                else {}
            ),
            **({"metadata": dict(package.manifest.metadata)} if package.manifest.metadata else {}),
        } == expected["manifest"], fixture["id"]
        assert [
            {"path": resource.path, "size": resource.size} for resource in package.resources
        ] == expected["resources"], fixture["id"]
        assert package.member("SKILL.md") is not None


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (b"x" * (MAXIMUM_ARCHIVE_BYTES + 1), CODE_LIMIT_EXCEEDED),
        (
            None,
            "",
        ),
    ],
)
def test_exact_archive_and_resource_boundaries(payload: bytes | None, expected_code: str) -> None:
    if payload is None:
        payload = _make_zip(
            [
                ("SKILL.md", _manifest("limits")),
                ("assets/exact.bin", b"x" * MAXIMUM_RESOURCE_BYTES),
            ],
            zipfile.ZIP_DEFLATED,
        )
    if expected_code:
        with pytest.raises(SkillPackageError) as captured:
            validate_package(payload, "limits")
        assert captured.value.code == expected_code
    else:
        assert validate_package(payload, "limits").manifest.name == "limits"


def test_resource_over_limit_is_rejected_before_returning_content() -> None:
    payload = _make_zip(
        [
            ("SKILL.md", _manifest("limits")),
            ("assets/over.bin", b"x" * (MAXIMUM_RESOURCE_BYTES + 1)),
        ],
        zipfile.ZIP_DEFLATED,
    )
    with pytest.raises(SkillPackageError) as captured:
        validate_package(payload, "limits")
    assert captured.value.code == CODE_LIMIT_EXCEEDED


def test_manifest_and_path_boundaries() -> None:
    exact_description = "d" * MAXIMUM_DESCRIPTION_BYTES
    prefix = "name: boundary\ndescription: Boundary.\n"
    exact_frontmatter = prefix + "#" + "x" * (MAXIMUM_FRONTMATTER_BYTES - len(prefix) - 1)
    cases = [
        (_skill_document("a" * 64, "Exact."), "", ""),
        (_skill_document("a" * 65, "Over."), "", CODE_LIMIT_EXCEEDED),
        (_skill_document("boundary", exact_description), "boundary", ""),
        (
            _skill_document("boundary", exact_description + "d"),
            "boundary",
            CODE_LIMIT_EXCEEDED,
        ),
        (f"---\n{exact_frontmatter}\n---\n# Body\n".encode(), "boundary", ""),
        (
            f"---\n{exact_frontmatter}#\n---\n# Body\n".encode(),
            "boundary",
            CODE_LIMIT_EXCEEDED,
        ),
    ]
    for manifest, expected_name, code in cases:
        payload = _make_zip([("SKILL.md", manifest)], zipfile.ZIP_DEFLATED)
        if code:
            with pytest.raises(SkillPackageError) as captured:
                validate_package(payload, expected_name)
            assert captured.value.code == code
        else:
            validate_package(payload, expected_name)

    base = _skill_document("boundary", "Boundary.")
    exact_manifest = base + b"x" * (MAXIMUM_MANIFEST_BYTES - len(base))
    validate_package(_make_zip([("SKILL.md", exact_manifest)], zipfile.ZIP_DEFLATED), "boundary")
    with pytest.raises(SkillPackageError) as captured:
        validate_package(
            _make_zip([("SKILL.md", exact_manifest + b"x")], zipfile.ZIP_DEFLATED),
            "boundary",
        )
    assert captured.value.code == CODE_LIMIT_EXCEEDED

    path = "assets/" + "a" * 126 + "/" + "b" * 126 + "/" + "c" * 125 + "/" + "d" * 125
    assert len(path) == MAXIMUM_PATH_BYTES
    validate_package(_make_zip([("SKILL.md", base), (path, b"")], zipfile.ZIP_STORED), "boundary")
    with pytest.raises(SkillPackageError) as captured:
        validate_package(
            _make_zip([("SKILL.md", base), (path + "d", b"")], zipfile.ZIP_STORED),
            "boundary",
        )
    assert captured.value.code == CODE_PATH_INVALID


def test_exact_stored_archive_and_expanded_boundaries() -> None:
    manifest = _skill_document("archive-boundary", "Archive boundary.")
    block = b"x" * MAXIMUM_RESOURCE_BYTES
    entries = [("SKILL.md", manifest)] + [(f"assets/b{index:02d}", block) for index in range(15)]
    entries.append(("assets/fill", b""))
    base = _make_zip(entries, zipfile.ZIP_STORED)
    fill = MAXIMUM_ARCHIVE_BYTES - len(base)
    assert 0 < fill <= MAXIMUM_RESOURCE_BYTES
    entries[-1] = ("assets/fill", b"z" * fill)
    payload = _make_zip(entries, zipfile.ZIP_STORED)
    assert len(payload) == MAXIMUM_ARCHIVE_BYTES
    validate_package(payload, "archive-boundary")
    with pytest.raises(SkillPackageError) as captured:
        validate_package(payload + b"x", "archive-boundary")
    assert captured.value.code == CODE_LIMIT_EXCEEDED


def _manifest(name: str) -> bytes:
    return _skill_document(name, "Limits.")


def _skill_document(name: str, description: str) -> bytes:
    return f"---\nname: {name}\ndescription: {description}\n---\n# Body\n".encode()


def _make_zip(entries: list[tuple[str, bytes]], compression: int) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, content in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = compression
            archive.writestr(info, content)
    return output.getvalue()
