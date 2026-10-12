"""The shared fixture convention (tests/shared_fixtures.py) fails what breaks it."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import shared_fixtures
from shared_fixtures import SHARED_FIXTURES, SharedFixtureNamingError

FIXTURE = SHARED_FIXTURES / "readers.json"


def test_a_test_outside_the_convention_cannot_open_a_shared_fixture() -> None:
    with pytest.raises(SharedFixtureNamingError, match="is named test_shared_"):
        FIXTURE.read_bytes()


def test_a_module_outside_the_tests_cannot_open_a_shared_fixture_at_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "opens_a_shared_fixture.py").write_text(
        f"open({str(FIXTURE)!r}, 'rb').close()\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    with pytest.raises(SharedFixtureNamingError, match="outside a test"):
        importlib.import_module("opens_a_shared_fixture")


def test_a_module_that_opens_a_shared_fixture_at_import_holds_only_shared_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = "runtime/tests/test_reads_at_import.py"
    monkeypatch.setitem(shared_fixtures._module_reads, module, {"api/testdata/x.json"})
    path = shared_fixtures.REPOSITORY / module
    items = [
        SimpleNamespace(path=path, name="test_shared_x[1]", originalname="test_shared_x"),
        SimpleNamespace(path=path, name="test_y", originalname="test_y", nodeid="y"),
    ]
    config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: None))
    with pytest.raises(
        pytest.UsageError, match=re.escape("y: its module opens api/testdata/x.json")
    ):
        shared_fixtures.pytest_collection_modifyitems(config, items)
    shared_fixtures.pytest_collection_modifyitems(config, items[:1])
