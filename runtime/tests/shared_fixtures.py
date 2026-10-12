"""Holds Runtime tests to the shared fixture convention and records their reads.

The fixtures under api/testdata are shared with the Go Server and, for some,
the UI. make test-wire-cross-language runs every test_shared_* test, so every
Runtime test that uses a shared fixture is named test_shared_*. An audit hook
enforces it:

- a test that opens a shared fixture while it runs is a test_shared_* test;
- a test module whose body opens one while it is imported, for example to
  parametrize its tests, holds only test_shared_* tests;
- any other module opens one at import only if HELPER_READS names it with the
  reason; the tests that use such a helper do not check the fixture.

With CONTRACTOR_SHARED_FIXTURE_LOG naming a file, the plugin appends every
shared fixture read and the outcome of every test_shared_* test to it; with
CONTRACTOR_SHARED_FIXTURE_TESTS=1 it runs only the test_shared_* tests.
scripts/shared_fixture_gate.py reads that log.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SHARED_FIXTURES = REPOSITORY / "api" / "testdata"
TESTS = Path(__file__).resolve().parent
SHARED_TEST_PREFIX = "test_shared_"
LOG_ENV = "CONTRACTOR_SHARED_FIXTURE_LOG"
ONLY_ENV = "CONTRACTOR_SHARED_FIXTURE_TESTS"

# Helper modules that read a shared fixture at import to build test inputs,
# keyed by repository path. The fixture keeps one definition of the input;
# the tests that use the helper consume it and do not check it.
HELPER_READS = {
    "runtime/tests/fakes/spec.py": (
        "allocation_spec builds every fake allocation with the valid empty Runtime "
        "provenance fixture"
    ),
    "runtime/tests/fakes/telemetry.py": (
        "SERVER_EXPORT is the Server's telemetry export default, which the Runtime "
        "does not define itself (RCF-R102)"
    ),
}

_records: list[str] = []
_module_reads: dict[str, set[str]] = {}


class SharedFixtureNamingError(RuntimeError):
    """A test or module outside the test_shared_ convention opened a shared fixture."""


def _fixture_path(target: object) -> str | None:
    """The repository path of an opened shared fixture, or None for any other file."""

    if not isinstance(target, str | bytes | os.PathLike):
        return None
    try:
        path = Path(os.fsdecode(target))
        if not path.is_absolute():
            path = Path.cwd() / path
        path = Path(os.path.normpath(path))
    except (TypeError, ValueError, OSError):
        return None
    if not path.is_relative_to(SHARED_FIXTURES):
        return None
    return path.relative_to(REPOSITORY).as_posix()


def _reading_module() -> str:
    """The module whose body runs the open: the innermost module body on the stack.

    While a module is imported that is the importing module, even when its body
    reaches the open through a function; otherwise it is the entry script.
    """

    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == "<module>":
            filename = Path(frame.f_code.co_filename)
            if filename.is_relative_to(REPOSITORY):
                return filename.relative_to(REPOSITORY).as_posix()
            return str(filename)
        frame = frame.f_back
    return "<no module>"


def _is_test_module(module: str) -> bool:
    path = Path(module)
    return path.parent == TESTS.relative_to(REPOSITORY) and path.name.startswith("test_")


def _audit(event: str, args: tuple[object, ...]) -> None:
    if event != "open" or not args:
        return
    fixture = _fixture_path(args[0])
    if fixture is None:
        return
    module = _reading_module()
    if module in HELPER_READS:
        return
    current = os.environ.get("PYTEST_CURRENT_TEST")
    if current is None:
        if _is_test_module(module):
            _module_reads.setdefault(module, set()).add(fixture)
            _records.append(f"python-module\t{module}\t{fixture}")
            return
        raise SharedFixtureNamingError(
            f"{module} opens the shared fixture {fixture} outside a test; read it in a "
            f"{SHARED_TEST_PREFIX}* test or in the body of a test module that holds only "
            "such tests, or name the helper module in HELPER_READS with its reason"
        )
    node = current.rsplit(" ", 1)[0]
    function = node.split("::")[-1].split("[", 1)[0]
    if not function.startswith(SHARED_TEST_PREFIX):
        raise SharedFixtureNamingError(
            f"{node} opens the shared fixture {fixture}; a test that reads api/testdata "
            f"is named {SHARED_TEST_PREFIX}* so make test-wire-cross-language runs it"
        )
    _records.append(f"python\t{node}\t{fixture}")


sys.addaudithook(_audit)


def _function_name(item: pytest.Item) -> str:
    return getattr(item, "originalname", item.name)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    problems = []
    for item in items:
        module = item.path.resolve().relative_to(REPOSITORY).as_posix()
        if module in _module_reads and not _function_name(item).startswith(SHARED_TEST_PREFIX):
            fixtures = ", ".join(sorted(_module_reads[module]))
            problems.append(
                f"{item.nodeid}: its module opens {fixtures} at import, so its tests are "
                f"named {SHARED_TEST_PREFIX}*"
            )
    if problems:
        raise pytest.UsageError("\n".join(problems))
    if os.environ.get(ONLY_ENV) != "1":
        return
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        (selected if _function_name(item).startswith(SHARED_TEST_PREFIX) else deselected).append(
            item
        )
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    name = report.nodeid.split("::")[-1].split("[", 1)[0]
    if not name.startswith(SHARED_TEST_PREFIX):
        return
    if report.when == "call" or report.outcome != "passed":
        _records.append(f"python-test\t{report.nodeid}\t{report.when}:{report.outcome}")


def pytest_sessionfinish(session: pytest.Session) -> None:
    del session
    log = os.environ.get(LOG_ENV)
    if not log or not _records:
        return
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("".join(record + "\n" for record in _records))
