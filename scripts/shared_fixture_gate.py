"""Run every shared fixture test on both sides and prove each fixture's readers.

The fixtures under api/testdata hold the Go Server, the Python Runtime and the
UI to one definition of a contract. A Go test that reads one is named
TestShared* and reads it through internal/contracts/contractstest, which fails
any other test; a Runtime test that reads one is named test_shared_*, which an
audit hook in runtime/tests/shared_fixtures.py enforces. api/testdata/readers.json
declares which sides read each fixture.

make test-wire-cross-language runs this gate. It fails when

- Go code builds an api/testdata path other than through contractstest, or a
  TestShared* test sits behind a build constraint the gate does not set;
- a side has no shared test, or a discovered TestShared*/test_shared_* test
  did not run and pass (a skipped test, such as a PostgreSQL test without
  CONTRACTOR_TEST_DATABASE_URL, is not a pass);
- a fixture matches no readers entry, a declared side does not read it, an
  undeclared side reads it, or it has one reader and no recorded reason.

Usage: python3 scripts/shared_fixture_gate.py [--sources-only]
--sources-only runs only the source checks, as make lint does.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "api" / "testdata"
MANIFEST = FIXTURES / "readers.json"
CONTRACTSTEST = "internal/contracts/contractstest/"
GO_ROOTS = ("api", "cmd", "internal", "tests", "tools")
SIDES = ("go", "python", "ui")
LOG_ENV = "CONTRACTOR_SHARED_FIXTURE_LOG"
ONLY_ENV = "CONTRACTOR_SHARED_FIXTURE_TESTS"

GO_SHARED_TEST = re.compile(r"(?m)^func (TestShared\w*)\(\w+ \*testing\.T\)")
PY_SHARED_TEST = re.compile(r"(?m)^\s*(?:async\s+)?def (test_shared_\w+)\(")
GO_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
GO_PATH_LITERAL = re.compile(r'"[^"\n]*api/testdata[^"\n]*"')
GO_PATH_ELEMENTS = re.compile(r'"api"\s*,\s*"testdata"')
UI_IMPORT = re.compile(r"""(?:from|import)\s*\(?\s*["']([^"']*api/testdata/[^"']+)["']""")


@dataclass(frozen=True)
class Entry:
    readers: frozenset[str]
    reason: str


@dataclass
class Observation:
    """What one gate run saw: shared fixture reads and shared test outcomes."""

    reads: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    problems: list[str] = field(default_factory=list)


def load_manifest(path: Path = MANIFEST) -> dict[str, Entry]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if set(document) != {"description", "readers"}:
        raise ValueError(f"{path}: expected description and readers")
    entries: dict[str, Entry] = {}
    for key, value in document["readers"].items():
        if not isinstance(value, dict) or not set(value) <= {"readers", "reason"}:
            raise ValueError(f"{path}: entry {key!r} needs readers and an optional reason")
        readers = value["readers"]
        if not readers or len(set(readers)) != len(readers) or not set(readers) <= set(SIDES):
            raise ValueError(f"{path}: entry {key!r} has invalid readers {readers!r}")
        entries[key] = Entry(frozenset(readers), value.get("reason", ""))
    return entries


def declared_readers(fixture: str, manifest: Mapping[str, Entry]) -> tuple[str, Entry] | None:
    """The longest manifest key covering fixture, a path relative to api/testdata."""

    best: tuple[str, Entry] | None = None
    for key, entry in manifest.items():
        covers = fixture.startswith(key) if key.endswith("/") else fixture == key
        if covers and (best is None or len(key) > len(best[0])):
            best = (key, entry)
    return best


def fixture_files(root: Path = FIXTURES) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "README.md" and path != root / "readers.json"
    )


def check_coverage(
    fixtures: Iterable[str], manifest: Mapping[str, Entry], reads: Mapping[str, set[str]]
) -> list[str]:
    """Hold every fixture's observed readers to its declared readers."""

    problems: list[str] = []
    used: set[str] = set()
    for fixture in fixtures:
        match = declared_readers(fixture, manifest)
        if match is None:
            problems.append(f"api/testdata/{fixture}: no entry in api/testdata/readers.json")
            continue
        key, entry = match
        used.add(key)
        observed = reads.get(fixture, set())
        if len(entry.readers) < 2 and not entry.reason.strip():
            problems.append(f"api/testdata/{fixture}: one reader ({key}) and no reason")
        for side in sorted(entry.readers - observed):
            problems.append(f"api/testdata/{fixture}: declared side {side} does not read it")
        for side in sorted(observed - entry.readers):
            problems.append(f"api/testdata/{fixture}: {side} reads it but {key} does not declare {side}")
    for key in sorted(set(manifest) - used):
        problems.append(f"api/testdata/readers.json: entry {key} covers no fixture")
    return problems


def go_source_problems(root: Path = ROOT) -> list[str]:
    """Go code reaches api/testdata only through contractstest calls."""

    problems: list[str] = []
    for directory in GO_ROOTS:
        for path in sorted((root / directory).rglob("*.go")):
            relative = path.relative_to(root).as_posix()
            if relative.startswith(CONTRACTSTEST) or "/testdata/" in relative:
                continue
            problems.extend(go_file_problems(relative, path.read_text(encoding="utf-8")))
    return problems


def go_file_problems(relative: str, text: str) -> list[str]:
    problems: list[str] = []
    code = GO_COMMENT.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), text)
    lines = code.split("\n")
    for number, line in enumerate(lines, 1):
        if GO_PATH_LITERAL.search(line):
            problems.append(f"{relative}:{number}: builds an api/testdata path; use contractstest")
        for match in GO_PATH_ELEMENTS.finditer(line):
            if not re.search(r"contractstest\.(ReadFile|Path)\(\w+,\s*$", line[: match.start()]):
                problems.append(f"{relative}:{number}: builds an api/testdata path; use contractstest")
    if relative.endswith("_test.go") and GO_SHARED_TEST.search(text):
        for line in text.split("\n"):
            if line.startswith("//go:build"):
                problems.append(f"{relative}: TestShared tests behind {line!r} do not run in the gate")
                break
            if line.startswith("package "):
                break
    return problems


def discover_go_tests(root: Path = ROOT) -> dict[tuple[str, str], str]:
    """Every TestShared* test as (package directory, name) -> file."""

    tests: dict[tuple[str, str], str] = {}
    for directory in GO_ROOTS:
        for path in sorted((root / directory).rglob("*_test.go")):
            if "/testdata/" in path.as_posix():
                continue
            for name in GO_SHARED_TEST.findall(path.read_text(encoding="utf-8")):
                package = path.parent.relative_to(root).as_posix()
                tests[(package, name)] = path.relative_to(root).as_posix()
    return tests


def discover_python_tests(root: Path = ROOT) -> dict[str, str]:
    """Every test_shared_* function of the Runtime test modules -> file."""

    tests: dict[str, str] = {}
    for path in sorted((root / "runtime" / "tests").glob("test_*.py")):
        for name in PY_SHARED_TEST.findall(path.read_text(encoding="utf-8")):
            tests[f"{path.name}::{name}"] = path.relative_to(root).as_posix()
    return tests


def go_outcomes(events: Iterable[str], module: str) -> dict[tuple[str, str], str]:
    """The final action of every top-level test in go test -json output."""

    outcomes: dict[tuple[str, str], str] = {}
    for line in events:
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        test = event.get("Test", "")
        if not test or "/" in test or event.get("Action") not in {"pass", "fail", "skip"}:
            continue
        package = event["Package"].removeprefix(module + "/")
        outcomes[(package, test)] = event["Action"]
    return outcomes


def check_go_outcomes(
    discovered: Mapping[tuple[str, str], str], outcomes: Mapping[tuple[str, str], str]
) -> list[str]:
    problems: list[str] = []
    if not discovered:
        problems.append("Go: no TestShared* test exists")
    for key, path in sorted(discovered.items()):
        outcome = outcomes.get(key)
        if outcome != "pass":
            state = "did not run" if outcome is None else outcome
            problems.append(f"Go: {key[1]} ({path}) {state}")
    if not any(outcome == "pass" for outcome in outcomes.values()):
        problems.append("Go: the shared test run passed no test")
    return problems


def check_python_outcomes(discovered: Mapping[str, str], records: Iterable[str]) -> list[str]:
    """Each discovered test_shared_* function passed every case and failed none."""

    passed: set[str] = set()
    problems: list[str] = []
    for record in records:
        kind, *fields = record.split("\t")
        if kind != "python-test":
            continue
        node, outcome = fields
        function = _python_function(node)
        if outcome == "call:passed":
            passed.add(function)
        else:
            problems.append(f"Python: {node} {outcome}")
    if not discovered:
        problems.append("Python: no test_shared_* test exists")
    for function, path in sorted(discovered.items()):
        if function not in passed:
            problems.append(f"Python: {function} ({path}) did not run and pass")
    return problems


def _python_function(node: str) -> str:
    """tests/test_x.py::Class::test_y[case] -> test_x.py::test_y."""

    parts = node.split("::")
    return Path(parts[0]).name + "::" + parts[-1].split("[", 1)[0]


def observed_reads(
    records: Iterable[str],
    go_passed: set[tuple[str, str]],
    python_modules: set[str],
) -> dict[str, set[str]]:
    """Fixture -> sides whose passing shared tests read it, from the gate log."""

    reads: dict[str, set[str]] = defaultdict(set)
    for record in records:
        fields = record.split("\t")
        kind = fields[0]
        if kind == "go":
            package, test, fixture = fields[1:]
            if (package, test) in go_passed:
                reads[fixture.removeprefix("api/testdata/")].add("go")
        elif kind == "python":
            # The test's own outcome is checked separately; a read counts.
            reads[fields[2].removeprefix("api/testdata/")].add("python")
        elif kind == "python-module":
            module, fixture = fields[1:]
            if Path(module).name in python_modules:
                reads[fixture.removeprefix("api/testdata/")].add("python")
    return reads


def ui_reads(root: Path = ROOT) -> dict[str, set[str]]:
    """Fixtures a UI source imports."""

    reads: dict[str, set[str]] = defaultdict(set)
    for path in sorted((root / "ui" / "src").rglob("*.ts*")):
        for specifier in UI_IMPORT.findall(path.read_text(encoding="utf-8")):
            target = os.path.normpath(path.parent / specifier)
            relative = Path(target).relative_to(root).as_posix()
            reads[relative.removeprefix("api/testdata/")].add("ui")
    return reads


def module_path() -> str:
    for line in (ROOT / "go.mod").read_text(encoding="utf-8").splitlines():
        if line.startswith("module "):
            return line.split()[1]
    raise RuntimeError("go.mod declares no module")


def run(command: list[str], *, cwd: Path, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=dict(env), capture_output=True, text=True, check=False)


def main(argv: list[str]) -> int:
    problems = go_source_problems()
    if "--sources-only" in argv:
        report(problems, "shared fixture sources")
        return 1 if problems else 0
    manifest = load_manifest()
    go_tests = discover_go_tests()
    python_tests = discover_python_tests()
    with tempfile.TemporaryDirectory(prefix="shared-fixtures-") as directory:
        log = Path(directory) / "reads.log"
        log.touch()
        env = {**os.environ, LOG_ENV: str(log)}
        packages = sorted({"./" + package for package, _ in go_tests})
        go = run(["go", "test", "-json", "-count=1", "-run", "^TestShared", *packages], cwd=ROOT, env=env)
        outcomes = go_outcomes(go.stdout.splitlines(), module_path())
        if go.returncode != 0:
            problems.append(f"Go: go test exited with {go.returncode}")
            failed = [line for line in go.stdout.splitlines() if '"Action":"output"' in line]
            sys.stdout.write("".join(json.loads(line)["Output"] for line in failed[-200:]))
            sys.stdout.write(go.stderr)
        problems.extend(check_go_outcomes(go_tests, outcomes))
        python = run(
            ["uv", "run", "pytest", "-p", "no:randomly", "tests"],
            cwd=ROOT / "runtime",
            env={**env, ONLY_ENV: "1"},
        )
        print(python.stdout[-4000:], python.stderr[-4000:], sep="\n")
        if python.returncode != 0:
            problems.append(f"Python: pytest exited with {python.returncode}")
        records = log.read_text(encoding="utf-8").splitlines()
    problems.extend(check_python_outcomes(python_tests, records))
    go_passed = {key for key, outcome in outcomes.items() if outcome == "pass"}
    python_modules = {function.split("::")[0] for function in python_tests}
    reads = observed_reads(records, go_passed, python_modules)
    for fixture, sides in ui_reads().items():
        reads[fixture].update(sides)
    fixtures = fixture_files()
    problems.extend(check_coverage(fixtures, manifest, reads))
    print(
        f"shared fixtures: {len(fixtures)} fixtures, {len(go_passed)} Go and "
        f"{len(python_tests)} Python shared tests"
    )
    report(problems, "shared fixture gate")
    return 1 if problems else 0


def report(problems: list[str], name: str) -> None:
    if problems:
        print(f"{name} failed:", file=sys.stderr)
        for problem in problems:
            print("  " + problem, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
