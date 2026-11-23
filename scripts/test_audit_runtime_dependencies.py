"""Offline tests for the Runtime advisory audit; run by make lint."""

import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import audit_runtime_dependencies as audit

# Stands in for pip-audit: reports every pin in -r, the known-vulnerable one
# with an advisory, and exits like pip-audit. BLIND reports no advisories.
FAKE_SCANNER = """
import json, re, sys
requirements = open(sys.argv[sys.argv.index("-r") + 1]).read()
pins = re.findall(r"^([A-Za-z0-9_.-]+)==([^ \\\\\\n]+)", requirements, re.M)
blind = {blind}
dependencies = [
    {{"name": name, "version": version,
      "vulns": [] if blind or (name, version) != ("urllib3", "1.26.4") else [{{"id": "PYSEC-2021-108"}}]}}
    for name, version in pins
]
print(json.dumps({{"dependencies": dependencies, "fixes": []}}))
sys.exit(1 if any(dependency["vulns"] for dependency in dependencies) else 0)
"""


class RuntimeAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        # Keep the audit's report lines out of the lint output.
        quiet = contextlib.ExitStack()
        self.addCleanup(quiet.close)
        quiet.enter_context(contextlib.redirect_stdout(io.StringIO()))
        quiet.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def scanner(self, blind: bool = False) -> list[str]:
        path = self.directory / ("blind.py" if blind else "scanner.py")
        path.write_text(FAKE_SCANNER.format(blind=blind))
        return [sys.executable, str(path)]

    def lock(self, text: str) -> Path:
        path = self.directory / "runtime-requirements.txt"
        path.write_text(text)
        return path

    def test_clean_lock_passes(self) -> None:
        audit.audit(self.scanner(), self.lock("requests==2.34.2 \\\n    --hash=sha256:00\n"), self.directory)

    def test_known_vulnerable_pin_fails_the_audit(self) -> None:
        with self.assertRaisesRegex(SystemExit, "have published advisories"):
            audit.audit(self.scanner(), self.lock(audit.KNOWN_VULNERABLE), self.directory)

    def test_scanner_that_misses_the_known_advisory_is_not_trusted(self) -> None:
        with self.assertRaisesRegex(SystemExit, "cannot be trusted"):
            audit.audit(self.scanner(blind=True), self.lock("requests==2.34.2\n"), self.directory)

    def test_scanner_failure_without_report_fails(self) -> None:
        with self.assertRaisesRegex(SystemExit, "pip-audit failed"):
            audit.scan([sys.executable, "-c", "import sys; sys.exit(2)"], self.lock("requests==2.34.2\n"))

    def test_export_checks_the_lock_against_pyproject(self) -> None:
        with mock.patch.object(audit.subprocess, "run") as run:
            audit.export_runtime_lock(self.directory / "out.txt")
        command = run.call_args.args[0]
        self.assertIn("--locked", command)
        self.assertNotIn("--frozen", command)

    def test_scanner_install_requires_hashes_for_its_whole_closure(self) -> None:
        with mock.patch.object(audit.subprocess, "run") as run:
            audit.install_scanner(self.directory)
        self.assertIn("--require-hashes", run.call_args_list[-1].args[0])
        blocks = re.split(r"\n(?=\S)", audit.SCANNER_REQUIREMENTS.read_text())
        pins = [block for block in blocks if block and not block.startswith("#")]
        self.assertIn("pip-audit==2.10.1", "".join(pins))
        for block in pins:
            self.assertRegex(block, r"^[A-Za-z0-9_.-]+==\S+ \\")
            self.assertIn("--hash=sha256:", block)


if __name__ == "__main__":
    unittest.main()
