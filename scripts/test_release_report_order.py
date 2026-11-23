"""Exercise shared release evidence ordering through the real make prerequisites."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "make" / "core.mk"


class ReleaseReportOrderTest(unittest.TestCase):
    def run_target(self, target: str) -> str:
        with tempfile.TemporaryDirectory(prefix="release-report-order-") as directory:
            root = Path(directory)
            report = root / "reports" / "shared-fixture-reads.log"
            report.parent.mkdir()
            report.write_text("stale\n")
            makefile = root / "Makefile"
            makefile.write_text(
                f"include {CORE}\n"
                ".PHONY: aggregate runtime-venv\n"
                "aggregate: RELEASE_CONSOLIDATED := 1\n"
                "aggregate: test-go test-runtime\n"
                "runtime-venv test-hardening-matrices: ; @:\n"
                "test-go:\n"
                "\t@printf 'go\\n' >> \"$(SHARED_FIXTURE_LOG)\"\n"
                "test-runtime-hardening:\n"
                "\t@printf 'python\\n' >> \"$(SHARED_FIXTURE_LOG)\"\n"
            )
            environment = os.environ.copy()
            for name in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "RELEASE_CONSOLIDATED"):
                environment.pop(name, None)
            subprocess.run(
                ["make", "--no-print-directory", "-f", str(makefile), target],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            return report.read_text()

    def test_inherited_release_flag_preserves_both_language_readers(self) -> None:
        self.assertEqual(self.run_target("aggregate"), "go\npython\n")

    def test_direct_go_initializes_evidence_before_writing(self) -> None:
        self.assertEqual(self.run_target("test-go"), "go\n")

    def test_standalone_runtime_initializes_evidence_before_writing(self) -> None:
        self.assertEqual(self.run_target("test-runtime"), "python\n")


if __name__ == "__main__":
    unittest.main()
