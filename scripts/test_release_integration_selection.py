"""The consolidated integration pass keeps both Audit selections without duplicates."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import release_integration_tests as integration


class IntegrationSelectionTest(unittest.TestCase):
    def test_audit_gate_selection_includes_untagged_siblings_but_no_other_builds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts/test-audit-completion-e2e.py").write_text(
                'PATTERN = "AuditCompletion|^TestImporter|^TestStandard"\n'
                'def matrix(): return {"go": {"github.com/grauwolf32/contractor/internal/a": []}}\n'
            )
            (root / "scripts/test-audit-iterative.py").write_text(
                'def matrix(): return {"postgres": {\n'
                '"github.com/grauwolf32/contractor/internal/a": '
                '["TestImporterVariant/retained", "TestIterativeStorage/custom", "TestIterativeStorage/generic"],\n'
                '"github.com/grauwolf32/contractor/internal/b": ["TestTagged/recovery"]}}\n'
            )
            package = root / "internal/a"
            package.mkdir(parents=True)
            (package / "plain_test.go").write_text(
                "package a\nfunc TestAuditCompletionExtra(t *testing.T) {}\n"
                "func TestImporterVariant(t *testing.T) {}\nfunc TestStandardVariant(t *testing.T) {}\n"
                "func TestIterativeStorage(t *testing.T) {}\n"
                "func TestUnrelated(t *testing.T) {}\n"
            )
            (package / "tagged_test.go").write_text(
                "//go:build integration\n\npackage a\nfunc TestAuditCompletionTagged(t *testing.T) {}\n"
            )
            (package / "process_test.go").write_text(
                "//go:build e2e\n\npackage a\nfunc TestAuditCompletionProcess(t *testing.T) {}\n"
            )
            with (
                mock.patch.object(integration, "ROOT", root),
                mock.patch.object(
                    integration, "discover", return_value={"./internal/b": {"TestTagged"}}
                ),
            ):
                self.assertEqual(
                    integration.release_selection(),
                    {
                        "./internal/a": {
                            "TestAuditCompletionExtra",
                            "TestImporterVariant",
                            "TestStandardVariant",
                            "TestAuditCompletionTagged",
                            "TestIterativeStorage",
                        },
                        "./internal/b": {"TestTagged"},
                    },
                )
