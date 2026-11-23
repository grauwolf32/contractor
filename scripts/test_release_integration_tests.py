"""Focused iterative cases must enter the canonical integration pass once."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import release_integration_tests as integration


class IterativeIntegrationSelectionTest(unittest.TestCase):
    def test_tagged_completion_and_iterative_subcases_are_one_top_level_selection(self):
        package = "github.com/grauwolf32/contractor/internal/fixture"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "internal/fixture"
            fixture.mkdir(parents=True)
            (fixture / "ordinary_test.go").write_text(
                "package fixture\nfunc TestCompletion(t *testing.T) {}\n"
                "func TestScale(t *testing.T) {}\n"
            )
            (fixture / "process_test.go").write_text(
                "//go:build e2e\n\npackage fixture\n"
                "func TestCompletionProcess(t *testing.T) {}\n"
            )
            completion = {
                "matrix": lambda: {"go": {package: ["TestCompletion"]}},
                "PATTERN": "^TestCompletion.*$",
            }
            iterative = {
                "matrix": lambda: {
                    "postgres": {package: ["TestScale/custom", "TestScale/generic", "TestTagged"]},
                },
            }
            with (
                mock.patch.object(integration, "ROOT", root),
                mock.patch.object(integration, "discover", return_value={"./internal/fixture": {"TestTagged"}}),
                mock.patch.object(integration.runpy, "run_path", side_effect=[completion, iterative]),
            ):
                selected = integration.release_selection()
            self.assertEqual(selected, {"./internal/fixture": {"TestTagged", "TestCompletion", "TestScale"}})
