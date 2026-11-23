import unittest

from check_spec import check


class CheckSpecTest(unittest.TestCase):
    def test_consistent_documents_pass(self):
        docs = {
            "artifacts/artifact-store.md": (
                "**ART-R1.** A revision is immutable.\n\n"
                "- **ART-R2.** Uploads are bounded by LIM-ART-1.\n\n"
                "See [PRJ-R1](projects.md).\n"
            ),
            "artifacts/projects.md": "**PRJ-R1.** A Project has one owner; see ART-R1.\n",
            "platform/limits.md": "| ID | Value |\n| --- | --- |\n| `LIM-ART-1` | 64 MiB |\n",
        }
        self.assertEqual(check(docs), [])

    def test_task_references_are_rejected(self):
        docs = {"a/b.md": "Verified by V32-008 in [the gate](../../tasks/done/v32-008-x.yml).\n"}
        errors = check(docs)
        self.assertIn("a/b.md:1: task reference 'V32-008'; specs do not cite tasks", errors)
        self.assertIn("a/b.md:1: task reference 'tasks/done/v32-008-x.yml'; specs do not cite tasks", errors)

    def test_duplicate_and_unknown_ids_are_rejected(self):
        docs = {
            "a.md": "**ART-R1.** One.\n\nUses LIM-ART-9 and ART-R7.\n",
            "b.md": "**ART-R1.** Two.\n",
            "platform/limits.md": "| `LIM-ART-1` | 1 |\n| `LIM-ART-1` | 2 |\n",
        }
        errors = check(docs)
        self.assertIn("b.md: requirement ART-R1 is also defined in a.md", errors)
        self.assertIn("platform/limits.md: registry entry LIM-ART-1 is also defined in platform/limits.md", errors)
        self.assertIn("a.md: unknown requirement ART-R7", errors)
        self.assertIn("a.md: unknown registry entry LIM-ART-9", errors)

    def test_requirements_use_the_declared_prefix(self):
        docs = {"execution/runs.md": "Requirement prefix: `RUN`\n\n**RUN-R1.** One.\n\n**SCH-R2.** Two.\n"}
        self.assertEqual(
            check(docs),
            ["execution/runs.md: requirement SCH-R2 does not use the document prefix RUN"],
        )

    def test_broken_links_are_rejected(self):
        docs = {"execution/runs.md": "See [Scheduler](scheduler.md#x), [old](../19-audits.md) and [web](https://example.com).\n"}
        existing = {"execution/scheduler.md"}
        self.assertEqual(check(docs, existing.__contains__), ["execution/runs.md: broken link ../19-audits.md"])

    def test_ids_in_code_are_examples(self):
        docs = {"conventions.md": "Write `**ART-R12.** text` and refer to `[ART-R12](x.md)`.\n\n```\nLIM-ART-3\n```\n"}
        self.assertEqual(check(docs, lambda _: False), [])


if __name__ == "__main__":
    unittest.main()
