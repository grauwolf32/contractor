import unittest

from check_tasks import Policy, Record, validate

POLICY = Policy(
    statuses=frozenset({"pending", "in_progress", "deferred", "completed", "archived"}),
    priorities=frozenset({"P0", "P1", "P2", "P3"}),
    exit_tasks={},
)
COMMIT = "0" * 40


def task(task_id, status="pending", **fields):
    data = {
        "schema_version": "1.0",
        "id": task_id,
        "title": "t",
        "milestone": "m",
        "priority": "P1",
        "status": status,
        "depends_on": [],
        "goal": "g",
        "context": {},
        "scope": {},
        "acceptance_criteria": [],
        "tests": [],
    }
    if status == "completed":
        data["implementation_commit"] = COMMIT
    data.update(fields)
    return data


class ValidateTasksTest(unittest.TestCase):
    def test_open_and_finished_tasks_in_their_folders_pass(self):
        records = [
            Record("v1-001-open.yml", task("V1-001")),
            Record("done/v1-002-done.yml", task("V1-002", "completed", depends_on=["V1-001"])),
            Record("done/v1-003-archived.yml", task("V1-003", "archived")),
            Record("v1-004-deferred.yml", task("V1-004", "deferred")),
        ]
        self.assertEqual(validate(records, POLICY), [])

    def test_finished_task_outside_done_fails(self):
        errors = validate([Record("v1-001-x.yml", task("V1-001", "completed"))], POLICY)
        self.assertEqual(errors, ["v1-001-x.yml: completed task belongs in tasks/done/"])

    def test_open_task_inside_done_fails(self):
        errors = validate([Record("done/v1-001-x.yml", task("V1-001", "in_progress"))], POLICY)
        self.assertEqual(errors, ["done/v1-001-x.yml: in_progress task belongs in tasks/, not tasks/done/"])

    def test_undeclared_status_and_priority_fail(self):
        errors = validate([Record("v1-001-x.yml", task("V1-001", "complete", priority="P9"))], POLICY)
        self.assertIn("v1-001-x.yml: status 'complete' is not declared", errors)
        self.assertIn("v1-001-x.yml: priority 'P9' is not declared", errors)

    def test_completed_task_needs_full_commit(self):
        record = Record("done/v1-001-x.yml", task("V1-001", "completed", implementation_commit="abc123"))
        self.assertEqual(
            validate([record], POLICY),
            ["done/v1-001-x.yml: completed task needs a full implementation_commit"],
        )

    def test_file_name_duplicate_and_dependency_checks(self):
        records = [
            Record("v1-002-wrong-name.yml", task("V1-001")),
            Record("v1-001-dup.yml", task("V1-001")),
            Record("v1-003-x.yml", task("V1-003", depends_on=["V9-999"])),
        ]
        errors = validate(records, POLICY)
        self.assertIn("v1-002-wrong-name.yml: file name does not start with v1-001-", errors)
        self.assertIn("v1-001-dup.yml: duplicate id V1-001 (also v1-002-wrong-name.yml)", errors)
        self.assertIn("v1-003-x.yml: depends_on unknown task V9-999", errors)

    def test_missing_fields_and_milestone_exit_task(self):
        data = task("V1-001")
        del data["tests"]
        policy = Policy(POLICY.statuses, POLICY.priorities, {"m": "V1-404"})
        errors = validate([Record("v1-001-x.yml", data)], policy)
        self.assertIn("v1-001-x.yml: missing tests", errors)
        self.assertIn("tasks/index.yml: milestone m exit_task V1-404 has no task file", errors)


if __name__ == "__main__":
    unittest.main()
