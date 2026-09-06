# Offline Python repair

The exact source ZIP is already hydrated at the project root. Read
`calculator.py` and `check.py`. Fix `add` by changing `return a - b` to
`return a + b` using the selected `edit` tool. Do not change the checker.

Run `python3 -B check.py` with `exec_command`, cwd `""` (the project root) and
timeout 30 seconds. `.` is not accepted as a project-path component.
Require a completed command with exit code zero. A nonzero exit is not a passing
check; a sandbox failure must not be retried through a host or network tool.

Read `report.json` using `read_file`. Publish precisely its UTF-8 bytes as an
ordinary Artifact using `write_artifact`: namespace `builder`, name
`check_report`, media type `application/json`, base64-encoded data and no expected
revision for the first write. Return the exact revision granted by that write.
Never invent an ArtifactRef or report success from a file path alone.

Only the report is a workflow output. Source edits and unuploaded files are
disposable and disappear on release. There is no overlay export, package install,
network access, background job, PTY, skill-script execution or host mount tool.
