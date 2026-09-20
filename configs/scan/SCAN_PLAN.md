# Deterministic scan planning

This catalog also provides two model-free `scan-plan@1` workflows:

- `request-set-scan@1` consumes a required `requests` artifact with media type
  `application/vnd.contractor.http-requests+json` and uses the existing SQLMap
  Worker. Its example policy selects only the `query` parameter; change that
  explicit selection to match the intended checks before using this workflow.
- `target-scan-plan@1` consumes a required `targets` artifact with media type
  `text/vnd.contractor.target-list` and uses the existing nuclei and naabu
  Workers. No string Run parameters or model credentials are required.

Each Stage's `scanPlan.inputArtifact` names a required Stage context artifact.
The Stage pins its exact revision before planning. `maxInputs` is 1–1,000,
`maxJobs` is 1–100, and `maxTotalSeconds` is 1–86,400. The last budget is the
sum of selected Workers' configured timeout costs, not a physical HTTP request
limit. Every entry in `tools` has its own positive `maxJobs` and
`maxTotalSeconds` within those same caps. A budget may be too small to select
any job; the plan records that outcome instead of expanding its budget.

`tools[].worker` must cover every declared Stage Agent exactly once, with at
most four Workers and one Worker per scanner. All use `tool@1`, `scan@1`, and
the `report` result slot. The scheduler prepares this fixed set before the
Planner starts. A missing scanner binary can therefore prevent Stage
preparation; the Planner does not allocate replacement Workers.

The template bindings for generated job inputs are fixed:

| Scanner | Argument | Source | Name |
| --- | --- | --- | --- |
| nuclei | `url` | `parameter` | `target` |
| naabu | `host` | `parameter` | `target` |
| SQLMap | `request_ref` | `artifact` | `request` |
| ffuf | `url` | `parameter` | `target` |
| ffuf | `wordlist_ref` | `artifact` | `wordlist` |

Other arguments must be literals in the pinned template. SQLMap requires a
nonempty `testParameters` policy and cannot also bind its URL mode. ffuf
requires `wordlistArtifact`, naming a required Stage context artifact; Workflow
inputs for it accept `text/vnd.contractor.wordlist` or `text/plain`. These
per-job inputs do not need artificial declarations as Workflow string
parameters or Stage context inputs. This exception applies only to validated
`scan-plan@1` Stages.

The Stage exposes exactly one required `application/json` result named
`report`. Its `from.namespace` must be separate from every Worker namespace
and must not be reserved. For `scan-plan@1`, `from.name` is a prefix of at most
100 bytes: the Planner appends `.` and the first 16 hexadecimal characters of
the StageExecution ID's SHA-256 digest. The result returns the exact artifact
revision, separating reports from different Stage attempts. Individual job
reports remain in their selected Worker's namespace.

Planner model configuration, Worker model configuration, and project
workspaces are unsupported. Ordinary passthrough tool Workflows retain their
existing input and result binding rules.
