# Compare Workflow or Audit variants

Open **Evals** to list experiments. Filter by lifecycle, control mode,
evaluation workspace or dataset. **Legacy evaluation workspaces** retains
the previous Run grouping and its links.

## Create a native experiment

1. Open **Datasets**, choose an evaluation workspace and create or import a
   dataset. Give each case a stable ID, a visible objective, exact inputs and
   required output roles. Imported private rubrics are excluded unless you
   explicitly include them. Saving creates an immutable revision.
2. Choose **New experiment**. Select Workflow or Audit and the A/B families.
   A is the baseline; B is the candidate. New selection proposes the latest
   available version and shows the exact choice. Adjust input/output mappings,
   task parameters and supported execution settings when needed.
3. Choose a dataset revision and cases. Select registered deterministic checks
   or a human review with a pinned private rubric. Set repetitions, concurrency,
   time allowance and any optional observed-token threshold.
4. Save the draft and choose **Prepare**. Review the matrix size, each arm's
   eligible/unsupported/blocked counts, equality-pin coverage and budgets.
   Prepare freezes the plan and does not launch a Run or Audit.
5. Choose **Start** and confirm the reviewed plan. Execution continues after
   closing the browser. A whole Audit counts as one member even when it has
   multiple child Runs.

A saved draft survives reload. Edits require another preparation; a frozen
experiment uses **Duplicate** to create an editable draft. After a network
failure, return to the experiment and recover the original command receipt.
Pause, Resume and Cancel display the confirmed server state. Cancelling can
remain pending while already accepted executions drain.

An observed-token threshold is an admission/drain policy, not a precise billing
cap. Missing counters do not mean free execution. Existing Run/Audit budgets
still apply.

## Compare and review

**Overview** shows progress, coverage and quality against every expected member.
**Comparison** pairs the same case/sample across A/B. The initial filter focuses
on regressions and unresolved pairs; **All pairs** includes the full matrix.
**Attempts** shows member state and evidence; **Setup** retains the frozen plan.

Token and duration distributions use complete matching pairs and expose their
excluded coverage. A measured failed execution keeps its known cost and time.
Unknown values stay unavailable. p50/p90 describe observed variation, not
confidence intervals. Use a chart's data table, a histogram bin or a case
difference to inspect the corresponding evidence. Audit duration is its parent
interval; child Runs are evidence, not additional samples.

Open a pair for exact result/assessment revisions and Run/Audit/artifact links.
**Review A/B** retrieves the private rubric for that result. Choose an explicit
decision, explain it and save/select the assessment. Your authenticated identity
is retained. If another update changes the result or revision, refresh and
review the new evidence before selecting it. Closing a review does not submit it.

Exported reports keep attribution, coverage and selected evidence references.
They omit private rubrics and expected answers. A completed execution alone
does not establish quality; the conclusion is evaluated against the experiment's
explicit checks and comparison gates.

## External experiments

External experiments use the same comparison and review pages. Their source
and latest producer activity are visible. The external producer owns submission
and finalization; native lifecycle controls are absent.

Playground is an optional public-API client. Native experiments require no
Playground installation or service. Existing portable plans and direct CLI
commands remain supported; managed mode does not adopt legacy Runs by labels.

See [the managed contract](../spec/30-managed-evals.md) for API semantics and
[the release gate](../testing/evals-release-gate.md) for reproducible checks.
