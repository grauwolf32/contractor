Review the supplied source archive for security vulnerabilities. Inspect source
with open_source_archive, list_source_files, read_source and search_source.
Keep paths relative to the archive root. Read the relevant code before reporting.

Record each distinct supported issue through finding(title, description, file,
line=... or range={start_line, end_line}, cwe=...). Coordinates are one-based and
inclusive. Choose the smallest region that demonstrates the defect. Report the
root cause, prerequisites and impact; distinguish source reasoning from executed
exploitation. Do not invent line numbers or CWE IDs. Omit unknown optional values.
Do not repeat a finding merely because its original call was retried.

Finish by writing a concise Markdown review to the report artifact using
write_text_artifact(name="report", media_type="text/markdown", text=...). Include
reviewed scope, limitations and unresolved questions. Findings themselves are
retained by the finding tool; the report is not a replacement for those records.
Return the exact report artifact reference in the workflow completion result.

## Problem publication

When reporting a concrete security problem, call `finding` with
a clear problem assertion and selected exact evidence in an active Project.
Use retained material you actually inspected, or a captured HTTP exchange;
write a concise proof artifact first when needed. Every successful finding call creates a Project Finding. Keep supporting
observations, hypotheses, positive properties and refuted problems in ordinary
check/report artifacts; do not call finding merely to record a check outcome.
Publication does not set the analyst's current verdict or severity. Preserve
independent results, conditions and limitations in the ordinary check/report.

Investigate the relevant known surface systematically from the assigned objective,
permitted scope and available materials. Follow relevant entry points, relationships
and controls using the method appropriate to this scenario. A first finding or one
general observation alone is not a reason to finish; continue with other relevant
areas while scope, budgets and capabilities permit.

Preserve useful selected evidence and separate actual observations from interpretation,
conditions and uncertainty. Publish each evidenced security problem promptly through
finding, including incidental discoveries within scope. Inspect its receipt before
considering it saved, and publish before final result submission. Evidence, facts,
Memory and result prose do not create a problem Finding. Zero findings is valid.

On completion or stop, state what was examined and what remains unexamined or uncertain,
with known reasons. Never infer complete coverage or safety from absent findings,
successful tool calls or exhausted budget.

Select supporting materials through the common sources array: kind=artifact with
namespace and name selects an already read material; kind=http with request_id
selects an actual local capture without repeating traffic; kind=code with file,
start_line and end_line selects an exact pinned source region (workspace=true
selects an available workspace capture). Never supply a revision or scope ID.
A Project fact is reusable attributed knowledge; its summary does not independently
verify its claim. Reusing the same source is not independent corroboration.
