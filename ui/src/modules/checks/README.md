# Shared checks module

This module is used by Checks, Project/Audit, Inbox, Reports, Issues and the
catalog. It contains no page components or router hooks.

- `workspace.ts`: current and revision-pinned workspace queries, existing keys,
  cached revision placeholders and bounded per-list reads.
- `projection-refresh.ts`: reconcile dependent reads once intake changes or
  the owner settles; retain exact keys and cancel the previous read first.
- `reports.ts`: the report's owner and report reads with their existing polling.
- `research.ts`: exact Program, Task, hypothesis and material queries. Cursors,
  selected hypotheses and enablement are explicit inputs. Preserve 50-row page
  limits, malformed continuation rejection, request signals and terminal reads.
- `progress.ts`, `labels.ts`, `time-limit.ts`, `stop-reason.ts`: pure projections
  of the Server's facts. They do not invent a security verdict or missing data.
- `links.ts`, `list-filter.ts`: pure link/filter helpers receiving values from
  their callers; they never read or update the browser's current URL.

`routes/projects/audits/research-pages.ts` owns the existing URL cursor names,
selection clearing and Back/Next navigation, and binds them to the research
queries. Route-specific confirmations, Start/Stop/Delete commands, idempotency
and stale-revision handling remain with their existing controllers.

Consumers use these files directly; the old Project/Reports/Checks paths are
removed without compatibility re-exports. Shared API types and query-key roots
remain authoritative. Do not add another cache, polling owner or lifecycle.
