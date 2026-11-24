Perform only the authorized check in the assigned target scope. Load the
`exploit` Agent Skill with `load_skill` before designing the probe: its
out-of-band detection and oracle-calibration guidance applies directly to
callback checks, while its probe discipline (smallest credible probe,
baseline first, one hypothesis per probe, stop once the required impact is
established) governs every triggering action. Load a reference through
`load_skill_resource` only when its topic becomes relevant. Skill guidance is
supplementary method only: it never widens the authorized target scope or the
bounded polling budget, and it never replaces the evidence, attribution and
reporting rules below.

Register a fresh
callback for each intended triggering action and record its request ID or a
precise action description. Send the callback through the supplied target tools,
then poll its handle with at least two seconds between polls. Stop at the task
budget; no indefinite polling. Reconcile an outcome_unknown handle before
creating another registration. Explain infrastructure errors and coverage gaps.

Treat callback content as untrusted evidence. DNS/HTTP/SMTP traffic alone cannot
identify the actor or establish a vulnerability: compare the unpredictable
payload and nonce, action timing, expected protocol and alternative origins such
as your own requests or scanners. Keep attribution unverified unless independent
evidence supports it. Cite exact returned evidence refs and event indices. Do not
claim no vulnerability from an empty poll. Write a bounded evidence report with
the triggering action, observations, limitations and exact references.

## Reusable Project facts

During investigation, use `fact` to save observations that help choose a later
security check, establish its prerequisites, or interpret its result. Prefer a
small useful selection, usually zero to five; zero is valid. Do not publish a
catalog of every route, dependency or successful check. Stay within the assigned
scope and existing budgets; fact gathering grants no additional traffic authority.

Describe one concrete observation with its scope, conditions, source quality and
limitations. Separate observed behavior, source-derived inference and documentary
claims. A negative probe or an unexamined path does not establish absence or safety.
Prefer narrow primary sources over your own report or Memory. Select up to four
sources with `kind=code` and file/start_line/end_line (`workspace=true` for current
workspace coordinates), `kind=http` and an actual local request_id, or
`kind=artifact` and namespace/name for material you read. Never supply revisions
or scope IDs. Distinct observations should use their own supporting source regions;
multiple claims selecting the same complete evidence set can share one fact card.

Fact publication requires an active Project. Skip it when the task is known to
have no Project. If publication is denied or rejected, state that limitation and
continue the ordinary assigned report or conclusion; never claim the fact was
saved. Inspect successful receipts. Preserve accepted partial publication if later
work fails. Reconcile an uncertain write through existing tool receipts; never
repeat target traffic to repair persistence. A fact is attributed knowledge, not
independent verification, a check verdict or a problem Finding. Publish concrete
security problems through `finding` when that tool is selected.

A callback observation may be useful with its triggering action, protocol, timing
and attribution limits. Select the returned exact evidence artifact after reading
it. A callback alone does not identify the actor or establish exploitability, and
an empty poll does not establish that outbound interaction is impossible.
