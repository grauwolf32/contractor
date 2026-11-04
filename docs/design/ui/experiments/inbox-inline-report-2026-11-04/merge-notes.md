# Future Inbox integration

User decision, 2026-11-04: **active test approval will be removed soon**. The user will separately request the main merge. This note authorizes neither a merge now nor early removal of its current implementation.

When that request arrives, compare with the then-current main first. Preserve its domain/API removals. Port the reviewed Inbox layout and full issue/report readers onto their current owners instead of restoring this branch's old approval components, queue fixtures, pending counts, API calls or tests. If active-test approval is already gone from main, omit that row and its detail/decision controls from the integration. Recheck keyboard order, first selection and decision counts using the remaining real queue items.

The latest preview omits **Runs and capacity**. Keep the actionable Run rows and ordinary Runs/Operations navigation; the Inbox needs no duplicate capacity/workflow summary. Production source remains unchanged until the separately requested transfer.

Retain all original OpenDesign inputs, output, backups and experiment history. The current synthetic approval example remains only to represent the current branch; it is not a requirement to bring that mechanism back during integration.
