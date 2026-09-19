# Local model recovery — V52-003

Retrospective verification on 2026-09-19 confirms the recovery acceptance
criteria. It does not establish a fix to the NVIDIA driver or inference backend.
No new inference, model restart, driver update or workstation reboot was needed
to close this documentation task.

The contemporaneous playground report records NVIDIA Xid 8 / RC watchdog events
for `llama-server` at **2026-09-08 01:33:16 and 02:16:25 MSK**. The first incident
failed the FastAPI taint baseline; after the second, LM Studio still advertised
the model but inference stopped advancing. LM Studio automatically replaced the
process. A bounded inference probe then returned **HTTP 200 in 6.47 seconds**;
the fresh Workflow progressed through multiple model and tool calls.

The retained result envelopes were read again at closure:

| Attempt | Contractor Run | UTC interval | Outcome |
| --- | --- | --- | --- |
| Original | `run_dadc76fa00901b0339c831d00c21adfc` | 2026-09-07 22:24:51–22:37:03 | Failed, score 0 |
| Recovery repeat | `run_50409a0ec8f4b373d6866a2ad3e7171d` | 2026-09-07 23:08:11–23:24:56 | Passed, score 1; 4/4 annotations and a nonempty diff |

The case, project manifest hash, source hash, suite hash, Workflow selector and
Runtime labels are equal between the two envelopes. The failed attempt remains
retained; it was not replaced by its successful repeat. The annotation score is
the existing bounded fixture check, not a general proof of trace correctness.

Evidence in the sibling playground checkout (paths relative to its root):

- `docs/workflow-baselines-2026-09-08.md`, sections “Initial FastAPI provider
  failure” and “Repeated local GPU watchdog”; SHA-256
  `410ad8f90dd4274e09ebc69006a435ba0bb97df3942c659526d1c30fb112a525`.
  The timestamps and probe observation above come from this historical report;
  they were not remeasured during closure.
- `results/taint-trace/20260907T222451.466690Z-14d0a0b2ecd9/result.json`;
  SHA-256 `f94dadbdcdffcb839897492bf7ac5d0261ff5b270f4ceba559e7662c96192fba`.
- `results/taint-trace/20260907T230811.973940Z-03b08426ac03/result.json`;
  SHA-256 `6ec613b1bcc84ab755a33675ba9f8a2b61ae59fd1bd000005d3394580d0f6907`.

Those local evidence files remain unchanged. This tracked summary retains the
task-specific observations without publishing private execution artifacts.

Acceptance review: **A1** has both incident timestamps and the failed attempt;
**A2** has the recorded inference probe, multi-call progress and independently
checked successful repeat; **A3** has recovery and its remaining stability
limitation recorded here. `git diff --check` passes.
