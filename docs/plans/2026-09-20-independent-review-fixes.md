# Independent review fixes — 2026-09-20

The user requested breaking the [11 reproduced findings](../research/2026-09-20-independent-code-review.md)
into tasks and starting the fixes. Implementation is based on current code and
checks, not the status of earlier reviews. Baseline commit: `5856beb5c13429f4be54856015ad7e86bd8b573d`.
Existing uncommitted changes belong to earlier work and are preserved.

| Task | Finding | Outcome |
| --- | --- | --- |
| V60-013 | CR-01 | Pending operations and CLI validators do not survive confirmed Runtime cleanup |
| V60-014 | CR-02 | WorkerHandle is copied without sharing a mutable AgentCard |
| V60-015 | CR-03 | Placement/Rebind read credentials through their own transaction |
| V60-016 | CR-04 | Resume escalation preserves configuration and correct attempt history |
| V60-017 | CR-05 | HTTP tool preserves original query bytes |
| V60-018 | CR-06 | An old session response cannot change newer authentication state |
| V60-019 | CR-07 | Audit projections receive the latest update |
| V60-020 | CR-08 | HTTP credential picker supports subsequent pages |
| V60-021 | CR-09 | Static server accepts valid encoded route identities |
| V60-022 | CR-10 | CLI download --force preserves private file permissions |
| V60-023 | CR-11 | CLI preserves positional-argument protection after -- |
| V60-024 | All | Combined verification of the fixes and updated results |
| V60-025 | Readability | Separate, justified refactors after the fixes |

The first implementation proceeds in parallel across non-overlapping areas:
Runtime (013/017), Go execution (014/015/016), UI (018–021), CLI (022/023).
Tasks 013–023 move to `in_progress` before edits. Task priority P2 follows the
repository follow-up policy; the original CR-01 finding retains priority P1.

Each fix receives a permanent regression test in its subsystem that first
reproduces the defect. DB checks use a separate temporary PostgreSQL instance;
Runtime checks use fake models and local subprocess fixtures. The Resume
migration must account for persisted histories and automatic escalation budgets.
Changes must not weaken authority, CAS, revision pins, URL bounds or release guarantees.

Implementation commits are separated by task; completion metadata records the
exact hash and checks performed. The integration task starts after the fixes
are complete. Broad structural simplification is a separate task and does not
substitute for correcting reproduced bugs. Production deployment, live-model
evals and external security targets are outside this phase.
