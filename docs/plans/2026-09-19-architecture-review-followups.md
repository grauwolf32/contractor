**Contractor: follow-up tasks after rechecking architectural recommendations**

Date: 2026-09-19. Based on the user's request to recheck all recommendations
after the ADK finalizer rationale was missed, then plan justified follow-ups.
The [repeat review](../research/2026-09-19-architecture-evolution-review.md)
contains evidence, decision history and a verdict for each of R01–R19.

Status as of 2026-09-20: V57-001, V57-002, V57-004 and V57-005 are complete;
V57-003 was archived after reassessing its necessity. Five pending V57 task files
were originally created at priority P2: two correct confirmed inaccuracies,
one makes an architectural decision, one describes an optional local refactor,
and one tests a hypothesis. V57-001 was completed in `ed3c4771`, V57-002 in
`f6494a55`; criteria and verification results are in their task files.
V57-005 was completed in `6f2bc096`, with a
[no-change decision for production](../research/2026-09-19-a2a-connection-reuse-decision.md).
V57-004 was completed in `7f553b22` after discussion: typed result assembly and
removal of only the artificial Audit JSON limit. Shared result limits were left
for separate discussion; [analysis and evidence](../research/2026-09-19-worker-result-assembly.md).
V57 has no remaining active pending tasks. Archiving V57-003 does not mean its
experiment was performed or the ordinary finalizer removed. Running Runs,
budgets, queues and owners of concurrent tasks were unchanged.

**Tasks and current outcomes**

| Task | Rationale | Outcome and boundary | Scope |
| --- | --- | --- | --- |
| [V57-001](../../tasks/v57-001-live-harness-effective-model.yml) — effective model in live harness | R03: the selected model does not reach current `worker@2` | Both selected workflows send the intended model through the real Runtime adapter in an offline fixture; provenance distinguishes requested model, effective route and unknown upstream revision | Small scope, Go → Python integration check |
| [V57-002](../../tasks/v57-002-documentation-status-reconciliation.yml) — status reconciliation | R04: current summaries contradict task files | V37/V39 described correctly; historical sequence separated from current state; V38/V40/V53 not declared complete | Small documentation change |
| [V57-003](../../tasks/v57-003-ordinary-completion-decision.yml) — completion investigation | R01: is the ordinary exact-copy LLM call needed? | Archived: history and typed assembly have already been examined; a separate fake-model pipeline offers disproportionate value. The contract question remains in the review | Removed from the active queue; no finalizer change implemented |
| [V57-004](../../tasks/v57-004-worker-result-assembly.yml) — decoder/assembler | R19: internal Audit JSON round trip removed | Complete: shared typed assembler, separate model decoder, only the artificial Audit encoded-size bound removed; finalizer, budgets and real wire limits preserved | Complete; 29 new boundary cases and a strengthened Audit runner regression |
| [V57-005](../../tasks/v57-005-a2a-connection-reuse-experiment.yml) — A2A reuse | R02: cost of repeated TLS handshakes measured | Complete: 9 → 1 TLS for 8 polls; local median at a 100 ms poll interval: 826.84 → 809.21 ms; 70 fault/lifecycle/response cases; no-change because of expiry/retry differences and unproven production benefit | Completed experiment; production transport unchanged |

Success evidence retention is outside V57-001: failure-only retention was the
original requirement. No third eval runner is created and V40 is not declared
ready. The fixture checks the model actually sent, not merely a YAML string;
real models and the user's backend are unnecessary.

V57-002 needs no new roadmap generator. Task files own status; editing prose
does not turn historical tests into fresh results.

V57-003 was archived after reassessment on 2026-09-20. Most preparation had
already been covered by V13/V17/V21 history and V57-004. Requiring an alternative
test-only pipeline even for a keep-current decision is excessive: predetermined
model responses neither justify a mandatory copying step nor measure its
production cost or reliability.

The question of whether the ordinary finalizer is needed remains in the
[review's R01 section](../research/2026-09-19-architecture-evolution-review.md).
If a change is selected, its specific patch must include focused checks for
budgets, cancellation, truncation, artifact authority and version transition.
No new standalone research task is assigned now.

V57-004 was implemented independently of V57-003: the shared typed assembler
exists, while the ordinary finalizer and model JSON/schema/equality checks are
preserved. Removing these checks after changing ordinary completion is outside
the completed refactor. V57-003 does not become completed because of that work.

V57-005 permits a justified `no-change` outcome. It must observe real SDK and
transport behavior, including adverse cases, rather than prove a predetermined
answer. If reuse is recommended, a separate implementation task defines
retry/body-drain/cleanup/certificate policy and verifies the production path.

**Original sequence and its update**

1. Finish the already assigned V55-004 within its existing scope. Independent
   V55-005, followed by V55-006, remains the path to the first scan release.
   V57 work is not a new dependency and does not transfer someone else's
   `in_progress` task to another implementer.
2. From the new queue, start with V57-001 and V57-002: they are confirmed fixes
   and can be implemented independently. V57-002 can proceed alongside code work.
3. V57-003 was originally proposed as the next investigation. After V57-004 and
   reassessing its value, it was removed from the queue; retaining the unresolved
   contract question is sufficient without a separate test environment.
4. V57-004 and V57-005 are complete. No new V57 task is layered on top;
   subsequent priorities come from the current product backlog and V60.

All five new tasks have `depends_on: []`: no new task's output is a mandatory
input to another. Their existing contractual foundations are implemented and
listed in the task files. P2 follows the repository follow-up policy; it does
not imply that the regression is unimportant. `active_delivery_order` was not
rewritten for this review.

**Quick wins: small changes with clear outcomes**

V57-001 was the first completed quick win. It fixes a specific existing harness
bug: the selected model now affects the actual request, and attribution no
longer describes only user intent. The main cost is regression verification
through Go configuration → Runtime adapter, so the task is more than replacing
a filename.

V57-002 was completed in parallel and improves documentation accuracy;
it does not accelerate Runs.

V57-004 was completed after detailed discussion. The Audit-generated JSON
wrapper was removed and decoder separated from assembler. The only selected
change to accepted data is removal of the intermediate Audit JSON wrapper limit;
real limits are preserved. This is a maintenance improvement; CPU/memory savings
were not measured.

V57-005 is complete: handshake count decreases, but introducing reuse is not yet
justified; see the decision above. Replacing the ordinary finalizer could remove
one mandatory model call per ordinary completion; its share of total Run time
is unknown. Archived V57-003 neither implemented nor measured this. Potential
savings alone do not justify a separate fake-model environment.

A separate uncommitted workspace digest optimization already exists in the
[memory report](../operations/runtime-memory-2026-09-19.md): local replay showed
a peak reduction from 361.1 → 205.0 MiB with the same digest. Completing its own
verification and delivery may be closer to a measured memory benefit than a
new streaming/cache redesign. Do not duplicate concurrent work or promise a
universal improvement from this single replay.

**Existing backlog: continue or clarify rather than duplicate**

| Stream | Review connection | Work within its scope |
| --- | --- | --- |
| V55-004/005/006 | Deterministic scans and the first user-facing result | Complete the described bounded scan scope; do not make pentest/evidence redesign a prerequisite |
| [V55-008](../../tasks/v55-008-deterministic-scan-planner.yml) | R15 receipts/restart | Check job identity together with Stage-attempt identity, completed replay and unknown outcomes without rescanning; do not promise arbitrary ADK Worker resume |
| [V61-003](../../tasks/v61-003-run-toolset-pinning-allocation.yml) | R14 resolved execution merge | Implement the already required pure merge and selected allocation projection; preserve Audit authority without adding a second compiler epic |
| [V61 plan](2026-09-19-toolset-runtime-configuration.md) | Separate toolset/MCP stream | 001/002/004 complete in branch, 003 under development; implementation is not yet in main. Preserve current boundaries and coordinate integration with Evals |
| [V40-002](../../tasks/v40-002-agent-instruction-paired-runner.yml), V40-003 | Live instruction evaluation | Resolve their own pins/dispatch/mapping/observation blockers and honor separate frozen-plan gates; V57-001 does not replace this |
| [V38-006](../../tasks/v38-006-eval-assessment-comparison.yml) and V38-007/008 | Remaining Evals UX | V38-001–005 are in main; current collection/comparison and subsequent UI work follow their own stream without a second experiment authority |
| [V54 draft](2026-09-19-native-pentest-audit.md) | R09 + narrow R10 | If pentest becomes the next product priority, first decide scope/identity/round-routing/recon/exchange/replay, then implement its contract/capture tasks |

The V54 draft lists future IDs, but task files have not yet been created. This
plan does not present links inside that draft as an existing implementation
backlog. If pentest is selected, the next concrete work is to finish its contract
task, not immediately promise a universal evidence contract or new main UI flow.

V37/V39/V41/V45/V50 are complete. Scheduler simplification, the eliminated
findings N+1, `tool@1` and the unified private DTO surface are not planned again.
V53-001/002 remain archived. User-approved model budgets are not tightened and
archived investigations are not resumed.

**Ideas without new implementation tasks: conditions for revisiting**

| Item | Why not assigned now | What would change the decision |
| --- | --- | --- |
| R05 CI | No timing/execution evidence of expensive repetition | Measured critical path, repeated setup and pass/skip/flaky inventory; then fix one costly step |
| R06 scanner-only profile | Model-free execution already works; lighter installation is a different feature | Compare full/scanner compositions for identical startup, first tool Run and repeated allocations, with capability parity |
| R07 private DTO codegen | No demonstrated drift or recurring maintenance cost | Repeated specific schema defects despite shared fixtures, justifying a separate pilot |
| R08 installer | No reproduced installation failure or accepted packaging requirement | One selected host/fresh-install scenario and measured bootstrap cost; upgrade/restore designed separately |
| R11 sidecar-primary | Conflicts with the purpose of annotated source; recommendation withdrawn | Only an explicit product-requirement change. For speed, first benchmark annotate/analyze while preserving output |
| R12 streaming | Buffered integrity contract has bounds and admission; inadequacy is unmeasured | Transfer RSS/copies/latency by backend and concurrency demonstrate violation of the required resource budget |
| R13 shared source index | Cache lifetime/release is intentional; reuse exists within an allocation | Repeated parse/build remains significant after the current digest fix; then decide pins/ownership/eviction contracts |
| R14 shared execution compiler | Some merge work is already V56; Audit checks are needed for authority | Specific duplication or a second completion consumer with a demonstrably common model |
| R15 transparent Stage resume | Tool receipts do not restore conversations or arbitrary side effects | A selected recovery scenario with quantified loss and an explicit unit of durable progress |
| R16 per-item settlement | Bounded batching reduces preparation cost; whole-batch retry is accepted | Measurements show retry amplification outweighs batching benefits, followed by a separate settlement design |
| R17 allocation on demand | A fixed set provides full admission and stable sessions | Representative workload demonstrates substantial idle-slot cost, followed by capacity/deadline/fairness/recovery design |
| R18 unified executor | Language does not remove distinct roles and authority | A parity experiment demonstrates reduced real integration complexity without moving failures into new layers |

R11/R13 should be measured together on one source replay: unchanged reads,
changed=false annotation, real insertion, stale snapshot, new allocation and
explicit discovery-report reuse. Count snapshot/hash, parse/build and model
analysis separately so repeated model work is not mislabeled as parser-cache
cost. This benchmark is not assigned here as a sixth mandatory task and does
not duplicate the digest optimization already performed.

**Changes that would require redesign**

V57-001/002/004 were completed through local changes to the current project.
V57-005 tested the transport and left it unchanged; V57-003 was archived.
A future finalizer replacement or transport patch may be bounded, but a completion
migration must account for every affected nonterminal Run, including queued Runs
and later Stages. Draining only current allocations does not preserve Run semantics.

A pentest profile and durable HTTP exchanges can be added to the existing product
under agreed contracts. Before the V54 decision, neither exact cost nor a
comprehensive safe migration can be promised.

Transparent resume, per-item settlement and allocation on demand redesign
execution/recovery/import/dispatch. They require new state transitions, fault
cases and a transition strategy preserving Memory, annotated outputs and provenance.
They do not automatically require replacing UI, PostgreSQL or the entire Runtime,
but they are not simple DTO additions. There is no basis for assigning a full
Contractor rewrite now.

**Verification of this plan**

During initial planning, code/history review was not presented as completed V57
acceptance. Checks covered task/index YAML without duplicate keys, uniqueness of
341 indexed IDs, dependency existence and acyclicity of the full graph. The five
V57 tasks were checked for status/priority/commit, acceptance/covers links and
contract-input paths. History/cutover was reviewed through code walkthrough,
not pytest assertions; archiving V57-003 does not change that evidence level.

All 19 review sections, 78 distinct local Markdown link targets across two
documents and two indexes, and whitespace were checked. During verification,
an existing HEAD link from docs/README.md to missing
`docs/reviews/architecture-review-2026-09-15.md` was removed; the new review did
not replace historical evidence. `git diff --check` passes.

V57-003 comparison fixtures were neither created nor run; the original acceptance
criteria remain in the archived task. The V57-005 experiment ran later and has
its own evidence; its results do not establish V57-003 completion.

During subsequent V57-001/002 implementation, offline harness, config and
Gateway checks passed: eight effective Worker routes, 24 Python tests on the
isolated HEAD and 33 on Runtime updated by concurrent work. Three documents
shared with the V38 branch passed a trial three-way text merge without conflicts;
managed Evals changes were not included in these commits. Evidence and verification
limits are recorded in the task files.

These two tasks did not change production Runtime, services or user Runs.
Concurrent Runtime/memory and V56 changes were preserved. No numerical performance
promises or calendar deadlines were assigned without measurements.
