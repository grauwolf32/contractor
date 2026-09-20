# V60-036: Restore the generated Code Analysis policy reference

The required release gate failed `TestCodeAnalysisAcrossHeterogeneousRuntimeProcesses`
before Server readiness: its generated graph AgentTemplate selects the absent
`domain_worker@1` ModelPolicy. The generated shallow template has the same stale
selector. Production configuration loading correctly rejects this reference.

Commit `394018a04a57c52a1e4cdce8d0c4ef63edd74114` removed that production policy
and changed the corresponding production graph template to `worker@2`.
`stageE2EConfiguration` copies the production catalog before these two test-only
templates are generated. Their hard-coded old references therefore no longer
resolve. Other E2E Go generators contain no `modelPolicy:` entries; Taint copies
the delivered catalog and its current release process test passed in 55.19 s.

## Correction and preserved checks

Select `worker@2` in both generated templates. It retains the scripted
`worker-model` alias. The retired alias is not restored, and production policies,
timeouts, Runtime behavior and process assertions are unchanged. The separate
`configs/e2e` policy `worker@1` is not suitable here: its eight-model-call budget
cannot execute the graph script's thirteen ordinary model calls and twelve tool
calls. The shallow script requires four ordinary model and three tool calls. Each
script also requires one mandatory result finalizer, so the precheck includes
fourteen total model calls for graph and five for shallow. Result finalization
remains exercised by the existing process Gateway script.

The focused regression stages the actual source catalog and invokes
the exact process configuration generator. It loads both generated workflows,
checks parity with the current production Worker policy, verifies the scripted
model alias and complete tool surface, and checks that call budgets accommodate
the unchanged scripts. All existing heterogeneous placement, graph analysis,
overlay invalidation and lifecycle assertions remain in the process gate.

## Verification

The focused regression was run using explicit Go overlays so that the ongoing
required release invocation continued against frozen source. Before the two
selector changes it failed with the same unknown-policy error in 2.839 s. With
the proposed changes it passed in 2.899 s: three Go PASS events, zero skips.
`git apply --check` accepts the saved patch. Exact commands and local artifact
hashes are recorded in [V60-036 evidence](../../tasks/evidence/v60-036.json).

Implementation `1e21f1fa367c396e08b7d398c08f8b24a26f4539` applies the correction.
The focused command then passed on actual tracked source, without an overlay,
in 5.47 s: three Go PASS events and zero skips. Full
`make test-code-analysis-e2e` then passed in the fail-fast integrated make
invocation at source `747839c255fc3eef800281d837b1c57ab87f892d`. The unchanged
heterogeneous Runtime process completed in 77.96 s (Go package 77.975 s).
The complete target and prerequisites recorded 140 Go PASS events and
78 Python passes, with zero failures or skips. The next target began normally.

The immutable complete-target log is `.local/v60-review/v60-036-full-target.log`
(SHA256 `530e473d4aa48bef0fa7ff573ffab948943923b260a3bdd2941aa39cfa696840`), copied
from lines 1–303 before any later target output. Both required V60-036 commands
passed; completion metadata remains separate from implementation. The overall
release invocation is still in progress and is not claimed passed. No live
models or production services are used.
