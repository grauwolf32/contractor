# V60-033 — Production Memory fixture instruction dependencies

The full release gate reproduced a test fixture defect: `TestProductionMemoryTemplatesAcrossProcesses`
failed during configuration loading, before any Runtime or model work, with
`spec.summarizer: instructions: unknown instruction ref "instructions/terminal-summarizer.md"`.
The original log is retained in `.local/v60-review/v60-033-release-failure.log`.
Production configuration validation had passed; the referenced document exists.

`stageProductionMemoryConfiguration` copied the active template, its Worker
instructions and ModelPolicies, but its partial YAML projection omitted
`summarizer.instructions.ref`. V47-004 and spec 15 require the explicitly
selected instruction ref/digest/text to remain in the resolved template and
immutable snapshot. Omitted instructions retain the legacy built-in behavior.
Removing the production reference or replacing it with built-in text would
change the configuration under test.

The correction adds the missing instruction ref to the test-only projection and
copies its selected file through the existing helper when the ref is nonempty.
It changes no production configuration, instruction bytes, policy or process
assertion. Implementation: `58784b0229aab6c4dcda55cbfac6187563aa14de`; task start:
`bd059661988d747ef01432e3b44dc6323e5911d7`.

The focused regression resolves a source catalog and its staged copy, then
compares the complete AgentTemplate, including instruction content and digests.
Distinct Worker and summarizer refs catch hard-coded filenames; literal braces
and Unicode remain exact. A second case omits summarizer instructions and
preserves the legacy resolved form. Before the correction the configured case
fails with the missing ref and the legacy case passes; both pass afterward.
The first draft of the regression used a Gateway URL without `/v1` and failed
URL validation; that test setup mistake was fixed before recording the actual
missing-instruction reproduction, and both logs remain in evidence.

Actual checks, durations, case counts and SHA-256 values are recorded in
[tasks/evidence/v60-033.json](../../tasks/evidence/v60-033.json). The real
`make test-production-memory-e2e` uses disposable PostgreSQL, Go Server,
Python Runtime processes, mTLS and a local scripted Gateway. It checks shared
Memory lifecycle, retries/reuse, isolation, telemetry and the production domain
artifact output. No live model or external provider is involved.

Verification: focused configured/legacy staging cases passed in **2.947 s**;
`make test-production-memory-e2e` passed in **105.999 s**, **1 real process test,
0 skips**; `git diff --check` passed. The complete release aggregate is recorded
separately by the V60 release review.
