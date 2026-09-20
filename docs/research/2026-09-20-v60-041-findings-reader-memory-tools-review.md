# V60-041 Findings reader Memory-tool fixture review

The post-merge Findings gate at `43cea5b89ac8c9bf9d43101f399079c31aabd439`
failed because its boundary Gateway fixture expected the retired three-tool
surface. `findings-review@2` selects `findings_analyst@2`, which explicitly adds
six Memory operations. The Gateway correctly rejected that mismatch before the
empty-collection success scenario completed. Later boundary failures reuse that
same recorded Gateway error; they are not separate production findings.

The [Memory specification](../spec/08-memory-tools.md) requires the exact selected
operations. The [Findings specification](../spec/27-findings-tools-and-collections.md)
requires complete collection preparation and verified exact references before
exposing the reader. The active [Workflow](../../configs/workflows/findings_review_v2_memory.yaml)
and [AgentTemplate](../../configs/agent-templates/findings_analyst_v2_memory.yaml)
therefore require nine tools: the original `list_findings`, `read_artifact` and
`write_text_artifact`, plus `append_memory`, `list_memories`, `list_memory_tags`,
`read_memory`, `search_memory` and `write_memory`.

Task start `083e931f` preceded implementation
`c25b4b28ed0e865dd54fb0bd74b917a1306e99f0`. The one-line fixture change uses
`withMemoryTools` to add that explicit set. Expected tools remain independent
of the observed request; sorted exact equality still rejects missing or extra
operations. Missing/foreign input rejection, explicit empty success, interrupted
preparation with exact-ref reuse, conflicting evidence, invalid ZIP rejection,
unchanged conflicting bytes and the completed-stage boundary for failed
preparation all remain unchanged. No additional test duplicates this fixture.

The original complete integration invocation failed after 1229.67 seconds.
Its Findings portion produced 48 passing Runtime cases and eight passing/five
failing Go events, including parents and subtests, with no skips. The failed
boundary parent took 15.77 seconds; the independent producer/reader and retention
parents passed in 35.79 and 75.82 seconds. Their successes do not make the
Findings target pass. Earlier focused, database-race, recovery, build and compile
checks passed, as did the Memory, Project, Audit and Skills process targets.
The UI target was not reached.

The corrected full `make --trace test-findings-e2e` passed at implementation
`c25b4b28`: **13 Go events (three parents and ten subtests), 48 Runtime cases,
zero skips or failures**. The Go process package took 177.265 seconds;
Runtime tests took 0.65 seconds. The boundary parent passed in
65.58 seconds, including all five subcases. The wrapper explicitly confirmed
all required cases and fail-fast make advanced to UI installation. The subsequent
UI stack was still running when this Findings result was recorded; no overall
integration-remainder success is claimed here.

[Machine-readable evidence](../../tasks/evidence/v60-041.json) records exact
source commits, commands, failure-log boundaries and SHA-256 hashes. The original
failed invocation remains preserved separately from any corrected result.
