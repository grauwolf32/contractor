# Operations · OpenDesign exploration

User chose Operations on 2026-11-02 after approving New check. This experiment
starts with Operations Overview and labelled section navigation. It preserves
current public facts and proposes bounded read-only process/allocation
drill-downs. Production UI/API/execution and configuration behavior is unchanged.

The actual React baseline is rendered by `current-fixture.tsx` through Application
and the real Operations route with a synthetic read-only PublicAPI transport and
local event socket. Every populated OperationsSnapshot is validated by the
current public schema. `capture-current.mjs` retains ten baseline captures and
compressed exact DOM/CSS references. `fixtures.json` also contains the exact
shared app icon markup, copied from Icon through React rendering.

`build-seed.mjs` combines the existing global theme tokens/font families, the
bounded fixtures and `controller.js` into `seed.html`. The controller owns data,
labels, local disclosures, navigation acknowledgements and error behavior.
OpenDesign receives this seed and `brief.md` in a NEW isolated native project;
its contribution is recorded separately as presentation CSS and assembled HTML.
No private deployment inventory, repository mount or business write is used.

## Reading the design

- The summary describes currently observed Runtime processes and allocations,
  not saved principal inventory or compatible Run capacity.
- The slot distribution preserves idle/busy/reserved/draining/fenced states.
- Reconciliation follows the current Overview's exact rule: a reported reason OR
  differing observed/authoritative allocation IDs. A reserved assignment awaiting
  a Runtime observation is qualified separately from a fenced lease failure.
- Process disclosures keep full identity, observed process versus reconciled
  slot, both allocation references, missing timestamps and frozen capabilities.
- Allocation disclosures keep authoritative phase separate from observed phase
  and retain exact Run and template provenance. Prepared is not a Worker call
  in flight, and incomplete counters are not performance measurements.
- Reconnection retains its previous snapshot; a failed read has unknown counts.
  Empty is an explicit valid empty snapshot. No queue or CPU/GPU figures are
  invented. Performance/history links retain their existing destinations.

## Local review

```sh
node docs/design/ui/experiments/operations-opendesign-2026-11-02/serve.mjs 17568
node docs/design/ui/experiments/operations-opendesign-2026-11-02/verify.mjs
```

The wrapper switches OpenDesign/current/seed, normal/inspection/empty/
reconnecting/read-failure examples, light/dark/black and 1440/1100/390/320px or a
live viewport. The iframe has a restrictive CSP, no external connections and no
business writes. Current is a static capture; candidate interactions are local
preview actions. Exact Run/section links announce their destination rather than
navigate into a live deployment. Copy reports success only when the browser
actually provides clipboard access, otherwise it exposes the exact value.

The native input, output, rationale, run identity, verified recovery archives
and preservation/relay-cleanup proof are retained with generation provenance.
Previous OpenDesign project files and history are never replaced or pruned.
A production transfer follows the design review and must reuse the Operations
snapshot, live resync/backoff, capability and exact-resource owners.

## Native generation and verification

OpenDesign project `contractor-operations-20261102`, conversation
`0d9a719d-24cd-4a50-9556-0d763914966b`, run
`89e4398f-15b8-4d47-a72f-e97e39f2269d` succeeded. The configured Contractor
worker model stayed unchanged. `opendesign.css`, `original.html` (native
`index-operations-v1.html`) and `rationale.md` are the exact native outputs;
`preview.html` is byte-identical to that HTML. `seed-submitted.html`, the brief,
prompt and schema-valid fixtures preserve its exact inputs. No manual styling
corrections were applied to the generated presentation.

`provenance.json` records input identity, full verified pre/post recovery
archives, native run and cleanup. All 173 previous native project files and
all previous Project/Conversation/Message rows remain identical. The new
project adds sixteen files. The internal-only sandbox kept its read-only root
and sole state volume; the bounded relay used 20 of 24 requests, was stopped,
and its upstream credential was removed.

The browser gate passed 60 theme/width/scenario layouts, exact fixture-derived
counts and unknown states, labelled navigation, process/allocation diagnostics,
Run/section destinations, live resize and zero horizontal overflow, page errors
or external requests. Selected native captures and complete metrics are retained
in `review/`; all sixty captures remain in `ui/test-results/operations-design`.
Baseline capture refuses to overwrite the frozen references; seed regeneration
refuses input drift after native provenance exists.

Preview: <http://127.0.0.1:17568/?version=new&case=attention&theme=dark&width=1440>.
The implementation branch contains the review artifact. Production transfer
and a main merge remain subsequent work after feedback.
