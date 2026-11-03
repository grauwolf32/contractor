# New check · OpenDesign exploration

Non-normative V429-020 preview, approved for exploration on 2026-11-02.
The initial exploration changed no production behavior. V429-024 transfers the
approved design into the working UI; see the production section below.

```sh
node docs/design/ui/experiments/new-check-opendesign-2026-11-02/serve.mjs 17567
node docs/design/ui/experiments/new-check-opendesign-2026-11-02/verify-opendesign-pass.mjs
node docs/design/ui/experiments/new-check-opendesign-2026-11-02/verify.mjs
node docs/design/ui/experiments/new-check-opendesign-2026-11-02/verify-scope.mjs
```

The old inventory-based Go content verifier is retired. Current execution is
verified through the common Audit process gates; assertion authoring returns
with its common hypothesis-bound caller. The preview remains historical design data.

Open <http://127.0.0.1:17567/?version=new&case=ready&theme=dark&width=1440>.
The external toolbar switches candidate/current reference/generated original/input
seed, ready/missing/unsupported scenarios, theme and width. No toolbar ships in
the product. The candidate's Start, Save and Add material controls acknowledge
local demonstration actions; no check, draft or upload is created.

The visual composition is an actual OpenDesign generation using the unchanged
configured Contractor worker model through the existing gateway. The assistant
prepared the bounded functional seed and schema-valid synthetic data. OpenDesign
wrote responsive CSS and assembled the complete HTML without changing that
controller or its fixtures. `original.html`, `opendesign.css`, `rationale.md`,
`seed-submitted.html`, `prompt.txt` and `brief.md` retain the exact contributions.
`preview.html` retains the original controller and adds `review-objective.css`
for the user-requested V429-021 objective alignment correction. V429-022 adds
`review-scope.css` and a separate `review-scope.js` extension; these are assistant
review contributions, not new model output. `build-preview.mjs` assembles them
deterministically. `preview-objective-v2.html` retains the objective correction candidate;
`preview-material-scope-v3.html` retains the initial three-item editor before
the user-requested lighter, one-empty-item revision. The objective
now spans the workbench width with a compact textarea and wrapping label/hint.
`original.html` and all generation files remain unchanged. Shared fonts and
theme tokens remain.

`examples.json` contains validated real API shapes created with the UI's existing
test builders. Demo profile inputs are deliberately unambiguous (ZIP source,
YAML API spec, JSON checklist); these are not changes to the published catalog
or its matching rules. The ASVS pilot keeps its own authored identity. Input
matching says only that the format matches; it does not establish content
validity. Scope counts are profile limits, not invented parsed endpoint counts.
`baseline.json.gz` and `baseline.css.gz` losslessly retain six inert captures of
the actual React route at desktop/phone widths using these exact fixtures.

`review.json` records the run, hashes, backup receipts, retained original data and
verification. All 140 prior project files and every previous Project,
Conversation and Message row are byte-identical. Complete recoverable pre/post
state/sandbox backups passed SQLite integrity checks. The expiring single-model
relay used 22 of 24 allowed requests, is stopped, and its upstream credential was
removed. No production business data was accessed or changed.

The review server binds only to loopback, accepts GET only, serves fixed paths,
and embeds previews in an opaque sandbox with outbound connections disabled.
The verifier covers 36 layout combinations, local interactions, keyboard,
responsive iframe bootstrap/resizing, exact controller preservation and no
outbound requests or mutations. Candidate screenshots and actual reference
screenshots at 1440/390px are saved here; the larger test matrix is under the
ignored `ui/test-results/new-check-design/candidate-captures` directory.

## Material scope and custom checklist review (V429-022)

Use the toolbar's Check selector for API/source/ASVS/checklist examples. API
coverage comes from six operations in the selected synthetic OpenAPI document;
checklist coverage comes from its actual entries. These content fixtures have
new exact revisions and matching byte sizes. No endpoint subset is offered:
the demonstrated API profile inventories every operation. Source-only reviews
omit the Scope section; authored ASVS full/pilot choices remain in Advanced
options under Review coverage and retain their original profile identities.

Choose Custom checklist, then Write checklist. The composer starts with one
empty item and quiet guidance. No demo items are inserted into a new checklist.
Project materials retain their actual contents. Add/edit/remove items and Use
checklist to apply one local revision. Scope and the summary track the entries.
Dirty, empty and invalid content blocks Start. Keys survive edits/removals;
changed statements increment their item versions. Defaults are explicit under
Advanced options → Checklist defaults: static methods, always applicable, automatic review policy and
no required evidence types. Nothing changes human issue assessment. The editor
bounds itself to the demo profile's 128 items and the existing domain's UTF-8
string/document limits. Go validation feeds browser-authored output to the
existing checklist inventory owner; this is no replacement production codec.

The Engagement scope scenario shows the typed engagement under Scope, with
origins, allowed routes/methods, exclusions, identities and limits. It is a
read-only placement concept derived from the shared engagement fixture. It is
not attached as an undeclared input to the real API trace profile. Demo Start
is disabled because executable compatibility was not established. An OpenAPI
operation never grants target authority. Production transfer must respect the
profile's declared engagement input and Server-owned exact baseline and policy.

The static Current UI reference remains the original ready/missing/unsupported
API trace capture; selecting another Check affects the candidate. Original
model/seed versions retain their original controls. All old screenshots and
model files remain intact. New scope/editor/Engagement screenshots are separate.

Production transfer, if requested, must reuse current check-type selection,
material matching, readiness, exact revisions, labels/limits, error handling,
idempotent create/start and conflict reconciliation owners. The presentation
seed is not another readiness or launch implementation. Live web scan setup is
outside this exploration.

## Second actual OpenDesign pass (V429-023)

The user asked whether the manual checklist review had been run through
OpenDesign. It had not; this task performed a real fresh pass in the isolated
`contractor-checklist-polish-20261102` project, retaining V4 as the exact input.
The `checklist-opendesign-pass-2026-11-02/` directory contains that input,
brief, exact prompt, input CSS, raw generated CSS/HTML/rationale, provenance
and theme/phone captures. The active candidate is byte-identical to this
second native HTML output; no manual presentation correction was applied.

OpenDesign refined the header, Draft/Ready chip, item alignment, placeholder
contrast and footer actions. Use checklist is a neutral secondary button;
Start check keeps the primary emphasis. Both original scripts/fixtures and
shared font families/tokens remain unchanged. The native run succeeded with
the existing configured Contractor model in 11 of 24 allowed requests.

Complete before/after state and sandbox backups passed integrity checks; all
155 prior project files and every prior Project/Conversation/Message row are
identical. The relay is stopped and its upstream credential removed. No
production UI or business data changed. Choose Before this pass · V4 in the
external toolbar to compare, or Second pass original to inspect native output.
`verify-opendesign-pass.mjs` verifies the exact CSS-only insertion, scripts,
tokens, native preservation and cleanup. Existing browser and Go checks verify
the empty/editing/applied states, 128 items and retained scope behavior.

## Production transfer (V429-024)

The approved V5 composition now has a real React implementation in
`ui/src/routes/checks/start`. The objective precedes a searchable type chooser,
setup and summary. Phones fold the chooser and keep actions in document flow.
Checklist authoring starts with one empty item; Apply stays local and Start/Save
retains an ordinary exact Project Artifact through the existing Web scan
retention/reconciliation mechanism. Scope describes the exact selected small API
or checklist and declared typed Engagement. Source-only checks have no separate
Scope and standard coverage variants are in Advanced options.

`production-fixture.tsx` wraps the actual Application and route in a synthetic
read-only transport; no fixture ships in the production bundle and no preview
request creates business data. `verify-production.mjs` reproduces the local
preview and captures. Run with the existing Vite server on 17560 (or set
`NEW_CHECK_PREVIEW_ORIGIN`). The gate covers 60 layouts in all themes at
1440/1100/390/320px, exact scope contents, editor states, blocked/read-error
states, reachable mobile chooser and overflow. Full UI and domain tests also
cover shared authoring fixtures, byte/profile bounds, exact retention after a
lost answer, mismatched content, dirty edits and abandoned preparation.

Preview after generating it:
`http://127.0.0.1:17560/test-results/new-check-production/index.html?case=editor&theme=dark&width=1440`.
All prior OpenDesign outputs, native history, inputs, hashes and backups above
remain preserved. The production fixture/gate are additions, not replacements
for the reviewed model output.
