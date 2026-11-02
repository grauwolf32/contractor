# New check · OpenDesign exploration

Non-normative V429-020 preview, approved for exploration on 2026-11-02.
Production UI, APIs, published profiles and launch behavior are unchanged.

```sh
node docs/design/ui/experiments/new-check-opendesign-2026-11-02/serve.mjs 17567
node docs/design/ui/experiments/new-check-opendesign-2026-11-02/verify.mjs
```

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
for the user-requested V429-021 objective alignment correction. The objective
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

Production transfer, if requested, must reuse current check-type selection,
material matching, readiness, exact revisions, labels/limits, error handling,
idempotent create/start and conflict reconciliation owners. The presentation
seed is not another readiness or launch implementation. Live web scan setup is
outside this exploration.
