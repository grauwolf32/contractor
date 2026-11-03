# Operations Settings · focused OpenDesign refinement

User 2026-11-03 requests more work on Settings and approves the other Operations
sections. This revision changes only the Settings design preview. It preserves
the earlier native projects, artifacts and all approved tab sources.

The source of behavior is `ui/src/routes/operations/settings/index.tsx`,
`settings/models.tsx`, `ui/src/routes/settings/git-key.tsx` and
`settings/appearance.tsx`, with the Operations specification and shared UI tokens.
Models and Scheduler remain independent resources. The real Git key UI imports a
private key and returns only metadata; this revision does not suggest generating
one. The design example disables secret entry and performs no credential action.

## Design and local behavior

A wider execution column contains Models and Workflow scheduling. A separate
personal column contains Repository access and Appearance. Section jumps keep
longer views reachable. Role overrides and diagnostics are progressive disclosures.
Each Server resource shows its own draft, discard, save and stale-revision state;
a clean resource has no save footer. Both scheduling limits are edited together,
within the existing integer 1–32 bounds. Lowering limits never cancels work.

The model catalog and saved settings are fixed **synthetic view examples**, not
live API data. The native generation itself keeps the configured Contractor model
unchanged. Selecting an alternative model in the preview edits only a local draft.
Local Apply feedback says no Server request was sent. Theme choices are System,
Light, Dark and Black, scoped only to the preview document, without localStorage
or preference writes. The Git key textarea and Save button are disabled.

Normal, draft conflict, unconfigured catalog, reconnecting and failed read examples
keep their state distinctions. A failed read hides unknown Server forms and leaves
personal appearance available. Reset clears only its own stale draft. Closing the
preview discards local changes. No model, key, deployment or business data changes.

## Native contribution and assembly

`reference.html` is a focused Settings reference: exact Settings DOM, shared Icon
markup, global theme/font tokens and base styles, without unrelated tabs or the
host controller. `seed.html` is the full reviewed shell plus the same Settings DOM
and local controller. `inputs.json` hashes both and the unchanged prior tab files.
The native project receives only `reference.html` and `brief.md`, and generates
`opendesign.css` only. The host inserts the exact native style before `</head>`
in `seed.html`. Exact output, submitted reference, prompt, run identity, hashes,
full verified recovery manifests and preservation/cleanup proof are retained.
No host styling is represented as native output. `native-assembled.html` retains
that exact first assembly. Visual QA found that native 1100px rules restored two
personal columns on phones and compact two-column scheduling made captions too
narrow on desktop. The separate host `layout-fixes.css` restores a single personal
column below 640px, two readable theme swatches per row and full-width scheduling
lanes. `assembled.html` / `preview.html` add that independently hashed style after
the exact native CSS. No native file or reference was overwritten.

Other sections are served directly from the previous reviewed files. User
2026-11-03 additionally requests tighter Overview state/count pills. The separate
host `overview-legend.css` switches that grid to compact wrapping flex items with
4 px label/count and 4/8 px group gaps. Only this explicit Overview presentation
overlay is applied; its exact original native source stays unchanged. The wrapper
provides New / Previous Settings / Functional seed comparison. Independent source
hashes prove the prior Overview and multi-tab artifact were not modified.

## Local review

```sh
node docs/design/ui/experiments/operations-settings-opendesign-2026-11-03/serve.mjs 17570
node docs/design/ui/experiments/operations-settings-opendesign-2026-11-03/verify.mjs
```

Preview starts at
<http://127.0.0.1:17570/?section=settings&version=new&case=normal&theme=dark&width=1440>.
The wrapper supports all sections and 1440/1100/390/320px or Live window. Child CSP
keeps connect-src and form-action none; sandbox permits scripts and local form
events but no external connections or form navigation. Server accepts GET only.

This branch contains design artifacts. Production transfer, complete settings
integration and a main merge remain subsequent work after user feedback.


## Generation and verification

Native project `contractor-operations-settings-20261103`, conversation
`1f0ee118-4804-447a-8d1e-db84616609cc`, kept the configured Contractor model.
Initial run `d5e8163f-843d-4919-bd1d-d392aae1427b` reported succeeded but produced
no requested CSS. One deliberate continuation in the same 24-request relay
budget read a line-broken structural reference and produced the exact CSS:
`8986a7ac-9468-481b-b1c5-1df77924190a` succeeded. Both inputs and prompts are
retained. The relay used 10 requests, then stopped with its upstream credential
removed. No model substitution or extra model budget was used.

Full before/after recovery archives passed independent SQLite integrity checks.
All 198 prior native project files and all previous Project/Conversation/Message
rows remain identical. Eleven new project files were added. Provenance records
the native output, both run outcomes, reference/seed/source/CSS/assembly hashes,
separate host layout fixes and recovery manifests.

The browser gate passes 60 Settings theme/width/state layouts plus 12 compact
Overview legend layouts. It checks independent drafts/saves/reset/conflicts,
role overrides, whole-number scheduling bounds, disabled secret entry, System
and resolved theme choices, previous/new comparison, unchanged-tab navigation,
keyboard disclosures and live resizing. Personal cards must be one column on
phones. Overview preserves the exact five source counts/colors with a 4px
label/count gap and 4/8px group gaps. No overflow, page errors or production/
external requests. Selected screenshots and all Settings metrics are in `review/`.
Task/spec validation and `git diff --check` pass. No production UI gate was run
because production source code is unchanged.
