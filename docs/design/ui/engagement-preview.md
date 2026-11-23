# Saved engagement preview

The production Material reader shows exact saved engagement policies as Egress,
Identities and Limits, with literal Source retained. Behavior is owned by
[UI-R237](../../spec/ui/ui.md#ui-r237); it does not add a policy API or execution
authority. Add material uses the same ordinary upload dialog as the other kinds.

The composition reuses the previously reviewed OpenDesign engagement design:
`contractor-engagement-design-20261106`. Its original CSS, brief and inputs remain
in the common external experiment archive at
`~/Documents/ChatGPT/AppSec/archives/contractor-ui-experiments-2026-11-06/experiments.tar.gz`,
under `docs/design/ui/experiments/engagement-editor-opendesign-2026-11-06/`.
The native CSS SHA-256 is
`829ecd70f317182a4785ffd06d6d2cfd0add86ee462dbe8f415c2879ac2892ab`.
This is a production read-only adaptation of that existing design, not another
native generation. Shared controls, exact saved values, bounded navigation and
phone behavior replace editor actions and local draft fixtures. Shared fonts and
theme tokens preserve consistency with sitemap and the other readers.

The shared engagement JSON Schema validates presentation shape and versions.
Its existing source remains authoritative for enums and array bounds. Server
and Runtime validators continue to own canonical network syntax, identity
cross-field rules and actual execution admission. Preview grants no permission,
reads no credentials and contacts no target. In particular, absent HTTP limits
remain Not set rather than an invented default or an unlimited-execution claim.

The synthetic production-component host is
`/test-results/engagement-preview/index.html` on the local Vite server. It offers
light/dark/black, desktop/phone widths and configured/partial/absent/unsupported
policy examples. The new host, fixture, production source snapshot and review
evidence are preserved beside the earlier native originals in the common archive,
under `docs/design/ui/experiments/engagement-material-preview-2026-11-07/`.
Restore its `host-preview/*` into `ui/test-results/engagement-preview/` in a
checkout containing this reader, then run the normal UI Vite dev server.

`ui/e2e/engagement-preview.spec.ts` tests the actual Add material and exact
Material routes, including all four identity kinds, saved method subsets,
optional limits, malformed/oversized content, maximum route/nested collections,
long origins and keyboard geometry. Unit tests exercise ordinary upload
publication and shared saved-document examples.
