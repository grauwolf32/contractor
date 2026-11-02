# Material viewer — OpenAPI, LikeC4 and archive exploration

Non-normative extension of V429-016 for review before production transfer. Run
from the repository root with Node 24 and installed UI dependencies:

```sh
node docs/design/ui/experiments/material-formats-opendesign-2026-11-01/serve.mjs 17565
node docs/design/ui/experiments/material-formats-opendesign-2026-11-01/verify.mjs
```

Open <http://127.0.0.1:17565/?case=openapi&theme=dark&width=1440>.
The external review toolbar selects seven formats, themes, widths and states.
The previous candidate remains available, with its original server at port17564.
Current UI is an inert capture of the actual React route; diagram geometry in
that static reference was captured at desktop width and does not recalculate.

The new candidate adds operation/schema navigation for a synthetic OpenAPI
contract, two authored diagram views with local zoom/pan, and a ZIP file browser
with Markdown, source, escaped HTML and binary cases. Each download contains the
exact saved synthetic source or the actual ZIP bytes. Nothing calls an API,
executes a request, installs a Skill or extracts an archive on the user's machine.

`seed-submitted.html` preserves the assistant-authored input actually given to
OpenDesign; `original.html` is the unchanged generated HTML with its CSS-only
composition pass. `seed.html` and `preview.html` include separately recorded
review corrections. Original prompts, CSS, rationale, hashes, run identity,
preservation receipts and the division of contributions are in `review.json`.

The source `.c4` was parsed and laid out by the installed production LikeC4
browser engine, with the source/layout hashes pinned by `diagram-provenance.json`.
The local SVG uses those nodes, relations and positions; the overview uses the
engine's vertical landscape layout on phone. This is a presentation harness,
not a replacement LikeC4 implementation. The OpenAPI example passes the existing
production parser. `source-bundle.zip` matches every entry in `examples.json`.

The review server binds only to loopback, serves fixed paths, rejects mutations,
and applies an opaque browser sandbox with no outbound connections. Shared fonts
and theme tokens remain unchanged. Model access exists only during generation
through the bounded temporary relay. Earlier projects/files are preserved and
independently checked in recoverable state/sandbox backups.

Production transfer must reuse the existing OpenAPI/LikeC4/archive components,
explicit load/size policy, revision/scope authorization, limits and errors. The
seeded controls, simplified SVG styling and local archive navigation exist only
to review composition. The new `invalid` state simulates a renderer failure over
a retained fixture; it does not invent another invalid file revision.
