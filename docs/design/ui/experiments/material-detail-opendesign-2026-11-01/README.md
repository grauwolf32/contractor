# Material / evidence detail — OpenDesign exploration

Non-normative design candidate for review before a production UI change. The
Artifact contract and its production renderers remain authoritative. This task
does not change a production route, API, store, model selection or business data.

From the repository root, using Node 24 and the installed UI dependencies:

```sh
node docs/design/ui/experiments/material-detail-opendesign-2026-11-01/serve.mjs 17564
node docs/design/ui/experiments/material-detail-opendesign-2026-11-01/verify.mjs
```

Open <http://127.0.0.1:17564/?case=source&theme=dark&width=1440>.
The review toolbar is outside the product frame. It selects four file formats,
current/historical and failure examples, three themes and four real iframe
widths. **Current UI** is an inert snapshot of the actual React route, captured
with these same synthetic fixtures after explicitly loading its preview.

`seed.html` is the assistant-authored working harness supplied to OpenDesign.
`original.html` preserves the generated HTML unchanged; `preview.html` contains
the reviewed candidate and any separately recorded corrections. The generation
brief, model contributions, run identities, hashes, limitations and preservation
receipts are recorded in `review.json` and the accompanying source files.

`examples.json` contains only invented data. Its selected HTTP exchange passes
the existing production parser; the demo Authorization and Cookie values are
fake. Sensitive header values start masked and reset when changing revision.
Downloads contain the selected synthetic revision's bytes. The binary example
allocates four MiB of zero bytes only on explicit local download. Upload and
return-context interactions announce their local intent without sending a write.

The server binds only to loopback, serves fixed paths, rejects mutations and
places the HTML in an opaque sandbox with `connect-src 'none'`. The candidate
has no external scripts, assets, fonts, API calls or storage. The configured
Contractor model is available to OpenDesign only during generation through the
bounded temporary relay; it is not available to the browser preview.

This is a presentation exploration over four formats and one selected HTTP
exchange. A production transfer must reuse the owning preview/parser components,
explicit load/size policy, authorization, scope/revision identity, paged history
and lineage. The small local Markdown renderer, seeded history and binary fixture
are demonstration code, not proposed replacement implementations.
