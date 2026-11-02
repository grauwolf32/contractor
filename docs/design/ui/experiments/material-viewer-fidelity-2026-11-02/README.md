# Material reader fidelity review

V429-019 corrects the incomplete UI 0.11.1 transfer. The source and OpenAPI
screens were compared at the same 1440/390px viewport, dark theme, saved
fixture and selected operation. `index.html` displays the actual pairs.

The reference captures come directly from the preserved V429-017
`material-formats-opendesign-2026-11-01/preview.html` through its isolated
server. Actual captures use the real React routes with the same seven-format
read-only API fixtures. The reference source, model output, conversations,
backups and previous transfer evidence were not modified. No new model call
was made and no business data was changed.

## Reviewed differences corrected

- Bare file name with exact namespace/name accessible label and exact copy;
  current/revision/lock chips and shared fonts remain.
- Real line numbers, including blank/wrapped lines, with unchanged literal
  source and CRLF/LF. Above 4000 lines, complete text remains in one node.
- Saved in / Project (real cached detail), Created and Format · Size, followed
  by independent lazy Versions and Technical details/Lineage.
- API intro, coloured methods, selected operation and schema browser. On
  phones Browse API collapses above the current operation.
- Existing Scalar handles parameters, responses, references and Swagger/OpenAPI;
  all original components/path-level parameters stay available. References to
  other paths keep the complete original document, with selected operations
  exposed through Scalar's public slug generator. Search and 50-row local pages
  cover realistic inventories within the existing one MiB input limit.

## Deliberate differences

The product-wide Contractor shell, return context, Rendered/Source terminology,
locale-aware dates, refresh actions and actual metadata remain. A Project file
has no invented “Source checklist” provenance. Scalar retains its rich response
and schema controls, field ordering and examples; these differ from the
prototype's simplified HTML tables. They share Contractor's fonts and tokens.
OpenAPI initially selects the first saved operation; paired captures explicitly
select Get an order, matching the reference's focused issue example.

## Reproduce and inspect

- Actual UI: `http://127.0.0.1:17560/test-results/material-production/index.html`
- Reference: `http://127.0.0.1:17565/?version=new&case=openapi&theme=dark&width=1440`
- `capture-metrics.json` records width/no-overflow and shared sans font checks.
- `review.json` records the completed verification and capture hashes.

Production browser tests cover seven formats, three themes, 320/390/1100/1440px,
User/Project/Run/Eval context, revisions, exact copy/download, read-only archive,
Swagger path-level parameters/definitions, local path references and navigation
of a roughly 6800-operation specification. Existing tests cover upload conflicts,
strict scope/revision ownership, explicit reads and large-file/renderer limits.
