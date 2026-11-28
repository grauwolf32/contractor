# Materials and New check material chooser

Interactive OpenDesign iteration for Contractor, created on 2026-11-28 in
`codex/materials-opendesign-20261128` from main `7e46ebca5`.

Open http://127.0.0.1:17574/ while the review server is running. The toolbar
switches between Project materials and New check, three themes, and 1440,
1024 and 390px viewports. Wide viewports scale to fit the host panel while
retaining their selected layout width.

Restart from this experiment directory with
`python3 -m http.server 17574 --bind 127.0.0.1`.

## Journeys

Materials combines search, category filters and a selected material reader.
Rows expose format, size, time and the Git commit when available. The reader
opens immediately, shows the exact revision, supports Raw and details, and
provides copy and download actions. Secondary categories live in More. On a
phone, selecting a material opens its reader and Back restores the search.

New check opens a material chooser over the current setup. Search reaches the
compatible synthetic candidates. Selecting a candidate opens its preview.
Attach material confirms the selection. Cancel retains the earlier attachment.
Add material opens an upload dialog inside the chooser and Add and attach
returns to the setup with the objective, scope and time limit retained.
The phone chooser uses the full screen with a compact confirmation footer.

The prototype uses synthetic Invoice Lab data in browser memory. Uploads stay
in that memory, Start and Save as draft show demo outcomes, and downloads
contain fixture text. Reloading restores the supplied examples.

## OpenDesign provenance

`native/opendesign.css` and `native/rationale.md` are the exact native outputs.
`native/inputs/` retains the four inputs read by the generation. Creation,
submission and terminal run metadata are retained beside them. The run is
`c3729282-8627-43e9-998e-55fcf3364b41` and completed successfully through the
configured Contractor gateway and worker model.

The generation ran in an isolated copy of the saved OpenDesign state. A
read-only container root and internal network bounded its environment.
`preservation.json` records the private complete state backups and verifies
that 278 old project files and the prior Project, conversation and message
keys remain present. The temporary model relay was stopped and its upstream
credential file removed after generation.

`reference/` contains the functional host scaffold. `host-polish.css` repairs
closed mobile dialog visibility, long-reference wrapping, mobile check-type
selection and attachment density. The host also puts secondary categories
in More, keeps a preview within its filtered selection, preserves form state
through uploads and shows exact material details.

The common archive copy lives under
`/Users/ruslan/Documents/ChatGPT/AppSec/archives/contractor-ui-experiments-2026-11-06/iterations/2026-11-28-materials-picker`.

## Verification

Fifteen Chromium scenarios cover three themes and desktop, medium and phone
widths. The checks cover overflow, search, categories, direct preview, cancel,
exact revision confirmation, upload validation, retained setup state, required
Engagement, copy, Raw, details, empty/error/loading states and the review host.
`verification.json` records the result. `review/` retains captures of the
scaffold and the applied OpenDesign iteration.

To repeat the browser checks from this directory with UI dependencies installed,
run `node ../../../../../ui/node_modules/@playwright/test/cli.js test --config playwright.config.mjs`.
Set `MATERIALS_PREVIEW_URL` when serving the prototype on another local port.
