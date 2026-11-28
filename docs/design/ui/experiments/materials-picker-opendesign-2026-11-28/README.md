# Materials and New check material chooser

Interactive OpenDesign iteration for Contractor, created on 2026-11-28 in
`codex/materials-opendesign-20261128` from main `7e46ebca5`.

The approved iteration is now implemented in the working UI. See
[the transfer and verification](production/README.md) and open
http://127.0.0.1:17575/test-results/materials-production/index.html?screen=materials&version=production&theme=dark&width=1440
for the interactive production-component comparison.

Open http://127.0.0.1:17574/ while the review server is running. The toolbar
switches between Project materials, Material preview and New check, three themes, and 1440,
1024 and 390px viewports. Wide viewports scale to fit the host panel while
retaining their selected layout width.

Restart from this experiment directory with
`python3 -m http.server 17574 --bind 127.0.0.1`.

## Journeys

Materials combines search, category filters and full-width rows. Rows expose
format, size, time and the Git commit when available. Clicking a material
navigates to a dedicated page with Preview already open at full content width.
The URL records the exact material and revision, supports reload and can be
opened in a new tab. The page supports Raw, details, copy and download.
Back to materials restores the list's search, category and scroll position.
Secondary categories live in More. This flow is the same on desktop and phone.

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
selection and attachment density. The host also puts secondary categories in
More, preserves form state through uploads and shows exact material details.
A follow-up requested during review replaces the library's narrow reader with
a dedicated material page. The New check chooser keeps its quick preview and
explicit attachment confirmation. The original native OpenDesign outputs
remain byte-identical.

The common archive copy lives under
`/Users/ruslan/Documents/ChatGPT/AppSec/archives/contractor-ui-experiments-2026-11-06/iterations/2026-11-28-materials-picker-full-view`.
The earlier `2026-11-28-materials-picker` archive copy remains intact.

## Verification

Twenty Chromium scenarios cover three themes and desktop, medium and phone
widths. The checks cover overflow, search, categories, full-page preloaded preview,
deep links and reload, Back/Forward navigation, chooser cancel,
exact revision confirmation, upload validation, retained setup state, required
Engagement, copy, Raw, details, empty/error/loading states and the review host.
`verification.json` records the result. `review/` retains captures of the
scaffold and the applied OpenDesign iteration.

To repeat the browser checks from this directory with UI dependencies installed,
run `node ../../../../../ui/node_modules/@playwright/test/cli.js test --config playwright.config.mjs`.
Set `MATERIALS_PREVIEW_URL` when serving the prototype on another local port.
