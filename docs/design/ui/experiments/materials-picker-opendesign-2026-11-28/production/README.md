# Working UI transfer

Implemented the approved OpenDesign Materials and New check iteration in
`codex/materials-opendesign-20261128`, integrated with main `b708454f7`.
The native generation and its inputs remain byte-identical in `../native/`.

Interactive review:
http://127.0.0.1:17575/test-results/materials-production/index.html?screen=materials&version=production&theme=dark&width=1440

The review host switches between Materials, a material page and New check,
with three themes and four widths. Version switches to the approved design.
It mounts the real application routes and components with a synthetic,
contract-checked API adapter. Preview uploads live in browser memory.
The adapter and its fixtures are test-only and are absent from the UI build.
The working UI uses the existing authenticated API, upload, download and
check-launch operations without a new backend endpoint.

## Fidelity and behavior

- Full-width grouped material rows, category-tinted shared icons, search,
  primary category chips and a More selector. Exact links lead to the material
  page with its bounded preview already loaded. Return context includes the
  namespace, page, search, category and reading position. Page-local search and counts explicitly
  describe their scope when the inventory has more pages.
- The Project/Eval material reader uses the content width. Copy, download,
  versions and refresh remain available. Facts and eligible version uploads
  follow the content. Automatic archives open SKILL.md, root README.md or the
  first readable file. User/Run file pages retain explicit loading.
- New check uses attachment cards and a searchable preview-and-confirm chooser.
  Clicking a candidate is tentative. Attach commits the exact revision; Cancel
  preserves the earlier choice. Small screens use a full-height chooser with
  Back to choices and a fixed confirmation footer.
- Add and attach reuses the abortable upload form with the input's accepted
  formats, then reads the exact saved metadata before attaching it. Failure
  retries the read, never the completed write. Closing the chooser ignores a
  late result. The objective, time limit and other setup state stay mounted.
- The current matching rules, bounded inventory pagination, optional inputs,
  custom checklist, web-scan setup and launch validation remain the owners of
  real behavior. Synthetic defaults from the design are not substituted for
  the actual profile or inventory.
- Existing Markdown, Scalar OpenAPI, source, diff, archive, Engagement, Sitemap
  and LikeC4 engines remain intact. Their native controls and size policies are
  retained, including on-request rendering for large OpenAPI documents. Fonts
  and all themes use the project-wide tokens.

## Verification

- All 233 unit-test files pass: 2,637 passed and one expected failure.
- 68 Chromium scenarios pass against the production build. They cover the new
  journeys, retained draft state and inline upload, the material engines,
  exact downloads, history and source wrapping, responsive layouts through
  320 px, Engagement/Sitemap bounds and Project context navigation.
- The live comparison host is separately checked for navigation, switching to
  the approved design, 390 px and opening the real chooser.
- TypeScript for source and browser tests, ESLint, Prettier, build, CSS/export
  gates and the specification consistency check are recorded in verification.json.

The production archive is
`/Users/ruslan/Documents/ChatGPT/AppSec/archives/contractor-ui-experiments-2026-11-06/iterations/2026-11-28-materials-picker-production`.
It retains changed source files, the review launcher, screenshots and the
approved native design inputs/outputs. Both earlier Materials archive copies
remain intact.

The UI changes are implemented on the branch. Main and the separate demo
installation have not been changed by this transfer.
