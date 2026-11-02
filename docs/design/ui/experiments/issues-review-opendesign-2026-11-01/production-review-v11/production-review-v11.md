# Production review v11 — React transfer, issue detail screen

**Inputs:** `reference/production-review-20261101/review-brief-compact.txt` only (real detail DOM, current issue CSS, accepted v10 CSS with shared tokens, measured geometry at 390px black / 1440px). Judged against the accepted OpenDesign design, not a new product. All content is synthetic; no snippets or human decisions were invented.

## Measured baseline (from the brief)

| Selector | 390px (measured) | 1440px (measured) | Accepted target |
|---|---|---|---|
| `.issues-workspace-title` | 16px / 650, 2 lines (h 43.2) | 22px / 650 | 16px on phones, 1.375rem desktop, weight 650 ✓ |
| `.issues-section-label` | 11px / 500 caps | 11px / 500 | `--text-2xs` / 500 ✓ |
| `.issues-narrative` text | 14px / 400 | 20px / 400 | `--text-md` mobile, `--text-lg` desktop ✓ |
| `.issues-detail-bar` | top 88, **h 267.8** | top 57, h 175.2 | compact, static ≤820 |
| `.ui-detail-pane-body` | top 355.8, h 1268.6, 24px block padding | h 682.3 | tighter block padding on phones |
| `.ui-detail-pane-footer` | static, h 163.5 | sticky, h 85.5 | static ≤820 ✓, pinned at desktop ✓ |

The type scale, column order and flow model largely survived the transfer. The visible breakage is concentrated in mobile chrome density, the evidence card, and the composer wrap.

## Findings and fixes

### F1 — Mobile header is heavy: 267.8px of chrome before content
**Cause:** every stacked row in `.issues-detail-bar` carries its own `--space-2` vertical margin (title `margin: var(--space-2) 0`, nav `margin-block-start: var(--space-2)`, identity `margin-block: var(--space-2)`) on top of the bar's `--space-3` padding; the AI-suggestion line additionally wraps as a ragged third chip row at 390px.
**Fix (v11 §1):** title margins `--space-2 0` → `--space-1-5` with the 16px / 650 / 1.35 title explicitly locked; nav margin → 6px; identity margins/gap tightened; `.issues-ai-suggestion` gets `flex-basis: 100%` so chips stay on one clean row and the suggestion reads as its own line; pane-body block padding 24px → 16px; workbench gap 20px → 12px. Expected ≈ 15–25px less header chrome and ≈ 16px less body padding, derived from the token math against the measured baseline (not re-measured in a browser).
**Before → after:** eight loosely spaced header rows → same rows with 6–8px rhythm; ragged chip wrap → chips row + full-width suggestion line.

### F2 — Source snippets hidden behind Preview (regression vs accepted)
**Cause:** in the transferred DOM each `.issues-evidence-file` renders only the toolbar — a Preview button with `aria-expanded="false"`, the copy chip, download, and the details link. No snippet node exists until Preview is used (`issues-evidence-preview` has zero occurrences in the DOM). The accepted workbench always rendered the code region (`.preview` → `pre.code`, mono on `--code-bg`).
**Fix (v11 §3):** accepted code treatment for the restored region — `.issues-evidence-preview` / `.artifact-preview-panel` as a sunken, divider-separated strip with `pre` in `--font-mono` at `--text-xs`, `--code-ink` on `--code-bg`, 24rem scroll cap, `pre-wrap` + `overflow-wrap: anywhere`. A clearly-marked stopgap un-hides a `[hidden]` snippet region so the composition shows snippets without waiting on the toggle.
**Component change (coding assistant):** render the bounded retained text snippets expanded by default inside the evidence card; the Preview button may remain as an affordance. Once that lands, delete the stopgap rule. No source text is fabricated by this CSS.
**Before → after:** snippet invisible until Preview → snippet visible by default with Preview/copy/download still working.

### F3 — "details" link sits inline with the actions instead of below the card divider
**Cause:** the details anchor is the last child of `.issues-evidence-actions` inside `.issues-evidence-toolbar`, so it renders in the same cluster as Preview/copy/download.
**Fix (v11 §2):** `.issues-evidence-file` becomes a flex column and `.issues-evidence-actions` is lifted with `display: contents`, so the anchor (matched via `a[aria-label^="Details for"]`) wraps to a full-width row at the toolbar's bottom edge with the card divider — edge-to-edge via symmetric negative margins matching the toolbar's `--space-4` inline padding. Preview/copy/download stay above the divider.
**Recommended DOM change:** move the anchor out of the actions cluster to below the preview region; the CSS keeps working either way. Browsers without `display: contents` degrade to the link inline (current behavior).
**Before → after:** `[Preview] [copy] [download] details` inline → actions row, divider, details row.

### F4 — Composer wraps raggedly on phones
**Cause:** five buttons plus `.ui-decision-spacer` wrap at natural widths and the `kbd` hints widen every button at 390px (footer measured 163.5px with ragged rows).
**Fix (v11 §1):** buttons stretch with `flex: 1 1 auto` (mirrors the accepted ≤520 rule), spacer and kbd hints hidden ≤480. All five actions (Confirm, Not an issue, Needs evidence, Mark as duplicate, Next item), their shortcuts, `data-defer-draft` and the static normal-flow footer are kept — the static ≤820 footer already ships in the current CSS and is not re-declared.
**Before → after:** ragged wrap with visible kbd chips → filled rows (2+2+1), identical capabilities.

### F5 — Supporting column order: verified correct, preserved
DOM order in `.issues-summary-side` is Impact (danger label "If real, the impact") → Verification (info) → Limitations (warning) → Duplicates → Sources → Technical details. Impact already precedes Verification/Limitations, and Duplicates is an explicit `<details>`. v11 makes no reorder and adds no boxes.

### F6 — Desktop: unboxed narrative + evidence workbench preserved
The summary panels use no `.issues-card` wrapper in this DOM: the narrative is the unboxed lede (`.issues-narrative` + proposal icon) and the only carded units are the `.issues-evidence-file` cards inside the workbench. The existing `>1100px` grid (`minmax(0,1fr) 18.75rem`) already matches the accepted 300px side column. v11 adds no card chrome at any width, so no repetitive identical-card stack appears.

### F7 — Type scale: verified conformant, guarded
Both metric sets match the accepted scale (table above). v11 re-asserts only the mobile title lock (16px / 650 / 1.35); no other font, weight, or global typography changes.

## Constraints honored
- Shared tokens only (`--space-*`, `--text-*`, `--line`, `--code-*`, `--surface*`, `--font-mono`); no new colors or fonts.
- Breakpoints 480 / 820 / 1100 only; v11 adds rules only inside ≤480 plus width-independent evidence-card rules.
- Every capability kept: Prev/Next with count on one row and View-in-its-check on the next (existing ≤480 grid preserved), tabs, Preview toggle, copy chip, download, details link, Duplicate, decisions with deferred draft, keyboard shortcuts, truthful status chips and AI-suggestion wording.
- All selectors are `.issues-workspace`-scoped: other routes and global fonts untouched. All previous files and conversations preserved; no network, installs, or unrelated commands.

## How to preview
Open `production-mobile-review-v11.html`. It loads the real rendered React page (`reference/production-review-20261101/react-before-390.html`, which carries its actual applied CSS) in a frame with 390 / 820 / 1440 width buttons and a Before/After toggle that attaches `production-react-fixes-v11.css` to the frame document. Browsers block script access into `file://` frames: opened straight from disk the frame shows the unpatched reference; serve the folder over any local static server (e.g. `python3 -m http.server`) or open the page in Firefox to see the fixes applied.

## Residual risks
- The snippet stopgap assumes the restored region uses the existing `.issues-evidence-preview` / `.artifact-preview-panel` hooks; delete the rule once snippets render expanded by default.
- The pixel-saving estimates in F1/F4 are derived from token arithmetic against the measured baseline, not from a fresh browser measurement.
