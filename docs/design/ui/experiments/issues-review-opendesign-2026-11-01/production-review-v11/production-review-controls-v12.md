# Production review — controls v12 (Why/Record composer + Duplicates)

Scope: the Issue detail workspace only. Inputs: `reference/production-review-20261101/controls-brief.txt` (duplicates.tsx, decision-bar.tsx, shared decision CSS, applicable issue CSS) — the only file read. Deliverable: `production-review-controls-v12.css`, a CSS-only override layer that assumes it loads after the existing issue stylesheet. No DOM, API, state, or route changes; v11 and all earlier files untouched. Shared tokens and fonts only; the only media breakpoint used is 480 (820/1100 need nothing for these two controls).

## 1. Why / Record decision (`ui/src/ui/decision-bar.tsx`)

**Diagnosis (from the brief's source + CSS).** The base `.ui-decision-compose` is a *wrapping flex row* (`display: flex; flex-wrap: wrap; align-items: flex-start`). Three consequences on a phone-width pane:

- The "Why" label sits inline to the **left** of the field, held level only by the `padding-top: 0.625rem` alignment hack on `.ui-decision-compose > .ui-decision-label` — a side-by-side arrangement, not a labeled input.
- The field is `flex: 1 1 16rem`, so it shrinks toward ~14rem once the label takes its share; the workspace's `min-height: 4rem` is the only breathing room.
- The Record button rides the same wrap row (`flex: 0 1 auto`). Below the 34rem container query it gets `margin-left: auto`, so the primary action lands wherever the wrap leaves it, visually detached from the reason it submits and no more prominent than the verdict buttons above it.

The mandatory-input help already exists and is already wired (`aria-describedby` → the standing rationale hint and the `ui-decision-hint` "Write a short reason." / "Choose severity." line) — it just reads as a detached status line under a cramped two-column block.

**Fix (all workspace-scoped):**

| Aspect | Before | After |
|---|---|---|
| Composition | wrapped row: label ∥ field ∥ button | column: label → field → button |
| Label | inline-left, alignment `padding-top: 0.625rem` | own line above the field, `--text-sm`, padding removed |
| Textarea | `flex: 1 1 16rem`, min-height 4rem | `flex: 0 0 auto; width: 100%`, min-height 4.5rem; `field-sizing: content` growth to 10rem and `resize: vertical` kept |
| Record button | in the wrap row, narrow-pane placement incidental | its own row beneath the field: `align-self: flex-end` (right-aligned) by default; `align-self: stretch; width: 100%` at ≤480 |
| Prominence | inherits base button sizing | `--text-sm`, weight 600, `--space-4` inline padding; keeps the base 2.375rem min-height and the ⌘/Ctrl+Enter kbd hint (the existing ≤34rem container rule already hides hints on narrow bars) |
| Focus | field's existing 2px `--focus` ring on `:focus-visible` | unchanged — a column layout keeps the full-width field under a clear ring |

**Preserved untouched:** severity radio row and its required/disabled behavior, verdict buttons with key hints, `more`/`extra` slots, the pending lock model (reason stays read-only, Record stays focusable but `aria-disabled`, focus stays in the bar), Ctrl/⌘+Enter submit, the missing-input hint chain, and the error alert. Recording still fires only when verdict + required severity + non-blank reason are all set.

## 2. Duplicates (`ui/src/routes/issues/duplicates.tsx`)

**Diagnosis.** The shipped issue CSS defines **no rules for any `.issues-duplicates*` class** — the disclosure renders with native markers, the list with default bullets and margins, and observation internals with raw defaults, so the section reads as foreign chrome next to the supporting column's quiet `--text-sm`/muted type. Separation also fails structurally: `PossibleMatches` is a *classless* `<details>` rendered directly after the matching-basis `TechnicalDetails`, so exact matches and possible matches run together with no divider or hierarchy. Inside, per-observation content (title link, "This observation"/state chips, Analyst/Severity, AI suggestion · Verification, Evidence link + provenance caption, nested Check context) is dense and unordered-feeling.

**Fix:**

- **Quiet disclosure matching supporting-column typography:** outer `Duplicates` summary at `--text-sm`/500/`--muted-strong` (the `.issues-side-section` register), nested summaries (`Check context`, `Possible matches`, `Search options`) at `--text-xs`/`--muted`; native markers replaced by a small CSS chevron (150ms ease-out) and an explicit `:focus-visible` ring keeps keyboard use visible.
- **Body rhythm:** `.issues-duplicates-body` becomes a flex column with a `--space-3` gap and normalized child margins; lead sentences ("Exact matches in this project…") and the three unavailable messages are readable `--text-sm`/`--muted-strong` instead of floating paragraphs.
- **Rows, not cards:** `.issues-duplicates-list` loses bullets; each `li` is a hairline-divider row (`border-block-start` on subsequent items, first item clean) — consistent with the desktop decision against a repetitive stack of identical cards.
- **Readable linked rows:** observation title (the `ContextLink` target, first child) at `--text-sm`/600 with `overflow-wrap: anywhere`; Analyst/Severity as `--text-xs` muted labels with 600 `--text` values; the AI-suggestion · Verification caption and the Evidence link + "In this check · Same/different source proposal" caption all retained, wrapped safely; Check context stays a nested quiet disclosure and its API read still happens only when opened (`enabled: open`).
- **Clear exact-vs-possible separation:** the Possible matches disclosure gets a top hairline divider plus inset spacing, and its summary uses the smaller nested style — it reads as a peer section, not a continuation of the exact list. Candidate rows keep their "Possible match" chip, reasons line (now `--text-xs` readable), truncation caption, and exact-reasons `TechnicalDetails`.
- **Code blocks:** the matching-basis, check-config, and reasons `<pre>` blocks get the accepted sunken mono treatment (`--sunken`, `--font-mono`, `--text-2xs`, 20rem max-height, scroll).
- **≤480:** body gap and row padding tighten one step for phone.

**No invented content:** no counts, no "winning" issue, no matching semantics, no new API reads or auto-fetches — every string on screen still comes from the existing DOM.

## DOM class additions

**None required.** All selectors reuse shipped classes. Two selectors lean on existing structure and are worth noting for maintainers:

- `.issues-duplicates-body > details:not(.issues-duplicate-options):not(.ui-tech)` targets the Possible matches disclosure, which ships classless; the `:not()` guards exclude the Search options disclosure and the `ui-tech` TechnicalDetails. If `TechnicalDetails` ever stops rendering the `ui-tech` class, the guard is harmless (it only excludes).
- `.issues-duplicates details > summary` targets the classless nested disclosures (Check context, Possible matches, Search options); the outer summary is addressed separately via `.issues-duplicates > summary`.

## Preserved behavior checklist

- Severity + rationale validation, pending focus/locks, actual writes, verdict/Next/Record shortcuts, and the ⌘/Ctrl+Enter hint on desktop — untouched (CSS only).
- Per-observation Evidence link, check membership/provenance caption, human analyst state, AI suggestion, verification line, and per-observation Check context (lazy API read on open) — all retained and visible.
- Pagination, stale-revision remount keys, partial/complete captions, search-work-limit and hints-unavailable notices — untouched.
- Exact matches and Possible matches remain separate data flows with their own disclosures; "no match is not proof of uniqueness" language untouched.

## Limits

- No browser QA was run or claimed. Changes are CSS-only against the brief's DOM and class list; layout outcomes at other pane widths follow from the same rules but were not measured.
- The chevron is decorative CSS; the summary remains the native disclosure control with its own accessible state.
- Only the 480 breakpoint is exercised; 820/1100 require no rules for these controls.
