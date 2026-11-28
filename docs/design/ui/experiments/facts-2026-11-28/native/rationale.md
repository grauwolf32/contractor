# facts.css — rationale

## Problem

At 390×844 the first fact observation started at y≈818, buried under seven issue status chips, a disabled severity field, and prose. The reader experienced the Facts workspace as issue triage, not as a record of verified project facts.

## Direction

A quiet **evidence ledger**. Facts read as ledger entries — hairline-separated typographic rows, no chips, no severity, no badges. `--accent` is reserved for focus rings, links, and the quoted-evidence rule; it never signals status or confidence.

## Decisions

- **All values from shared tokens.** Colors, spacing, and radii resolve to `--surface`, `--surface-subtle`, `--line`, `--text`, `.muted`, `--accent`, `--space-2/3/4`, `--radius-lg`. No hardcoded values, so dark, light, and black themes all hold. Fonts are inherited, not redeclared.
- **Compact toolbar.** One sticky band (~56px): search, exactly one Filters disclosure (`details`/`summary`, panel anchored under the summary), and the "n facts · m records" count right-aligned. This keeps the first whole `.fact-row` fully above the fold at 390×844.
- **Rows.** Title first, optional subject as quiet secondary text, date as muted tabular numerals. From 720px the row becomes a two-column grid with the date right-aligned on the baseline; 1200px adds calmer padding for 1440px.
- **Reader.** "Back to facts" link (`.fact-back`) at top, then `.fact-observation` as the dominant text (1.6 line-height, 72ch measure), `.fact-source` showing the named source with the exact evidence as a visibly quoted block — subtle ground, accent left rule, nothing hidden behind a link. Optional `.fact-limitations` is a subdued aside; `.fact-assessment` is clearly secondary small muted text under a hairline.
- **No invented signals.** No status chips, severity fields, or confidence badges — the design adds no ungrounded markers.
- **Accessibility.** `:focus-visible` accent outlines throughout, `prefers-reduced-motion` guard, hover/focus tints via `--surface-subtle` paired with visible focus states so color alone never carries meaning.

## Assumptions

- Each row renders a title element, optional subject paragraph, and a date styled as `<time>` (or a `.fact-date` span).
- Filters is a single `details`/`summary` disclosure; the count is an `<output>` or a `.fact-count` span.
- The back link uses one extra hook, `.fact-back`, documented here.
- `facts.css` is loaded only on the Facts list and reader views, so it cannot leak into issue-triage screens.
