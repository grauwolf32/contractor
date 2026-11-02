# Review v4 — composition rationale

Baseline: `reference/review-layout-v3-verified.html` (content, truthfulness rules, IDs, data-action hooks, keyboard and mobile behavior preserved). New file: `index-review-v4.html`.

## Problem

v3 solved width but read as a stack of interchangeable boxed cards — same caps label + bordered rectangle for every paragraph, little variation in role or density.

## What changed

1. **Unboxed narrative lede.** The proposal ("What the AI found") and Impact are no longer cards. The finding sits directly on the canvas as the largest text in the pane (1.25rem), with a warning-toned icon badge, an "automated proposal" provenance line, and Impact set as a danger-edged block beneath it. Risk reads by tone and hierarchy, not by another box.
2. **Endpoint strip.** A quiet `GET /orders/{id} · Shop API · Source checklist` strip under the title replaces the subject code buried at the end of the status line. First viewport now answers "what may be wrong, where" before any scroll.
3. **Evidence workbench.** Source location, code and the retained artifact are one coherent unit: a sunken file toolbar (file icon, `src/routes/orders.ts`, lines 41–49, revision + copy), a code block with a real line-number gutter (41–44, labeled as excerpt of the stated 41–49 range), an explicit source-evidence-only note, then the `order-handler@evidence-r1` file row with Preview / icon-only Copy / icon-only Download / details.
4. **Quiet supporting column.** Verification, Limitations/preconditions and Technical details moved to a hairline-separated right column (300px) — smaller scale, no borders around every block, but limitations remain fully visible, not buried.
5. **Compact decision composer.** The surface starts as a one-row action bar under a "Your review" kicker; the draft (severity / duplicate target / Why) reveals only after an action is chosen. Choose-then-record stays two steps; validation messages updated to guide that flow.
6. **Scannable list.** Rows now show title, a "Source checklist · Shop API" context line, and compact status icons (state icon + severity word + verification icon) instead of three pill chips per row.
7. **Tab treatment** changed from filled chips to understated underline tabs spanning the header; the shared tokens, font stacks, 76px labelled rail, theme palettes and QA bar are unchanged.

## Kept

- All IDs, `data-verdict` / `data-sev` / `data-dup` / `data-copy` / `data-preview` / `data-download` hooks, J/K, Prev/Next bounds, Back to possible issues, URL-less local tabs.
- Truthfulness: proposal / AI-suggested severity / verification / human decision separate; static source ≠ HTTP exchange; no HTTP captured stated plainly; demo recording says "Demo only — no decision was sent"; mandatory Why, severity for Confirm, duplicate target; stale banner and refresh paths; no Reopen for Proposed; list counts are displayed-list-only.
- (Fixed a malformed `data-preview alias` attribute carried in the baseline's Evidence tab button.)

No external assets, fonts, network calls or packages. Not browser-verified; the human assistant will inspect.
