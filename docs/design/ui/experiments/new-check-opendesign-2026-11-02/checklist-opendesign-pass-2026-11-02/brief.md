# Contractor — focused checklist design refinement

The current candidate is functional and uses shared Contractor fonts/themes.
The user likes the overall New check design, but wants the custom checklist
editor more polished, composed and less repetitive. Make a real visual design
pass rather than renaming a few colors or producing a long discussion.

## Deliverables — new files in this new project only

1. checklist-polish-v1.css: about 90–150 readable lines of focused responsive
   CSS overrides, using only already declared theme/spacing/type tokens.
2. index-checklist-v1.html: use Python to read reference/checklist-seed.html,
   insert your CSS in a NEW style element immediately before </head>, then
   write the new full HTML. Preserve every original byte outside that insertion.
3. rationale-checklist-v1.md: 3–5 sentences explaining concrete visible changes.

Read reference/current-scope.css for the current editor structure. The full
seed includes existing theme tokens and scripts; use Python for assembly
without spending calls reading huge scripts. Optional reference screenshots:
desktop-empty.png, phone-empty.png, desktop-applied.png. Do not probe unavailable
image tools, run browser QA, start a server, install a dependency, or read any
other project or Skill. Produce the files promptly and stop.

## Visual goals

- An intentional authoring surface with a clear heading, Draft/Ready state,
  a readable editable item, quiet ordinal and discreet remove action.
- Avoid a card inside each card. Keep item separation light and align the
  baseline/spacing of heading, editor and footer. Add item is an obvious
  lightweight action; Use checklist is a deliberate secondary setup action.
  The final Start check remains the page's primary action in the summary.
- Make the editor and right-hand summary feel related without making every
  section look identical. Improve hierarchy and breathing room, avoiding huge
  empty padding. Do not hide facts or controls to make it fit.
- At 390px and 320px keep editor header, item, remove and footer actions cleanly
  composed; normal document flow, no sticky/floating footer. Large text and
  long entered statements must wrap and remain editable.
- Light/dark/black use the shared tokens. Improve focus/hover and disabled
  readability. Keep typography consistent with Contractor; no imported fonts.

## Non-negotiable behavior / content

- A NEW checklist begins with ONE EMPTY item. Never insert sample items.
  Existing selected material can contain 3 real fixture items; preserve it.
- Preserve every script, identifier, fixture and event handler. No JavaScript,
  new controls, invented parsed counts, form semantics, network calls or assets.
  Do not change Add/remove/apply, dirty/invalid readiness or profile selection.
- Source-only checks have no Scope section. ASVS full/pilot choices remain in
  Advanced options. API coverage comes from its exact selected OpenAPI fixture.
- A checklist's Scope appears after Apply; draft count remains in the summary.
  Engagement remains a read-only Scope example with Start disabled; it grants
  no new authority. Do not remove it or make it executable.
- Do not use CSS generated content for meaningful labels or hide controls,
  labels, material facts, Scope, errors, optional settings or empty guidance.
- Do not change font families, root tokens, .main/.workbench layout, global
  rail, objective card, fixture data or other screens. Use existing classes.

## Key selectors

.checklist-editor; .checklist-editor-head; .editor-state[data-tone];
.checklist-intro; .checklist-entries; .checklist-entry;
.checklist-ordinal; .checklist-entry-body; .checklist-entry-label (screen-reader
label: do not make it visible or remove it); .checklist-statement;
.checklist-remove; .checklist-editor-actions; #add-checklist-item;
#apply-checklist and #apply-checklist.is-applied; .checklist-feedback[data-tone].
.summary (the existing page summary; all its facts and actions stay visible).
Other controls may be laid out consistently, but scope this pass to the editor
and its relationship to that summary. Do not use animation as decoration.
