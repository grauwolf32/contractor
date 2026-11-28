# Contractor Materials and New check material chooser

Design an iteration of an existing professional AppSec workspace. The review uses synthetic Invoice Lab materials. Existing production screenshots, revisions and the existing OpenDesign project history must be preserved.

The host has built an interactive scaffold. Deliver an additive opendesign.css and rationale.md. The host will apply your CSS directly to the scaffold and verify the two user journeys. You own the visual refinement and responsive composition. Make meaningful changes to layout, hierarchy, material identity and the selection experience.

Read only reference/brief.md, reference/layout.css, reference/theme.css and reference/screen-map.html. Do not use Skills or plugins, make network calls, install dependencies or run browser tools. Write only opendesign.css and rationale.md at this project's root. Stop after writing both files. All references are synthetic and self-contained.

## Existing Contractor visual language

Use the provided system sans-serif and monospace font tokens exactly. Use only the shared light/dark/black color tokens. The application has a compact 76px global navigation rail, a contextual project list and a Project header. Keep the global navigation and Project tabs recognizable. This is a dense reading and analysis tool. There are no illustrations, decorative badges, ornamental numbers or gradients. Icons come from the existing stroke-icon system. Color may help distinguish source, API and Engagement materials while text remains sufficient.

## Materials journey

The Project Materials page combines a searchable grouped list and an adjacent selected-material preview. Filtering by category must be discoverable. Show clean rows whose hierarchy is material name, format/size/update time and optional Git commit. The full namespace belongs in context rather than competing with the name. The right preview exposes exact revision, Current state, copy and download actions. Opening a material immediately shows its saved content. A compact details arrow lives below the content divider.

Work over the .library, .library-main, .filter-chips, .group, .group-title, .material-rows, .material-row, .row-main, .row-title, .row-meta, .kind-icon, .reader, .reader-head, .reader-title, .reader-meta, .reader-body, .reader-tabs, .reader-footer, .endpoint and .file-tree classes. Improve row selection, alignment, column balance and negative space. Keep long names and revisions readable.

At 390px show the list first. Selecting a row opens the preview with a clear Back to materials action. Preserve the filtered list when returning. The mobile header should leave room for actual materials.

## New check journey

The objective precedes the workbench. The left list selects Source review, API review or Active HTTP checks. The middle contains Materials and Scope. The compact right column contains time limit and Start/Save actions. Inputs show the chosen material name and exact revision with Preview and Change actions.

Change opens .picker-dialog. The header names the input, then .picker-grid contains .picker-list with search and compatible materials, plus .picker-preview showing the currently focused candidate. Selecting a candidate previews it without attaching it. The user confirms using Attach material in the .dialog-footer. The selected file and revision remain clear. Cancel preserves the earlier attachment. Add material is available inside this chooser. The check objective, scope and time limit remain in the underlying form.

At 390px the chooser becomes full-screen. List and preview switch within it. A short footer contains Cancel and Attach material. Avoid a large floating region that covers content.

Improve .card, .card-head, .slot, .slot-label, .attached, .slot-actions, .slot-empty, .check-type, .summary and the picker classes while retaining clear controls.

## Deliverables

Write 140–240 readable CSS lines in opendesign.css. Use existing color and typography variables, explicit responsive rules and real selectors from the scaffold. Avoid changing the global font scale. Then write rationale.md explaining 5–8 concrete design choices and any mobile treatment. All explanatory prose must use direct, plain language. Avoid contrast slogans, semicolons and decorative headings. Do not claim that synthetic formats prove actual input validity.
