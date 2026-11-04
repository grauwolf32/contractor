# Contractor Inbox — native design brief

The user approved Inbox as the next screen after Run, Project, Issues, Materials, New Check and Operations. Improve the real existing Inbox, not a separate marketing dashboard. The reference contains the actual current React DOM with synthetic public-API data and existing theme tokens. SVG contents are abbreviated; the actual app retains its shared icons.

Deliver one new `opendesign.css` with approximately 120–200 lines. Every selector must start with `.inbox-redesign` (a body class). The host will inject the exact CSS into the actual React Application; no DOM replacement, invented counts, generated content text or hidden facts. Preserve all previous projects and files. No Skills, dependencies, network requests, fonts, images, JavaScript, HTML assembly or browser QA. Read only `reference/inbox-reference.html` and this brief. Write the CSS promptly in one heredoc.

## Visual direction

Calm, crisp application UI with a stronger editorial hierarchy. The existing large empty Overview and four identical rows feel flat. Make its four existing overview actions a purposeful composition: Decide and Unblock are prominent, Ready and Running quieter; colored icon tiles, tasteful status borders and spacious counts. Use existing warning/danger/success/info tokens so light/dark/black work together. No gradients, new colors, oversized hero or decorative graphics. Keep the global font families and existing shared shell unchanged.

The left list should read as a compact triage queue: distinct section heading/count pills, selected row, readable project/time metadata, restrained color and explicit words. Keep every row and section, native scrolling, keyboard focus, SVG icons, progress segments and truthful empty/error notices. Do not hide sections or add a new navigation switch. A button with `aria-current=true` is selected. `.inbox-sections` holds four `.ui-list-section` children in order Decide, Unblock, Ready, Running. Heading: `.ui-list-section-heading`, `.ui-list-section-title`, `.ui-list-section-count`, `.ui-list-section-aside`. Rows: `.ui-row`, `.ui-row-control`, `.ui-row-glyph`, `.ui-row-main`, `.ui-row-title`, `.ui-row-meta`, `.ui-row-extra`.

Make selected details feel as considered as the overview. `.inbox-detail-bar` shows section, position, Open link, previous/next. `.ui-detail-pane-body` contains context, `.inbox-head` or finding summary, `.inbox-block`, facts, progress, result, technical details. Distinguish a model-recovery reason, result destination and progress with subtle surfaces, not a pile of identical cards. Keep the decision footer compact and the rationale/control ordering clear. Actual summary classes are included in the reference. All decisions/retries are intercepted by the synthetic transport: no live mutation.

## Layout invariants

Existing `.ui-panes` layout: above 1100px list360px plus flexible detail; 821–1100px list300px; <=820px list OR selected detail in normal page flow (`data-show=list|detail`). Preserve this behavior and `.ui-panes-back`. Do not override global shell, pane display/hiding, `[hidden]`, overflow with clipping, fixed positioning or viewport heights. You may improve inner spacing and overview grid. At 390 and 320px wrap actions, compact detail bar and padding; controls stay reachable. Do not change typography families or global fonts. Use existing token names defined in the reference.

## Product truth

Four sections, ordered Decide / Unblock / Ready / Running. Possible issue is a proposal, analyst verdict is separate; execution success does not establish result quality. Pending active-test/report reviews are current product behavior. Preserve unknown versus zero progress and all limits/errors. No bulk approve, mark-all-read, new workflow states or new permission logic.

## Examples in reference

Overview; possible issue and human assessment; active-test review; paused model recovery; finished Run with declared primary report; active check with partial coverage. Colors never imply a new domain outcome. Native CSS only; the host owns fixture wiring and verification.
