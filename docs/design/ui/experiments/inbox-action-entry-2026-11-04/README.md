# Inbox — act on entry, 2026-11-04

V429-030 corrects the previous experiment's redundant four-counter first screen after user feedback. This is a **host-owned entry-flow proposal** using the retained OpenDesign styling; OpenDesign was not run again. The original project, conversations, CSS, captures and preview remain intact.

On a successful desktop entry, the Inbox selects the first row in its existing model order: Decide, Unblock, Ready, Running. It opens the existing detail and controls without recording a decision. Mobile opens the queue. Explicit item links are preserved. Initial selection is consumed once, so refresh, Back, resizing and later reads cannot redirect a user's selection.

The unselected view is compact: a concrete Continue action when an item is available, an existing Start a check destination when the Inbox is empty, or explicit loading/read-error content. Partial reads do not claim the Inbox is clear or guess which missing item has priority.

## Run

```sh
node docs/design/ui/experiments/inbox-action-entry-2026-11-04/prepare.mjs
node docs/design/ui/experiments/inbox-action-entry-2026-11-04/verify.mjs
```

Use the existing local Vite server on 127.0.0.1:17560 and open `/test-results/inbox-action-entry/index.html`. Version offers Revised Inbox, Previous OpenDesign, Current UI and a side-by-side comparison. The old `/test-results/inbox-design/` preview is preserved.

`prepare.mjs` reuses the previous synthetic transport and native CSS. It creates ignored copies of the real Inbox index/detail modules, rewrites their imports to the actual shared components, injects a one-time initial-selection hook, and replaces only the unselected view with `focus-overview.tsx`. It does not edit production routes, shared fonts, API contracts or the existing native/host artifact files. Generated module and style URLs include their content revision because Vite does not watch ignored captures. A production transfer would apply the reviewed behavior directly rather than shipping these preview copies.

`preserved-artifacts.json` freezes all 54 files of the prior experiment. The browser verifier checks those hashes, the original production-source hashes, priority fallback, explicit links, mobile Back/resize, keyboard navigation, failure/empty states, rejected in-memory decisions, comparison frames and light/dark/black at 1440/1100/390/320px. `verification.json` and `review/` retain the results. No live API request or native-model call is required.
