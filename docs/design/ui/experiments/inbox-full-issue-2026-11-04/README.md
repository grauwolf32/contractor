# Full issue review inside Inbox — 2026-11-04

V429-031 follows the user's request to inspect the same full issue information from Decide as from Issues. This is a reviewable preview; production source and all previous native/host artifacts remain unchanged.

`full-issue.tsx` imports the actual `IssueDetail` from `ui/src/routes/issues/issue-detail.tsx`. It supplies the selected canonical finding, project and existing Inbox decision callback. There is no second issue-information renderer: identity, Summary, Evidence, History, exact retained sources, snippets and the review composer all come from the existing Issues owners.

Inbox keeps its own queue bar, previous/next, J/K, mobile Back, focus and late decision navigation. Only the redundant Issues-list navigation row is hidden; the project and position are shown in the Inbox bar. “Open in Issues” retains the selected tab and the existing return state. Issues styles are scoped to the right-pane wrapper, with a late link to the original stylesheet so the earlier native preview's broad rules do not override its reviewed layout.

The fixtures reuse two preserved examples through `issueReviewFixtures`: debug information with a captured HTTP exchange and saved JSON response, and SQL construction with two saved source snippets. They are mapped to the existing synthetic queue identities; no proposal is auto-confirmed. History reads the actual retained source/provenance and an explicitly empty history page, rather than inventing a check result. Mark as duplicate uses the shared Project target picker. No retired automatic-correlation interface is reintroduced.

Read failures use the registered `storage_transaction_conflict` error and the existing Issues error/Retry UI. Exact Project evidence metadata/bytes are served only for the retained revision. The existing 64 KiB inline rule, first-two-snippets bound and explicit structured-data preview remain with Issues/Artifact preview. Every mutation is refused by the in-memory transport; live Server/model access is unnecessary.

## Review

```sh
node docs/design/ui/experiments/inbox-full-issue-2026-11-04/prepare.mjs
node docs/design/ui/experiments/inbox-full-issue-2026-11-04/verify.mjs
```

Use the existing Vite server on 127.0.0.1:17560 and open `/test-results/inbox-full-issue/index.html`. Version offers the full Inbox, previous compact view, actual Issues reference and a paired comparison. View selects the HTTP or source-code issue; all prior non-issue contexts remain available. Synthetic state controls exercise missing issue/source/evidence.

`prepare.mjs` extends only ignored generated modules from the prior preview. It changes the Inbox issue slot to the adapter and exposes the memory router solely for browser verification. Existing production components, package version and native CSS are untouched. All new module/style URLs include a content revision to avoid ignored-directory Vite caching.

`preserved-artifacts.json` protects the previous experiments, Issues/Inbox/decision source and original example data. `verification.json` records all-theme/width checks, paired content/font parity, deep links/return state, queue/keyboard/mobile navigation, one shared decision owner, retained drafts, exact downloads, classified failures and zero live API requests. Review captures are in `review/`.
