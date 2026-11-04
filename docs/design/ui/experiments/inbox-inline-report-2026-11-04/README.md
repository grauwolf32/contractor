# Complete reports inside Inbox — 2026-11-04

V429-032 follows the user's request to load the report directly in the Inbox, alongside the full issue reader from V429-031. This is a reviewable preview; production source, package version and all previous OpenDesign artifacts remain unchanged.

`full-report.tsx` imports the actual `ReportContent`, `ReportStatusChip`, `ReportAcceptanceDecision`, `useAuditReport` and `useReportAcceptance` owners. Both a proposed report in Decide and a ready report use their full content, status and downloads. There is no second Markdown renderer, report-query state machine or acceptance implementation. The existing safe Markdown rendering, report revision handling and matching-review eligibility remain with Reports.

Inbox keeps its queue bar, previous/next, J/K, mobile Back, focus and late decision navigation. “Open report” retains the requested review identity. If a ready row disappears after a report read fails or becomes unavailable, the selected check still reaches the shared report reader and its actual error/status UI. Acceptance is never inferred from the queue row alone.

`report-fixtures.ts` provides two schema-validated synthetic reports below 16 KiB each. Their complete Markdown includes a table, limitations and follow-up beyond the former excerpt. Markdown and JSON downloads match the response exactly; retained metadata records real SHA-256 digests and byte lengths. Pending, unavailable, failed-read and mismatched-review cases use the existing Reports UI. All requests stay in the in-memory fixture, and every mutation is refused locally.

## Review

```sh
node docs/design/ui/experiments/inbox-inline-report-2026-11-04/prepare.mjs
node docs/design/ui/experiments/inbox-inline-report-2026-11-04/verify.mjs
```

Use the existing Vite server on 127.0.0.1:17560 and open `/test-results/inbox-inline-report/index.html`. View selects the proposed or ready report, full issue examples and earlier Inbox contexts. Version offers the inline report, previous excerpt, actual Reports reference and a paired comparison. Width and Theme cover the shared desktop/mobile layouts.

`prepare.mjs` extends ignored generated modules from the full-issue preview. It replaces only the preview's report slots, links the existing Reports stylesheet after the earlier native preview rules and adds revisioned module URLs. No new native OpenDesign or model call is made. Original experiment files and production components are protected by the hashes in `preserved-artifacts.json`.

`verification.json` records layout/state checks, paired Reports content and typography, exact downloads, queue/mobile navigation, refused decisions, the existing source-code issue reader and zero browser errors or live API requests. Captures are retained in `review/`. The preview is opened directly for the user; merging and production transfer are separate steps.
