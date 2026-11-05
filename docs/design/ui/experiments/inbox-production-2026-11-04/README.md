# Approval-free Inbox — production transfer, 2026-11-04

V429-034 transfers the reviewed Inbox design onto main after V411-010 removed item and report approvals. This preview renders the actual production `applicationRoutes` without replacement routes, injected style overrides or a second reader. Shared typography and theme tokens remain unchanged.

Decide contains canonical unassessed Project Findings. Its complete reader is the same Issues component, including Summary, saved evidence/snippets, independent-source History and the Project annotation owner. Inbox supplies only its queue bar, return links and guarded late-action navigation. Ready uses the existing Reports query/content/status/Markdown/download owners. There are no review requests, acceptance actions or approval compatibility paths. A completed check with no aggregate report shows ordinary unavailability and links to its retained task results.

Desktop selects the first item once after successful initial reads; explicit selections and partial reads win. Mobile starts with the queue. Back, resize, refresh and the last annotation never repeat automatic entry. Unselected/loading/partial/empty content is compact. The capacity disclosure and its optional Operations/Workflow reads are removed while actionable Run rows remain.

Production CSS adapts the retained OpenDesign section accents, count pills, selected rows and compact queue bar with explicit Inbox selectors and shared tokens. The broad native selector rules are not shipped over the shared Issues/Reports components. Their content and font metrics are compared directly in browser checks. All original native inputs/output and prior experiment files are protected by `preserved-artifacts.json`; no new model call or modification of the original experiments occurs.

## Preview and verification

```sh
node docs/design/ui/experiments/inbox-production-2026-11-04/prepare.mjs
node docs/design/ui/experiments/inbox-production-2026-11-04/verify.mjs
```

With the existing Vite server on 127.0.0.1:17560, open `/test-results/inbox-production/index.html`. View selects queue entry, both complete issue examples, a report, model recovery, a Run result or a running check. State exercises read failures, partial/empty queues and pending/unavailable reports. Version opens actual Issues/Reports reference routes; width and theme cover desktop/mobile and all themes.

Preparation extracts the current route test's bounded, schema-validated synthetic fixtures and emits ignored preview modules. Original saved issue examples supply exact source snippets and HTTP evidence. Markdown/JSON report fixtures carry correct byte lengths and SHA-256 digests. An in-memory public client validates JSON responses, refuses every mutation and sends no live API/model request. The preview is not connected to the demo database.

The route suite covers actual annotations, late responses/focus, explicit links, one-time desktop entry and complete reports becoming unavailable without losing Refresh focus. Browser verification additionally checks mobile Back/resize, J/K, exact downloads, reference content/font parity, all themes/widths, absence of approval/capacity calls, truthful no-report states and original artifact hashes. Captures and results are retained in `review/` and `verification.json`.

The UI patch release is 0.11.10. User authorization on 2026-11-04 includes adapting to merged Audit changes and merging this implementation into main after verification. This supersedes the earlier preview-only merge hold without changing the historical experiment records.
