# Shared subject-specific widgets

Widgets compose reusable presentation and local interaction. Their hosts own
authenticated API reads, query/poll lifecycles, exact revisions, creation keys,
navigation targets and authorized mutations. `src/ui` owns controls, focus and
themes. Widgets never import `src/routes`, including its CSS or lazy modules;
`scripts/layer-boundaries.test.mjs` checks this against production sources.

- `artifact-preview/view.tsx`: render already loaded bytes by media type, with
  Rendered/Source tabs, bounded source lines and safe Markdown. Parsers,
  renderer styles and the lazy OpenAPI/LikeC4 renderer and Worker belong here.
  Generic format dispatch is shared across Materials, results, evidence and
  archives; it is independent of the Workflow that produced the material.
  `routes/artifacts/preview.tsx` still owns the request/response shell and archive
  transport. Hosts retain size limits, scope authorization and exact revisions.
  Standalone lazy Markdown consumers import `previews/markdown` directly to keep
  heavy renderer dependencies out of the initial bundle.
- `audit-report/view.tsx`: report status, safe lazy Markdown, downloads and
  retained-file facts for Reports and Inbox. The host supplies its existing
  query state and destinations for task results/coverage gaps. The widget does
  not fetch, poll or choose a check URL. List selection and report queries stay
  in `routes/reports`.
- `llm-cost/card.tsx`: Project/Audit/Run monetary summaries and local breakdown
  controls. `app/llm-cost.tsx` and `app/llm-cost-query.ts` retain query ownership,
  refresh and terminal reconciliation. Amounts and coverage come from the API;
  a missing amount is never interpreted as zero.
- `scan-reports/view.tsx`: scanner-specific presentation selected by the
  report's tool and supported schema, independently of its originating
  Workflow. Shared error outcomes remain one definition. Unknown reports keep
  source preview, and scanner success does not establish result quality.

Each widget imports its own CSS; rendering in Inbox must not depend on first
visiting another page. Route styles own only their layout and list/header
presentation. Theme tokens and existing DOM contracts remain shared.

Only extract blocks with real consumers. These modules are ordinary bundled
React components; no dynamic plugin loader, user-defined JavaScript, separate
launch lifecycle or new widget registry is introduced.
