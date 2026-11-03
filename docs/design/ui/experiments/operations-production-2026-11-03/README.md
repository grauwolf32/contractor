# Operations: working React UI

User authorization: 2026-11-03, “Да, перенеси в рабочий ui”. Task: V429-028.

This preview renders the production `Application`, router, API clients and
Operations/Settings components. The transport supplies schema-valid synthetic
responses in memory. It has no connection to the demo database or a model.
Settings writes update fixture memory; inventory writes return an explicit
synthetic rejection. Refreshing the preview restores its fixture data.

## Open and reproduce

From the repository root, with the pinned Node 24.20 toolchain:

```sh
node docs/design/ui/experiments/operations-production-2026-11-03/prepare.mjs
cd ui
corepack pnpm dev --host 127.0.0.1 --port 17560 --strictPort
```

Open [working Operations](http://127.0.0.1:17560/test-results/operations-production/index.html?section=overview&version=new&case=attention&theme=dark&width=1440)
or [working Settings](http://127.0.0.1:17560/test-results/operations-production/index.html?section=settings&version=new&case=normal&theme=dark&width=1440).
The toolbar selects nine sections, themes, widths and synthetic read states.
The “Approved design” option uses the preserved Settings/reference server on
port 17570; run its `serve.mjs` if needed. Controls inside the Working UI use
the actual React navigation and forms.

`prepare.mjs` writes only ignored `ui/test-results/operations-production/`
files. Its entry URL includes a source digest because the development server
does not watch ignored test output. Start a fresh development server after a
UI version change; its build-time version is captured at startup.

## Transfer and ownership

The reviewed native sources remain under
`operations-opendesign-2026-11-02`, `operations-tabs-opendesign-2026-11-03` and
`operations-settings-opendesign-2026-11-03`. Their native generation history,
CSS, host assembly, recovery and verification evidence are retained intact.
`reference-manifest.json` records SHA-256 for all 107 reference files against
base commit `dbc23e036b31f9efd1d0ae9cc83806df919f56c7`.

The production transfer adapts their presentation in React and token-based
CSS; it is not another native generation. No model selection or live
credential was changed and no OpenDesign data was rewritten.

| Approved idea                                         | Production owner                                                                                                |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Labelled sidebar and Setup group                      | Existing Operations routes and capability gate; compact labelled mobile disclosure                              |
| Snapshot cards, colored slot track and compact legend | Overview derives exact counts from the existing snapshot                                                        |
| Inspection list and expandable processes              | Existing observed/authoritative process facts; hash links open the target                                       |
| Compact inventories and resource rows                 | Existing principals, allocations, pagination, bindings, credentials and history reads                           |
| CPU, memory, HTTP and database hierarchy              | Observed CPU/RSS bars inside the summary cards; existing independent Performance queries and metric definitions |
| Execution and personal Settings columns               | Existing separately saved models/scheduling and personal Git/theme components                                   |
| Scheduling steppers and dirty footers                 | Existing public bounds, one draft for both limits, independent CAS writes                                       |
| Progressive explanations and role overrides           | Full existing help, identities, diagnostics and forms retained on demand                                        |

Global font families, color tokens, shared icons, buttons, dialogs and
disclosures remain the same as elsewhere in Contractor. `operations/styles.ts`
orders the existing base styles before the new presentation so that a Run's
resource card cannot reverse style precedence through a different lazy import.

The Overview presents at most six processes, six allocations and four
inspection rows, with full-inventory links. Its counts cover the entire
snapshot. A mismatch outside the preview links to the full process inventory.
Reserved slots with no observation remain distinct from a reported lost lease.

## Verification

```sh
node docs/design/ui/experiments/operations-production-2026-11-03/verify.mjs
node docs/design/ui/experiments/operations-production-2026-11-03/compare.mjs
make ui-verify
make ui-gates
```

The browser verifier covers 108 section/theme/width layouts at
1440/1100/390/320px plus ten empty/read-failure layouts and one reconnecting
layout. It exercises real desktop/mobile navigation, process inspection,
completed resource disclosures, publication/binding dialogs, LLM filtering,
independent model/scheduling PUTs in fixture memory, bounds, roles, 412
conflicts, the enabled Git editor and theme preference. It rejects horizontal
overflow, invalid API responses, browser errors and all production/external
API requests. It also checks every reference hash. Selected screenshots and
measurements are retained in `review/`. `compare.mjs` additionally captures
40 paired approved/working screenshots across all nine sections, desktop and
phone, including expanded Runtime cards. It checks that tool names are visible
after one disclosure and that the fact cells occupy two actual columns on
desktop and one on phones. This was added after the user's review rejected the
first transfer for hidden toolsets, different graphs and other visual drift.

Open [the paired comparison](http://127.0.0.1:17560/test-results/operations-production/comparison/index.html?section=performance&width=1440&state=collapsed).
Its working UI is built from real components with the same synthetic transport.
The global production header/search, authoritative Server words and additional
working actions remain. The real 15-second history has 60 bars in this fixture;
the approved static sketch has 16 demonstration bars. Missing data is not
interpolated, measured zero stays zero, and full history/counters remain
available under their disclosures.

The production UI suites retain API validation, capability boundaries,
pagination, destructive confirmation, secret clearing, drafts and revision
conflict tests. The Operations forms, Git artifacts and model selection
browser suites, plus Performance/GPU, pass 19 scenarios against the built UI
with contract-checked API mocks. `make ui-verify` passes 224 test files,
2661 tests and one declared expected failure; lint, typecheck and build pass.
The final run limits Vitest to four workers; two earlier unrestricted runs hit
the existing five-second large-Issues-evidence test timeout under host load.
No unrelated test or timeout was changed.

The attempted `make test-ui-stack-operations` used an isolated disposable
PostgreSQL instance and failed before starting the UI: the staged existing
`configs/e2e/workflows/streamline_copy.yaml` selects the absent ModelPolicy
`worker@1`. Neither `configs/` nor `tests/ui-stack/` changed. This is an unproven
process-stack boundary, not a passing integration check. The temporary database
server was stopped and no demo data was involved. `make lint` also passed;
the final UI gates were rerun after the fidelity corrections.

This change is prepared on `codex/operations-ui`; merging into main requires
the user's subsequent instruction.
