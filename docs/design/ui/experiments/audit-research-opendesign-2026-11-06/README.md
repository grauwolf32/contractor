# Audit research · OpenDesign review · 2026-11-06

Design and production review of [V64-009](../../../../../tasks/done/v64-009-audit-priority-ui.yml),
assigned by the user on 2026-11-06. Production integration is implemented and
verified; the task records the completed gates. The native preview below is
synthetic and is distinct from those real API/process checks.

## Review

Run `node docs/design/ui/experiments/audit-research-opendesign-2026-11-06/prepare.mjs`
from the repository root, then serve `ui/test-results/audit-research/` on loopback.
`index.html` provides width, theme, section and failure-state controls. `native.html`
opens the actual frame without the review toolbar. The preview uses synthetic
Shop API data and performs no API writes or target traffic. All actions are local
simulations and explicitly labeled as such.

The six sections are Overview, Work, Hypotheses, Possible issues, Materials and
Activity. Review the fixed checklist separately from additional hypotheses,
supported/refuted/inconclusive results, exact grounds versus origin, saved proposal
versus registered assertion, and active Run versus accepted checking work.
Pause retains active work and explains draining. Material Preview and explicit
Project publication preserve the selected revision in the proposed journey.

## Ownership and preservation

| File | Owner |
| --- | --- |
| `native-inputs/seed.html` | Host: synthetic facts, existing shell vocabulary, controls and stable interaction hooks |
| `native-inputs/brief.md` | Host: reviewed contract constraints and visual direction |
| `native-inputs/theme.css` | Exact copy of the existing shared project font/color tokens |
| `native/index.html`, `native/audit.css` | Untouched files produced by the isolated OpenDesign project |
| `preview.js` | Host: local synthetic interactions; no production API client |
| `host-polish.css` | Host: explicitly separate dynamic detail/form styling and verified responsive repairs |
| `index.html`, `review.js`, `prepare.mjs` | Host: review toolbar and repeatable preview preparation |
| `provenance.json`, `preservation.json` | Sanitized generation identity, hashes and preservation checks |
| `review/`, `verification.json` | Browser screenshots and design-only verification results |

All original native outputs are retained independently of host repairs. The
existing OpenDesign instance has a read-only root, no repository host mount,
an internal network and a bounded relay to the already configured Contractor
gateway/model. It receives only synthetic reference inputs and existing theme
tokens. No native Skills, dependency installation or additional network access
are authorized by the brief. The host verifies output before serving it locally.

The private pre/post state archives live outside Git in the existing OpenDesign
backup directory. Verification checks old project files, application SQLite rows
and OpenCode history; shared operational timestamps/log appends are recorded
separately. The owned temporary model relay is stopped and its upstream credential
file removed after generation. No existing design project is deleted or replaced.

## Production boundary

The production implementation uses existing generated public clients, shared UI
components and query/polling owners. No new backend, registry, receipt or execution
path is proposed. Task counts come from the workspace projection, hypotheses and
attempts from their bounded owner pages, materials from accepted receipt links,
and possible issues from the canonical Project Finding owner. Content preview
uses existing bounded Artifact reads. Source Run deletion remains explicit
provenance, not a substitute link to a different Run.

Semantic guarantees follow `ACT-R14`–`ACT-R17`, `HG-R1`–`HG-R16`, `ART-R138`,
`AUX-R108`–`AUX-R112`, the Project Finding owner and shared UI vocabulary. A successful
Run cannot synthesize a hypothesis verdict or an analyst assessment. A saved
proposal is not a registered hypothesis; registration does not schedule a check.
The fixed seed denominator is immutable. Task progress is not security coverage.
Refuted/inconclusive checks remain scoped history; negative rechecks do not close
issues. Full exact IDs stay under Technical details, never in user-authored forms.

`node docs/design/ui/experiments/audit-research-opendesign-2026-11-06/verify.mjs`
is a **design-only** browser check. The task's production `ui-verify`, real public
API/PostgreSQL, basic process/browser and lint gates still apply when implementing
the production journey. A synthetic screenshot is not evidence those gates passed.


## Combined production preview

The existing Checks list, filters, summary and keyboard navigation remain the
entry point. The detail page adds the reviewed overview, colored work timeline,
Task list/detail workspace, hypothesis history, exact material receipts and
activity. It uses the project's shared components, typography and theme tokens.
Native OpenDesign inputs/outputs and preservation records above remain unchanged.

Run `node docs/design/ui/experiments/audit-research-opendesign-2026-11-06/prepare-production.mjs`
with the UI development server running, then open
`http://127.0.0.1:17560/test-results/audit-research-production/`.
This preview renders production React routes with synthetic contract fixtures;
writes are disabled. It is distinct from the untouched native design review.

Production browser verification uses `make test-audit-research-ui`, registered in
the release browser stage. It reuses the real hypothesis-continuation scenario,
Server, Runtime and PostgreSQL with a scripted test Gateway. The browser uses
ordinary session/CSRF APIs, resumes retained work, inspects negative and later
inconclusive checks, follows generated grounds, reads the canonical issue origin
and publishes an exact material into Project. The mock browser gate additionally
covers bounded pages, failed/ambiguous writes and all six sections in three themes.
No live model quality claim follows from these deterministic checks.


## Final verification

On 2026-11-06 `make ui-verify` passed all 214 UI test files (2419 passing tests
and one intentional expected failure), typecheck and build. `make ui-browser-mocked`
passed all 163 cases; `make test-audit-research-ui` passed the real browser/API
journey, including exact publication and 390 px in every theme. Public API and
PostgreSQL contract suites, all 12 clean-source common Audit process cases and
`make lint` passed. The final source implementation is
`5ff2678a13cf2d2c121326104facad5825a4c676`. The reviewed implementation landed in main on 2026-11-07 through
`579e6cc970ba14be4d072c8084957e7696ff6848`;
[V64-039](../../../../../tasks/done/v64-039-research-ui-main-integration.yml)
records the integration checks and preservation of concurrent work.
