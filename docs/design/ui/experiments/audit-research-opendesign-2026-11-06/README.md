# Audit research · OpenDesign review · 2026-11-06

Design stage of [V64-009](../../../../../tasks/v64-009-audit-priority-ui.yml),
assigned by the user on 2026-11-06. The production task remains **in progress**.
This preview does not establish real API, persistence or process acceptance.

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

The next implementation uses existing generated public clients, shared UI
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
