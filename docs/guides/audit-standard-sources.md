# ASVS and WSTG audit presets

[Audit guide](audits.md)

`owasp-asvs-5-0-l1-source-review@1` selects all **70 Level 1 requirements** from
ASVS 5.0.0. The previous `@2` five-requirement pilot and its standard package
remain immutable. The expanded package uses the catalog edition
`owasp-asvs@5.0.0-l1-source.1`; its upstream requirement IDs still use
`v5.0.0-…`. The suffix identifies Contractor's package edition, not a new OWASP
release. Every selected requirement produces exactly one Audit item.

`owasp-wstg-4-2-source-review@1` is a **separate Audit preset** with **94 active
WSTG 4.2 scenarios** across 12 categories. It pins `owasp-wstg@4.2`, using the
version-qualified `WSTG-v42-…` identifiers. It does not add WSTG checks to ASVS.

`owasp-wstg-4-2-active-http@1` is a second, independent WSTG preset for a running
website/API. Its edition `owasp-wstg@4.2-http.1` contains the same 94 scenarios
with HTTP methods and a distinct evidence contract. Source code is not required.

The two source-review presets require a `source` ZIP containing the application source and any
relevant configuration and documentation. They run one check per ordinary Run,
allow three attempts per check, and retain the exact standard package, mapping,
evidence policy and finding origin. Findings are proposals requiring analyst
confirmation. The worker has source-reading and evidence-publication tools and
does not have HTTP or active scanning tools.

## Evidence and scope

ASVS documentary requirements `2.1.1`, `6.1.1`, `8.1.1` and `15.1.1` retain human
applicability review. Source controls can be assessed only for the inspected
scope; unresolved paths remain gaps. ASVS deployment requirements `12.1.1`,
`12.2.1`, `12.2.2` and `13.4.1` do not permit `satisfied` from source alone.

WSTG describes tests of running systems. This preset adapts its scenarios to
source review: it can publish concrete source-backed violations, but cannot
report a passed dynamic test. Its evidence contract permits `violated`,
`inconclusive`, `blocked` and `not-tested`; it does not permit `satisfied`.
Missing live observations, external discovery and timing measurements are
explicit gaps. Completing the Audit is not a completed live WSTG assessment.

## Active HTTP WSTG

Select **WSTG 4.2 · Active HTTP checks** when creating a separate Audit. Supply
an exact `context` text/Markdown artifact describing endpoints, disposable test
data, prerequisites and exclusions. Set the Audit scope's `target` to the
authorized HTTP(S) origin and `authorizationScope` to the allowed paths and
actions. For example:

```text
target: https://staging.example.test
authorizationScope: /api/* and /login; synthetic test accounts only;
read-only probes and POST/PATCH against designated disposable records.
No load testing, brute force, account deletion or production data changes.
```

Every item starts in **Reviews** for the existing active-check approval. A
missing or ambiguous scope blocks the worker. The profile allows one attempt
per item to avoid automatically re-running state-changing probes; failed checks
remain gaps. Findings still require analyst confirmation.

The worker uses the existing HTTP toolset and retains response artifacts plus
redacted request/response observations. It can mark a scenario `satisfied` only
when live evidence establishes all applicable objectives for the declared
scope. Browser execution, raw network/TLS tests, external callbacks, external
discovery and isolated multi-user sessions are not implemented by this preset.
Unsupported objectives remain `blocked`, `inconclusive` or `not-tested`.

Scope restrictions and the 20-call worker limit are instruction-level limits;
the existing HTTP transport is not an engagement allowlist or a deterministic
proof/replay engine. It may retry idempotent requests internally (up to three
attempts). Managed Project Authorization overrides a supplied header on that
origin, so clearing a session does not prove an anonymous identity. This preset
uses the current approval-based HTTP workflow; it does not implement the
autonomous pentest contracts in draft specification 33.

## Pinned sources and licensing

The packages reproduce OWASP material under **CC BY-SA 4.0**, with attribution
inside each package. Contractor's titles, mapping objectives and evidence
policies are identified as adaptations under the same license.

| Source | Exact revision | Imported content |
| --- | --- | --- |
| [ASVS 5.0.0 JSON](https://github.com/OWASP/ASVS/blob/5cf9b032440be53ce345ab3c130fda46ba1ce7a2/5.0/docs_en/OWASP_Application_Security_Verification_Standard_5.0.0_en.json) | `5cf9b032440be53ce345ab3c130fda46ba1ce7a2` | Every requirement with `L = 1`, with unchanged statements |
| [WSTG 4.2 checklist](https://github.com/OWASP/wstg/blob/dd33419e10edb22b78d89325a6c2aad9f184e3a2/checklist/checklist.json) | `dd33419e10edb22b78d89325a6c2aad9f184e3a2` | Active scenario titles and test objectives |

The WSTG release checklist contains 98 rows. Four are not independent active
scenarios, as confirmed by the Markdown pages at the same revision:

- `INFO-09` was merged into `INFO-08`.
- `INPV-03` was merged into `CONF-06`.
- `ERRH-02` was merged into `ERRH-01`.
- The old buffer-overflow `INPV-13` page was removed. The active format-string
  injection scenario also uses `INPV-13` and is included exactly once.

These exclusions leave 94 scenarios: INFO 9, CONF 11, IDNT 5, ATHN 10, ATHZ 4,
SESS 9, INPV 18, ERRH 1, CRYP 4, BUSL 9, CLNT 13 and APIT 1. Scenario links use
the versioned `v42` documentation, not the moving `stable` alias.

## Rebuild and verify

Download the two exact source JSON files linked above, then run:

```sh
python3 scripts/build-audit-source-standards.py \
  --asvs-source /path/to/asvs.json \
  --wstg-source /path/to/wstg.json
go test ./internal/config ./internal/auditstandards ./tests/eval/audit_programs
```

The builder checks pinned SHA-256 hashes before generating the three packages. It
does not fetch data or change the legacy ASVS package. The ASVS regression
fixture in `tests/eval/audit_programs/testdata/asvs-5.0.0-level1.json` contains
the same upstream Level 1 statements and shares the attribution and license
above. Tests validate unchanged statements, category counts, exact inventory
generation, applicability decisions, evidence restrictions and compatibility.

Published catalog identities are create-only. Any future content change must
use another package edition and a new AuditProfile version; existing Audits
continue to use their retained packages.
