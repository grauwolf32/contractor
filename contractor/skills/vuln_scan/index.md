---
description: Vulnerability scanning skill — workflow overview, checklist references, common miss patterns, business-logic flaws, and secrets/sensitive-data exposure. Read this first, then load references on demand with skills_read.
---

# Vulnerability Scanning Skill

## Workflow

1. **Inventory** — glob source files, identify framework + language.
   If the target is PHP / a WordPress plugin (`*.php`, `Plugin Name:`
   header), load `vuln_scan/references/php-wordpress` FIRST — the generic
   patterns below miss WP AJAX authz, privilege escalation, `$wpdb` SQLi,
   file deletion, and stored XSS.
2. **Per-file scan** — for each handler file:
   a. `list_symbols` to inventory all handlers
   b. Check auth decorators on each handler (compare siblings)
   c. grep for dangerous sinks within the file
   d. Check ownership on data-access handlers
   e. Report each finding immediately
3. **Business-logic pass** — for each handler that moves money / changes a
   balance, quantity, quota, credit, vote, or ownership, or advances a
   multi-step workflow, load `vuln_scan/references/business-logic` and check
   atomicity, idempotency, amount bounds, client-trusted values, and
   step-order enforcement. These have NO dangerous sink and pass auth/authz —
   grep will not find them.
4. **Secrets & sensitive-data sweep** — load `vuln_scan/references/secrets`.
   Two halves: (a) secrets at rest in code/repo — grep provider-key regexes
   and glob for **sensitive files committed to the repo** (CWE-538 / CWE-312):
   `**/.env*`, `**/*.pem`, `**/*.key`, `**/*.p12`, `**/*.pfx`, `**/id_rsa`,
   `**/*.sql`, `**/*.sql.gz`, `**/*.dump`, `**/*.bak`, `**/*.backup`,
   `**/wp-config.php`, `**/credentials*`, `**/.git/config`,
   `**/.aws/credentials`, `**/*.kdbx`, `**/*.keystore` (exclude `.example` /
   `.sample` / test fixtures); plus weak crypto. (b) the **runtime exposure
   sweep** — secrets/credentials/PII/PAN returned, logged, or shipped to
   clients — run on every handler that returns/logs/persists data.
5. **Report** — `report_vulnerability` for each confirmed finding

## References (load on demand)

- `vuln_scan/references/grep-patterns` — ready-to-use grep patterns organized by severity; run these FIRST
- `vuln_scan/references/absence-detection` — patterns for finding MISSING controls (auth, ownership, rate-limit, role); **EQUALLY IMPORTANT as grep patterns**
- `vuln_scan/references/checklist` — per-endpoint control checklist (auth, authz, ownership, validation, output filtering, rate limiting)
- `vuln_scan/references/sink-patterns` — language-specific dangerous functions with safe vs vulnerable examples
- `vuln_scan/references/miss-patterns` — commonly missed vulnerability patterns with examples
- `vuln_scan/references/business-logic` — abuse-of-functionality flaws with NO sink and passing auth/authz (race/TOCTOU, missing idempotency, unbounded/negative amounts, client-trusted values, workflow bypass); grep can't find these — load whenever a handler moves money/quantity/state
- `vuln_scan/references/secrets` — secrets & sensitive-data exposure: secrets at rest (code/repo) + the runtime exposure sweep (secrets/credentials/PII returned, logged, or shipped to clients)
- `vuln_scan/references/php-wordpress` — PHP / WordPress-plugin sinks + absence patterns ($wpdb SQLi, `wp_ajax_nopriv_` missing-authz / privilege escalation, file deletion/upload, stored XSS); **load whenever the target has `*.php` files** — the generic patterns miss almost all of it
