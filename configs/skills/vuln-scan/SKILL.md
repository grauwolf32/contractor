---
name: vuln-scan
description: "Review source for vulnerabilities using endpoint controls, dangerous sinks, absence detection, business logic, and sensitive-data sweeps."
compatibility: "Contractor adk@1 native Agent Skill disclosure; requires source browsing; report through an available output tool or response"
metadata:
  source-revision: 9c76b56cf7b83377fb1dd5e4a17440fa27b723f3
---

# Vulnerability Scanning Skill

This `SKILL.md` is loaded only after `load_skill(skill_name="vuln-scan")`.
Load a reference only when its topic becomes relevant by calling
`load_skill_resource(skill_name="vuln-scan", file_path="references/<topic>.md")`;
references are not injected automatically. Treat the workflow verbs below as
operations to perform with whatever read-only source and reporting tools are
available in the Worker invocation, not as guaranteed tool names.

## Workflow

1. **Inventory** — glob source files, identify framework + language.
   If the target is PHP / a WordPress plugin (`*.php`, `Plugin Name:`
   header), call
   `load_skill_resource(skill_name="vuln-scan", file_path="references/php-wordpress.md")`
   first — the generic
   patterns below miss WP AJAX authz, privilege escalation, `$wpdb` SQLi,
   file deletion, and stored XSS.
2. **Per-file scan** — for each handler file:
   a. inventory all handlers with the available source-navigation interface
   b. Check auth decorators on each handler (compare siblings)
   c. grep for dangerous sinks within the file
   d. Check ownership on data-access handlers
   e. Report each finding immediately
3. **Business-logic pass** — for each handler that moves money / changes a
   balance, quantity, quota, credit, vote, or ownership, or advances a
   multi-step workflow, load
   `load_skill_resource(skill_name="vuln-scan", file_path="references/business-logic.md")`
   and check
   atomicity, idempotency, amount bounds, client-trusted values, and
   step-order enforcement. These have NO dangerous sink and pass auth/authz —
   grep will not find them.
4. **Secrets & sensitive-data sweep** — call
   `load_skill_resource(skill_name="vuln-scan", file_path="references/secrets.md")`.
   Two halves: (a) secrets at rest in code/repo — grep provider-key regexes
   and glob for **sensitive files committed to the repo** (CWE-538 / CWE-312):
   `**/.env*`, `**/*.pem`, `**/*.key`, `**/*.p12`, `**/*.pfx`, `**/id_rsa`,
   `**/*.sql`, `**/*.sql.gz`, `**/*.dump`, `**/*.bak`, `**/*.backup`,
   `**/wp-config.php`, `**/credentials*`, `**/.git/config`,
   `**/.aws/credentials`, `**/*.kdbx`, `**/*.keystore` (exclude `.example` /
   `.sample` / test fixtures); plus weak crypto. (b) the **runtime exposure
   sweep** — secrets/credentials/PII/PAN returned, logged, or shipped to
   clients — run on every handler that returns/logs/persists data.
5. **Report** — publish each confirmed finding through an available reporting
   mechanism; otherwise return it in the requested
   result artifact or response.

## References (load on demand)

- `references/grep-patterns.md` — ready-to-use grep patterns organized by severity; run these FIRST
- `references/absence-detection.md` — patterns for finding MISSING controls (auth, ownership, rate-limit, role); **EQUALLY IMPORTANT as grep patterns**
- `references/checklist.md` — per-endpoint control checklist (auth, authz, ownership, validation, output filtering, rate limiting)
- `references/sink-patterns.md` — language-specific dangerous functions with safe vs vulnerable examples
- `references/miss-patterns.md` — commonly missed vulnerability patterns with examples
- `references/business-logic.md` — abuse-of-functionality flaws with NO sink and passing auth/authz (race/TOCTOU, missing idempotency, unbounded/negative amounts, client-trusted values, workflow bypass); grep can't find these — load whenever a handler moves money/quantity/state
- `references/secrets.md` — secrets & sensitive-data exposure: secrets at rest (code/repo) + the runtime exposure sweep (secrets/credentials/PII returned, logged, or shipped to clients)
- `references/php-wordpress.md` — PHP / WordPress-plugin sinks + absence patterns ($wpdb SQLi, `wp_ajax_nopriv_` missing-authz / privilege escalation, file deletion/upload, stored XSS); **load whenever the target has `*.php` files** — the generic patterns miss almost all of it
