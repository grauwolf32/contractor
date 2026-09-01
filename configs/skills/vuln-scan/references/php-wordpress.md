---
description: How to audit PHP / WordPress code for vulnerabilities — handler discovery, authorization, and request-to-sink data flow. Load this when the target has *.php files or a WordPress plugin/theme layout. This is general methodology, not a catalogue of specific bugs.
---

# PHP / WordPress Audit Methodology

The generic (Python/Java/Go) handler and auth patterns miss PHP/WordPress
code because the idioms differ: requests arrive through superglobals, and
routing/authorization are library calls rather than decorators or
middleware. This reference is about *how to audit* such a codebase — not a
list of patterns to match. Apply it to any `*.php` target (WordPress
plugin/theme: a `Plugin Name:` header or `readme.txt` with `Stable tag:`).

## 1. Enumerate the handlers (entry points)

WordPress registers handlers as hooks, not routes. Find them all before
auditing:

- AJAX: `add_action('wp_ajax_<action>', <cb>)` and the **unauthenticated**
  `add_action('wp_ajax_nopriv_<action>', <cb>)`.
- REST: `register_rest_route(...)` — note its `permission_callback`.
- Forms / admin: `admin_post_*`, `admin_post_nopriv_*`, shortcode and form
  handlers.

The registered callback is the unit to audit. `nopriv` handlers are
reachable without authentication.

## 2. Check authorization on each state-changing handler

For every handler that changes state, ask: is there an authorization check
appropriate to the action (`current_user_can(...)`)? A nonce
(`check_ajax_referer` / `wp_verify_nonce`) only proves the request
originated from a WP page — it is **not** authorization. A state-changing
handler protected by a nonce but no capability check is a
missing-authorization issue (critical when registered `nopriv`).

## 3. Trace request data into sinks

Treat `$_GET` / `$_POST` / `$_REQUEST` / `$_FILES` / `$_COOKIE` as
attacker-controlled — even when a UI form normally supplies the value.
Follow each tainted value and ask whether it reaches a sink without the
control that would make it safe:

- **SQL** — does it reach a `$wpdb` query without a correct prepared
  statement? (Note: input sanitisers like `sanitize_text_field()` are not
  SQL escaping.)
- **Privilege** — does it set a user's role or capabilities without a
  server-side allowlist of permitted values? → privilege escalation.
- **Filesystem** — does it reach a file read/write/delete/upload without
  confining the path or validating the type server-side?
- **Output** — is it echoed or rendered into HTML without WordPress output
  escaping (`esc_html` / `esc_attr` / `wp_kses`, or a template's autoescape)?
  → reflected or (if stored first) stored XSS; report the output sink.

## 4. Confirm, then report

Read the sink and confirm a request value actually reaches it before
reporting. WordPress safety lives on the OUTPUT side (escaping) and in
correct `$wpdb->prepare()` usage — it is the *absence* of these around
tainted data that constitutes the bug. Report each confirmed finding once;
prefer lowering confidence over inflating the count.
