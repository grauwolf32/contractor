---
description: PHP and WordPress-plugin sink + absence patterns (SQLi via $wpdb, AJAX missing-authorization / privilege escalation, arbitrary file deletion/upload, stored XSS). Load this when the target is PHP or a WordPress plugin/theme.
---

# PHP / WordPress Vulnerability Patterns

The generic (Python/Java/Go) patterns miss almost everything in PHP and
WordPress code, because the idioms are different: requests arrive through
superglobals (`$_GET`/`$_POST`/`$_REQUEST`/`$_FILES`), routing is via
`add_action('wp_ajax_*')` rather than decorators, and authorization is a
`current_user_can()` call rather than a middleware. Use this reference
whenever the codebase contains `*.php` files or a `readme.txt` with a
`Stable tag:` / a top-level file with a `Plugin Name:` header.

## Entry points: find every request handler first

WordPress plugins rarely use routes. The real entry points are AJAX
actions, REST routes, and form/admin-post hooks:

```
grep "add_action('wp_ajax_"          # authenticated AJAX
grep "add_action('wp_ajax_nopriv_"   # UNAUTHENTICATED AJAX  ← highest risk
grep "register_rest_route"           # REST endpoints (check permission_callback)
grep "add_action('admin_post_"  "admin_post_nopriv_"
grep "add_shortcode("  "do_action("
```

Each `add_action('wp_ajax[_nopriv]_<action>', <callback>)` registers
`<callback>` as a handler. `wp_ajax_nopriv_` means **unauthenticated**
users can call it. For EVERY callback, audit the two controls below.

## Missing authorization (CWE-862) — the #1 WP plugin bug

A handler is vulnerable if it performs a privileged/state-changing action
without BOTH:
- a **capability check**: `current_user_can('...')`, and
- a **nonce check**: `check_ajax_referer()` / `wp_verify_nonce()`.

A nonce check alone is NOT authorization — it only stops CSRF. A handler
with only `wp_verify_nonce` and no `current_user_can` is still a missing-
authorization finding (any logged-in user, or with `nopriv` anyone, can
reach it).

```php
// VULNERABLE: data-changing AJAX with no capability check
add_action('wp_ajax_tutor_update_profile', [$this, 'update_profile']);
public function update_profile() {
    tutor_utils()->checking_nonce();      // nonce only — NOT authz
    update_user_meta($user_id, 'first_name', $_POST['first_name']);  // sink
}
```
**Detect**: for each AJAX callback, grep its body for `current_user_can`.
If absent and the body writes data (DB query, `update_*`, `delete_*`,
`unlink`, `wp_insert_*`) → report `missing-authorization-<action>`.

## Privilege escalation (CWE-269)

Registration/profile handlers that set a **role or capability from
request input**:

```php
// VULNERABLE: attacker-chosen role
wp_insert_user(['user_login'=>$u, 'role'=>$_POST['role']]);     // → admin
$user->add_role($_POST['wholesalex_registration_role']);
update_user_meta($id, 'wp_capabilities', $_POST['caps']);
wp_update_user(['ID'=>$id, 'role'=>$_POST['role']]);
```
```
grep "wp_insert_user"  "wp_update_user"  "->add_role("  "set_role("
grep "update_user_meta" with "wp_capabilities" or a request-derived role
```
**Detect**: does the role/cap value trace back to `$_POST`/`$_REQUEST`
without an allowlist (`in_array($role, ['customer','subscriber'])`)? →
report `privilege-escalation-<handler>`.

## SQL injection (CWE-89) via $wpdb

`$wpdb->prepare()` is the safe path — but ONLY with correct placeholders
(`%s`, `%d`, `%f`). Two vulnerable shapes:

```php
// 1. No prepare — direct concatenation
$wpdb->get_results("SELECT * FROM tbl WHERE id=" . $_POST['id']);
$wpdb->query("... WHERE id=".$location_id);   // tainted local var

// 2. prepare() with an INVALID placeholder — silently unsafe
$wpdb->get_results($wpdb->prepare(
   "SELECT * FROM state WHERE country_id=%1s", $cid));  // %1s is NOT valid → no escaping
```
```
grep "$wpdb->query"  "$wpdb->get_results"  "$wpdb->get_var"  "$wpdb->get_row"
grep "%1s"  "%1$s"   # invalid/positional placeholders that defeat prepare()
```
**Detect**: tainted value concatenated into the SQL string, OR a
`prepare()` whose format string uses anything other than `%s`/`%d`/`%f`.
Note: `sanitize_text_field()` does NOT make a value SQL-safe.

## Arbitrary file deletion / upload (CWE-22 / CWE-434)

```php
// Deletion: path from request, no confinement
$path = wp_upload_dir()['path'] . '/' . $_POST['file'];
unlink($path);                       // ../ escapes the upload dir

// Procedural variant
extract($_POST);                     // imports $path from request!
unlink($path);

// Upload: extension allowlist taken FROM the request → bypassable
$uploader = new FileUploader($_REQUEST['allowedExtensions'], ...);
$uploader->handleUpload($dir);       // attacker sets allowedExtensions
move_uploaded_file($_FILES['x']['tmp_name'], $dest);
```
```
grep "unlink("  "wp_delete_file("  "@unlink("
grep "move_uploaded_file("  "$_FILES"  "extract($_POST"  "extract($_REQUEST"
```
**Detect**: the path / filename / allowed-type set traces to a request
value with no `realpath()` confinement or server-side allowlist.

## Stored / reflected XSS (CWE-79/80)

WordPress escapes on OUTPUT, not input. A finding is unescaped output of
attacker-influenced data:

```php
echo $_GET['q'];                                   // reflected
echo $row->loginname;                               // stored (logged input)
<?php echo $user_input; ?>                          // template
{{ td.data|raw }}                                   // Twig raw filter = no escaping
```
Safe sinks: `esc_html()`, `esc_attr()`, `esc_url()`, `wp_kses()`,
`esc_html_e()`. Their ABSENCE around dynamic output is the bug. Stored
XSS has two ends — the **storage** point (e.g. a failed-login logger
writing `$_POST['loginname']`) and the **output sink** (the admin view
rendering it unescaped). Report the output sink as primary.
```
grep "echo "  "print "  "<?= "  "|raw"   # then check for esc_* wrapping
```

## Scan order for PHP / WordPress

1. `glob "*.php" "*.twig" "*.html"` — inventory.
2. `grep "wp_ajax_nopriv_"` then `"wp_ajax_"` then `register_rest_route` —
   build the handler list; nopriv handlers first.
3. For each handler callback: read it, check `current_user_can` presence,
   then check its body for the sinks above (DB write, role assign, unlink,
   upload, echo).
4. Project-wide grep the sink patterns above for anything missed.
5. Report each CONFIRMED finding (you have read the sink and traced a
   request value to it) via `report_vulnerability`.
