---
description: Ready-to-use grep patterns for rapid vulnerability discovery — organized by severity and vulnerability class. Run these greps first, then read to confirm.
---

# Grep Patterns for Rapid Discovery

Run these patterns early to find leads. Each hit needs confirmation
by reading 20-30 surrounding lines.

## CRITICAL — SQL Injection

```
grep "cursor.execute" "raw(" ".execute(" "text(" 
```
Confirm: is user input concatenated/formatted into the query?
False positive: parameterized queries (`cursor.execute(sql, [param])`)

For Python string concat into SQL:
```
grep "\" + " "f\"SELECT" "f\"INSERT" "f\"UPDATE" "f\"DELETE" ".format("
```
For Java:
```
grep "createQuery" "createNativeQuery" "Statement" "PreparedStatement"
```
Confirm: is the string built with concatenation, not parameters?

For Ruby/Rails:
```
grep ".where(\".*#{" "find_by_sql(" "#{" near ".order(\|.pluck(\|.group("
```
Confirm: is `#{...}` interpolation inside `where`/`order`/`pluck` instead of `?`/hash conditions?

## CRITICAL — Command Injection

Python:
```
grep "os.system" "subprocess.call" "subprocess.run" "subprocess.Popen" "eval("
```
Java:
```
grep "Runtime.getRuntime" "ProcessBuilder" "exec("
```
Go:
```
grep "exec.Command" "os/exec"
```
Ruby/Rails:
```
grep "system(" "exec(" "spawn(" "eval(" "Open3.\|IO.popen\|PTY.spawn" "`"
grep "constantize\|safe_constantize" ".send(\|.public_send(\|.__send__(\|.try("
```
Confirm: does user input reach the command string, or a method/class name (reflection RCE)?

## CRITICAL — Server-Side Template Injection (SSTI)

Python/Flask:
```
grep "render_template_string" ".from_string(" "Template(" "autoescape=False"
```
Go:
```
grep "text/template" "template.HTML("
```
Node:
```
grep "<%-" "compile(" near user input; handlebars/mustache with escape disabled
```
Confirm: does user input reach the template **string** itself (not just the data context)? Rendering user data into a precompiled template is safe; building the template source from user input is RCE.

## CRITICAL — Authentication Bypass

JWT none algorithm:
```
grep "PlainJWT" "alg.*none" "none.*alg" "Algorithm.NONE"
```
JWT without verification:
```
grep "WithoutValidation" "parse(" without "verify" nearby
grep "decode(" without "verify" nearby
```

## HIGH — SSRF

```
grep "requests.get" "requests.post" "http.request" "urlopen" "fetch("
```
Confirm: is the URL derived from user input?

## HIGH — Path Traversal

```
grep "os.path.join" "Path(" "open(" combined with request params
grep "send_file" "FileResponse" "download"
```
Confirm: is the file path derived from user input without confinement?

## HIGH — Missing Authentication

For Python/Django:
```
grep "def get\|def post\|def put\|def delete" in views.py
```
Then check: does each method have `@jwt_auth_required` / `@login_required`?

For Java/Spring:
```
grep "permitAll" in WebSecurityConfig
grep "@GetMapping\|@PostMapping" then check for @PreAuthorize
```

For Go:
```
grep "HandleFunc" then check for SetMiddlewareAuthentication wrapper
```

## HIGH — IDOR / Broken Access Control

```
grep "get(id)\|get(.*_id)\|findById\|objects.get(id" 
```
Confirm: after fetching by ID, is there an ownership check
(`if obj.user != request.user` or equivalent)?

## HIGH — NoSQL Injection

```
grep "bson.M\|bson.D\|\\$where\|\\$gt\|\\$ne\|\\$regex" 
```
Confirm: is the bson filter built from user input?

## MEDIUM — Mass Assignment

```
grep "request.data\|request.json\|request.body" near model save/create
```
Confirm: are extra fields (admin, role, balance) accepted?

## MEDIUM — Hardcoded Secrets

High-precision provider-key regexes (low false-positive — a hit is almost always a real key):
```
AWS access key:  \b((?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16})\b
GitHub PAT:      ghp_[0-9a-zA-Z]{36}                 github_pat_\w{82}
GCP/Firebase:    \bAIza[\w-]{35}\b
Slack bot:       xoxb-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*
Stripe:          \b(?:sk|rk)_(?:test|live|prod)_[a-zA-Z0-9]{10,99}\b
OpenAI:          sk-[A-Za-z0-9_-]{20,}T3BlbkFJ[A-Za-z0-9_-]{20,}
Anthropic:       sk-ant-api03-[a-zA-Z0-9_-]{93}AA
JWT:             \bey[a-zA-Z0-9]{17,}\.ey[a-zA-Z0-9/_-]{17,}\.[a-zA-Z0-9/_-]{10,}
Private key:     -----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY
```
Low-confidence fallback (catches generic literals but noisy — verify each hit):
```
grep "password.*=.*\"" "secret.*=.*\"" "api_key.*=.*\"" "SECRET_KEY.*=.*\""
```
Exclude: test files, example configs, environment variable reads

## MEDIUM — Weak Crypto

```
grep "md5\|sha1\|DES\|ECB\|RC4" in security-sensitive contexts
```
Confirm: used for passwords, tokens, or signatures (not checksums)?

## MEDIUM — User Enumeration

```
grep "not found\|not registered\|does not exist\|invalid user"
```
Confirm: different error messages for valid vs invalid users?

## MEDIUM — Information Disclosure

```
grep "debug.*True\|DEBUG.*True" "stack_trace\|traceback\|print_exc"
grep "password\|secret\|token\|api_key" in response serializers/models
```

## LOW — Missing Rate Limiting

```
grep "login\|signin\|otp\|verify\|reset.*password\|forgot.*password"
```
Then check: any rate limiting middleware applied?

## PHP / WordPress (see `php-wordpress` reference for full detail)

The patterns above are Python/Java/Go-shaped and miss most PHP/WP bugs.
For `*.php` targets, start here:

```
grep "wp_ajax_nopriv_"  "wp_ajax_"  "register_rest_route"   # entry points; nopriv = unauth
grep "$wpdb->query"  "$wpdb->get_results"  "$wpdb->get_var"  "$wpdb->get_row"   # SQLi sinks
grep "%1s"  "%1$s"                                           # invalid prepare() placeholder = unsafe
grep "wp_insert_user"  "wp_update_user"  "->add_role("  "update_user_meta"   # privilege escalation
grep "unlink("  "wp_delete_file("  "move_uploaded_file("  "extract($_POST"   # file deletion/upload
grep "echo "  "<?= "  "|raw"                                 # XSS output sinks (check for esc_* wrapping)
grep "$_GET\|$_POST\|$_REQUEST\|$_FILES"                     # taint sources
```
For EACH AJAX/REST handler, confirm a `current_user_can(...)` capability
check exists; its absence on a state-changing handler is a missing-
authorization / privilege-escalation finding (a nonce check alone is NOT
authorization). `sanitize_text_field()` is NOT SQL escaping.

## Scan order for maximum speed

1. `glob "*.py" "*.java" "*.go" "*.js" "*.ts" "*.php" "*.rb" "*.cs"` — inventory source files
2. Grep CRITICAL patterns first (highest value per time)
3. For each hit file, list_symbols to understand structure
4. Read 20-30 lines around each hit to confirm
5. Report immediately, then move on
6. After structural scan, do auth audit pass
