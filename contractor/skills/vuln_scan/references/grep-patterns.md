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
Confirm: does user input reach the command string?

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

## Scan order for maximum speed

1. `glob "*.py" "*.java" "*.go" "*.js" "*.ts"` — inventory source files
2. Grep CRITICAL patterns first (highest value per time)
3. For each hit file, list_symbols to understand structure
4. Read 20-30 lines around each hit to confirm
5. Report immediately, then move on
6. After structural scan, do auth audit pass
