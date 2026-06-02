---
description: Commonly missed vulnerability patterns — absence-based and configuration-level issues that code scanners frequently overlook. Check each pattern explicitly.
---

# Commonly Missed Vulnerability Patterns

These are vulnerability classes that automated scanners and even manual reviewers commonly miss. Check each pattern explicitly during your scan.

## 1. Unauthenticated sensitive endpoints

Many apps have endpoints that should require auth but don't:
- User listing / search endpoints returning emails, usernames
- Debug / diagnostic endpoints (`/_debug`, `/health` with sensitive data)
- Database seed / reset endpoints (`/createdb`, `/reset`)
- Admin endpoints without auth middleware

**How to check:** For each route, verify authentication is enforced. Read the middleware chain or decorator stack, not just the handler body.

## 2. Hardcoded credentials in seed/init functions

Seed functions (`init_db`, `populate_db`, `create_default_users`) often contain hardcoded user/password pairs that persist in production:
```python
User(username="admin", password="admin123")  # ships to production
```

**How to check:** Read all database initialization, migration, and seed functions. Look for literal password strings.

## 3. Missing rate limiting on auth endpoints

Login, registration, password reset, and OTP verification endpoints without rate limiting enable brute-force attacks. This is a finding even when no explicit rate-limit library is imported.

**How to check:** Search for rate-limiting middleware (Flask-Limiter, django-ratelimit, slowapi). If absent project-wide, report for auth-sensitive endpoints.

## 4. Plaintext password storage

Passwords stored without hashing (no bcrypt/argon2/scrypt). Look for:
- `password = db.Column(db.String(...))` with no hash at write time
- `self.password = password` in `__init__` (raw assignment)
- `user.password == input_password` (plaintext comparison)

## 5. Validation error detail leakage

Exception handlers that return internal validation errors verbatim:
```python
except ValidationError as e:
    return {"error": str(e)}, 400  # leaks schema internals
```
This reveals field names, types, and validation rules to attackers.

## 6. Differential error messages (user enumeration)

Login endpoints returning different messages for "user not found" vs "wrong password" enable username enumeration.

## 7. IDOR via path parameter without ownership check

Endpoints using URL path parameters (`/users/{id}/...`) to identify resources but checking auth without checking that the authenticated user owns the resource:
```python
user = User.query.get(user_id)  # user_id from URL, not from session
```

## 8. Mass assignment / extra fields in registration

Registration accepting unexpected fields that control privilege:
```python
if "admin" in request.json:
    user.admin = request.json["admin"]
```

## 9. Weak/guessable secrets

- `SECRET_KEY = "secret"` or `"random"` or `"changeme"`
- JWT signing keys that are short strings, not cryptographic keys
- API keys hardcoded in source

A generic `password = "..."` grep is a noisy low-confidence fallback. Sweep with the high-precision provider-key regexes for a near-zero-false-positive pass — any hit is almost certainly a live credential:

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

## 10. Debug mode in production code

`app.run(debug=True)` or `DEBUG=True` in settings. Enables interactive debugger, detailed stack traces, auto-reload — all dangerous in production.

## Reporting discipline

- Always report against the **source file** containing the vulnerable code, not spec files or config files that merely describe the endpoint
- Use the most specific CWE for the vulnerability class:
  - Missing auth → CWE-306
  - IDOR → CWE-639
  - SQLi → CWE-89
  - Plaintext passwords → CWE-256
  - Hardcoded credentials → CWE-798
  - Missing rate limit → CWE-770
  - User enumeration → CWE-204
  - Error detail leakage → CWE-209
  - Debug mode → CWE-489
  - Mass assignment → CWE-915
  - ReDoS → CWE-1333
- Do NOT report the same logical vulnerability twice with different CWEs. Pick the primary CWE.
