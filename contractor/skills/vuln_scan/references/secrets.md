---
description: Secrets & sensitive-data exposure for the vuln_scan skill. Two halves — (1) secrets at rest in code/repo (mostly covered by grep), and (2) the runtime exposure sweep (secrets/credentials/PII returned, logged, or shipped to clients) which scanners miss because nothing matches a sink grep. Run the sweep on every handler that returns, logs, or persists data.
---

# Secrets & Sensitive-Data Exposure

Two halves. The first (secrets at rest in source/repo) is mostly a grep
problem and is well covered elsewhere — use those, don't re-derive them. The
second (runtime exposure) is the commonly-missed half: a perfectly
authorized, validated handler can still hand secrets/PII to the wrong place.

## A. Secrets at rest (use the existing patterns)

- **Hardcoded provider keys** — high-precision regexes (AWS/GitHub/GCP/Slack/
  Stripe/OpenAI/Anthropic/JWT/private-key): see `miss-patterns` §9 and
  `grep-patterns` "Hardcoded Secrets". Any hit is almost certainly live.
- **Weak/guessable secrets** — `SECRET_KEY="secret"|"changeme"|"dev"`, short
  JWT signing keys: `grep-patterns` "Hardcoded Secrets".
- **Sensitive files committed to the repo** (CWE-538/CWE-312) — `.env*`,
  `*.pem`, `*.key`, `*.sql`, `*.bak`, `wp-config.php`, `.aws/credentials`,
  …: see `miss-patterns` §11 and the index workflow step 3.
- **Plaintext credential storage** (CWE-256) — passwords without bcrypt/
  argon2/scrypt; `==` comparison: see `miss-patterns` §4.

## B. Runtime exposure sweep (the missed half)

Run this on every handler that **returns, logs, or persists** data — even
when auth/authz/validation are all present.

### B1. Returned to the client → CWE-200 / CWE-359

Enumerate the fields *actually serialized* — follow the model / struct /
dict, not the variable name. `return user`, `jsonify(row)`, serializing a
whole ORM object, or `**obj.__dict__` leaks **every** field it has.

For each returned field ask: is it a password/hash, API key, token, secret,
internal flag, audit/owner id, **full PAN / card number / CVV**, SSN, or
email/phone (PII)? Any sensitive field with no explicit projection/DTO that
excludes it → finding. A sibling endpoint that DOES project the same object
confirms the gap.
```
grep -nE "password|secret|token|api_?key|ssn|card|cvv|pan" <serializers/models/response builders>
```
Then check whether the response path filters it. **Type of finding:**
sensitive data in response (CWE-200); PII/PAN specifically → CWE-359.

### B2. Written to logs / metrics / traces → CWE-532

Secrets, credentials, tokens, full request bodies, or PII written to logs.
```
grep -nE "log(ger)?\.(info|debug|warning|error).*\b(password|token|secret|authorization|card|cvv|ssn|api_?key)\b"
grep -nE "print\(.*(password|token|secret|card)"
```
Also: logging the raw request/headers (`log.info(request.json)` /
`log.debug(headers)`) leaks Authorization tokens and credentials.

### B3. Shipped to the browser / client config → CWE-200

Secrets embedded in client-delivered code or config: server-only API keys
placed in front-end bundles, templates, `window.__CONFIG__`, `/config`
endpoints returning private keys, or service-account JSON served statically.
```
grep -rnE "(secret|private_?key|service_account|api_?key).*(=|:)" <static/, templates/, public/, *.js>
```

### B4. Cleartext at rest (beyond passwords) → CWE-311 / CWE-312

Card numbers / PAN, bank/account numbers, SSNs, OTP/reset codes, or session/
refresh tokens stored unencrypted/unmasked, or transmitted over plain HTTP.
```
grep -nE "(card_?number|pan|iban|account_?number|ssn|cvv)\s*=\s*[^h]"   # stored raw
grep -nE "verify\s*=\s*False|InsecureRequestWarning|http://"            # cleartext transit / TLS off
```

### B5. Weak/none token signing → CWE-347 / CWE-330

JWT `alg: none` / unverified decode, HS256 with a guessable secret, or
security tokens (reset/OTP/session) built from non-CSPRNG randomness
(`random.random`, `Math.random`, `mt_rand`, time-based). See `grep-patterns`
"Authentication Bypass" (JWT) and "Weak Crypto".

## Reporting

- CWE: sensitive data in response CWE-200 · PII/PAN exposure CWE-359 · secret
  in logs CWE-532 · cleartext storage CWE-312 / missing encryption CWE-311 ·
  hardcoded secret CWE-798 · plaintext password CWE-256 · weak PRNG for
  security CWE-330 · signature not verified CWE-347.
- Report against the **source line** that exposes/stores the value (the
  serializer/log call/model), not a config file that merely names a key.
- Severity: cleartext credentials/PAN, or a server secret shipped to clients,
  is **high**; an over-exposed field in an authenticated-only response is
  **medium**; a low-sensitivity value logged is **low**.
- A `*_test_*` / `sk_test_…` / obvious placeholder (`XXXX`, `your-key-here`)
  is low/info, not high.
