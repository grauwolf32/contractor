---
description: Sink catalogue and per-sink vulnerability checklist for the trace skill. Walk the matching row when a sink is reached on the traced path. Reaching a sink alone is NOT a vulnerability.
---

# Sinks Catalogue & Per-Sink Vulnerability Checklist

Label a sink ONLY when the function directly performs it or clearly wraps
it. Do not infer sinks from name alone.

## Allowed sink labels

  filesystem.read | filesystem.write | filesystem.delete | filesystem.path.join
  db.query | db.exec | db.query.raw | db.exec.raw | db.orm.bulk
  shell.exec | shell.exec.args | process.env.write
  http.request | http.redirect | dns.lookup | socket.connect | smtp.send
  template.render | template.render.raw | html.render | pdf.render
  parser.process | parser.yaml.unsafe | parser.pickle |
    serializer.encode | serializer.decode
  cache.read | cache.write | queue.publish | queue.consume
  crypto.key.derive | crypto.random.seed | crypto.sign | secret.log
  auth.token.create | auth.token.verify | authz.policy.eval |
    auth.password.check | auth.password.hash
  reflect.eval | reflect.import | reflect.attr
  log.write | metric.record | audit.write
  cookie.set
  ldap.query | nosql.query | xpath.query

## Label precision (pick the more specific label when both apply)

Mislabeling here is the #1 source of false-positive Shape A findings —
parameterized ORM is NOT raw SQL.

  db.query.raw         not db.query           — string-built SQL
  db.exec.raw          not db.exec            — string-built SQL exec
  shell.exec           not shell.exec.args    — shell=True / single command string
  template.render.raw  not template.render    — unescaped/auto-escape disabled
  parser.yaml.unsafe   not parser.process     — yaml.load (no safe_load)
  parser.pickle        not serializer.decode  — pickle.loads specifically
  secret.log           not log.write          — secret/credential/PII written
  http.redirect        not http.request       — Location: header / 30x

If the parameterized form is clearly used (placeholders bound by the
driver), do NOT label it as `*.raw`.

## Quick FP filter

Before labeling a sink, answer:

1. Is this the function that performs the side effect, or a direct
   wrapper around it?
2. Is the sink kind visible from code, not inferred from the function name?
3. Are attacker-influenced values controlling structure, or only bound
   as values?
4. Is there a control that blocks exploitation on this path?
5. Is the label the most specific accurate label?

Examples:

- A wrapper with a name like `execute_query(sql, params)` that always
  delegates to the parameterized driver call → label `db.query`, even
  if the wrapper's name suggests something raw.
- A method that accepts an SQL string but interpolates the caller's
  values into it before calling the driver → `db.query.raw`.
- `subprocess.run([...argv], shell=False)` is `shell.exec.args` — the
  argv-list form is not the OS-command-injection mechanic.
- `subprocess.run("git show " + ref, shell=True)` → `shell.exec`.
- `os.path.join(safe_root, untrusted)` without a normalize+prefix-check
  is still `filesystem.path.join` with attacker-controlled segment —
  `..` traversal applies.
- `yaml.safe_load(x)` → not `parser.yaml.unsafe`. `yaml.load(x)`
  without safe loader → `parser.yaml.unsafe`.

## Shape A structural-control rule

For Shape A, tainted/derived input must control STRUCTURE, not just
appear as a bound value.

Structural control includes:

- filesystem path segment or full path
- SQL table, column, operator, clause, raw predicate, or query string
- shell command string, shell metacharacters, executable name, or
  argument with dangerous command semantics
- outbound URL, host, scheme, or IP
- redirect target
- template name, template source, or unescaped HTML context
- deserializer input
- reflection target, import path, attribute name, expression
- NoSQL/LDAP/XPath query structure
- email header structure

NOT structural by itself:

- value bound through SQL placeholders
- JSON response field value
- safe template variable with auto-escape
- argv list element when no shell is invoked, unless the target
  executable interprets it as dangerous structure
- user value stored as data with no immediate sensitive use

## Auth pair rule (read before reporting any token defect)

`auth.token.create` and `auth.token.verify` MUST be analyzed as a pair.
Annotating only the create side and reporting "no signature" is invalid
unless you have also opened the verify path and confirmed the signature
is never compared. Common bug: payload is decoded (looks like verify)
but the signature is never read or compared — that IS the defect, but
you must show the verify-side code to claim it.

## Per-sink vulnerability checklist

For each sink encountered on the traced path, walk its row. A "no required
control" answer is a candidate finding.

### filesystem.* (read / write / delete / path.join)
- path attacker-controlled (tainted/derived)?
- destination confined (allowlist, normalize+prefix-check, no `..`)?
- for write/delete: content/target attacker-controlled?
→ Path Traversal / Arbitrary Read|Write|Delete.

### db.query.raw / db.exec.raw
- tainted data inside concat / f-string / %-format / text()-substitution?
- controls QUERY STRUCTURE (not just parameter values)?
→ SQL Injection. Parameterized ORM (db.query, db.exec) is NOT this sink.

### shell.exec / shell.exec.args
- shell=True or tainted token in command string?
- allowlist or exec-with-args (no shell)?
→ OS Command Injection / RCE.

### http.request
- URL/host attacker-controlled? Outbound allowlist?
- internal/metadata addresses blocked?
→ SSRF.

### http.redirect
- target URL attacker-controlled and validated against allowlist?
→ Open Redirect.

### template.render.raw / html.render
- tainted data rendered without auto-escape or explicit escape?
- template selected from tainted input?
- render call visible in THIS service's traced path?
→ SSTI / XSS. Data persisted to a store and rendered elsewhere is out of
scope unless the render path is also traced.

### parser.pickle / parser.yaml.unsafe / serializer.decode
- input attacker-controlled? Deserializer safe (yaml.safe_load, JSON)?
→ Insecure Deserialization / RCE.

### reflect.eval / reflect.import / reflect.attr / db.orm.bulk
- symbol / expression / field-set attacker-controlled?
- allowlist of permitted keys?
→ RCE / Mass Assignment.

### auth.token.create AND auth.token.verify — CHECK AS A PAIR
- real cryptographic signature on create (HMAC/JWT lib)?
- verify actually reads AND compares the signature?
  (Common bug: payload decoded but signature never checked.)
- verify checks exp / nbf / iss / aud where present?
- signing key hardcoded? Algorithm pinned (no `none`)?
→ Token Forgery / Privilege Escalation / Replay.

### auth.password.hash / auth.password.check
- slow KDF (bcrypt/argon2/scrypt) with salt? Constant-time compare?
- plaintext storage or `===` comparison?
→ Credential Compromise.

### authz.policy.eval (or absence)
- subject identity tied to the resource being accessed?
- ownership / RBAC enforced on this handler?
→ Broken Access Control / IDOR.

### response with domain object
- object passed through a sanitizer/projection?
- sensitive fields (password, apiKey, token, paymentMemo) excluded?
- sibling handler returning the same object DOES sanitize but this one
  does not → Shape C finding.
→ Sensitive Data Exposure.

### cookie.set on session/auth
- HttpOnly, Secure, SameSite all set?
→ Session Fixation / Theft.

### crypto.random.seed / crypto.key.derive
- CSPRNG used for security-relevant values? Adequate key length?
→ Weak Randomness / Weak Key Material.

### log.write / secret.log
- secrets, credentials, or PII written to logs?
→ Sensitive Data Exposure.

### ldap.query / nosql.query / xpath.query / smtp.send
- tainted data interpolated into query / header structure?
→ LDAP / NoSQL / XPath Injection / Email Header Injection.
