# CWE Quick Reference

Load this reference **before** writing the `details` field of `report_vulnerability`.
Always include the primary CWE ID (e.g. `CWE-89`) in the `details` field.

## Shape A — Tainted Flow → Sink (structural control missing)

| Vulnerability Class              | Primary CWE | Sink Label           |
|----------------------------------|-------------|----------------------|
| SQL Injection                    | CWE-89      | db.query.raw         |
| NoSQL Injection                  | CWE-943     | nosql.query          |
| OS Command Injection             | CWE-78      | shell.exec           |
| SSTI (Server-Side Template Inj.) | CWE-1336    | template.render.raw  |
| XSS (Reflected / Stored)        | CWE-79      | response (unfiltered)|
| Path Traversal                   | CWE-22      | filesystem.read/write|
| Unrestricted File Upload         | CWE-434     | filesystem.write     |
| SSRF                             | CWE-918     | http.request         |
| Open Redirect                    | CWE-601     | http.redirect        |
| LDAP Injection                   | CWE-90      | ldap.query           |
| XPath Injection                  | CWE-643     | xpath.query          |
| Header Injection                 | CWE-113     | http.response.header |
| Insecure Deserialization         | CWE-502     | parser.pickle / parser.yaml.unsafe |
| XML External Entity (XXE)         | CWE-611     | parser.xml.unsafe    |
| Code Injection / eval            | CWE-94      | reflect.eval         |
| Prototype Pollution              | CWE-1321    | reflect.proto        |
| SMTP Injection                   | CWE-93      | smtp.send            |
| Log Injection                    | CWE-117     | log.write            |

## Shape B — Missing Control on Sensitive Operation

| Vulnerability Class              | Primary CWE | Control Missing      |
|----------------------------------|-------------|----------------------|
| Missing Authentication           | CWE-306     | auth                 |
| Missing Authorization / IDOR     | CWE-639     | authz / ownership_check |
| Broken Access Control            | CWE-284     | authz                |
| Missing CSRF Protection          | CWE-352     | csrf                 |
| Missing Rate Limiting            | CWE-307     | rate_limit           |
| Signature Not Verified           | CWE-347     | signature_verify     |
| Token Expiry Not Checked         | CWE-613     | expiry_check         |
| Privilege Escalation             | CWE-269     | role_check           |

## Shape C — Sensitive Value Without Protection

| Vulnerability Class              | Primary CWE |
|----------------------------------|-------------|
| Hardcoded Credentials / Secrets  | CWE-798     |
| Plaintext Password Storage       | CWE-256     |
| Weak Hashing (MD5/SHA1)          | CWE-328     |
| Sensitive Data in Response       | CWE-200     |
| Personal/Financial Data Exposure (PII/PAN) | CWE-359 |
| Missing Encryption of Sensitive Data (at rest) | CWE-311 |
| Verbose Error / Stack Trace      | CWE-209     |
| Missing Cookie Flags             | CWE-614     |
| Weak PRNG for Security           | CWE-330     |
| Cleartext Transmission           | CWE-319     |
| CORS Misconfiguration            | CWE-346     |

## Shape D — Business-Logic / Invariant Violation

| Vulnerability Class                       | Primary CWE | Control Missing  |
|-------------------------------------------|-------------|------------------|
| Race Condition / TOCTOU (check-then-act)  | CWE-362     | atomicity        |
| Replayable / Non-Idempotent Operation     | CWE-837     | idempotency      |
| Workflow / State-Machine Bypass           | CWE-841     | state_machine    |
| Client-Trusted Security-Critical Value    | CWE-602     | business_logic   |
| Unbounded / Negative Amount or Quantity   | CWE-20      | input_validation |
| Business Logic Error (general)            | CWE-840     | business_logic   |

## Details Field Template

```
CWE-{ID}: {vulnerability class}
shape: {A|B|C|D}
control_missing: {control name}
evidence_lines: {file}:{line}, ...
exploit_precondition: {e.g. unauthenticated, any authenticated user}
```
