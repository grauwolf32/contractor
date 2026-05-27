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
| SSRF                             | CWE-918     | http.request         |
| Open Redirect                    | CWE-601     | http.redirect        |
| LDAP Injection                   | CWE-90      | ldap.query           |
| XPath Injection                  | CWE-643     | xpath.query          |
| Header Injection                 | CWE-113     | http.response.header |
| Insecure Deserialization         | CWE-502     | parser.pickle / parser.yaml.unsafe |
| Code Injection / eval            | CWE-94      | reflect.eval         |
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
| Verbose Error / Stack Trace      | CWE-209     |
| Missing Cookie Flags             | CWE-614     |
| Weak PRNG for Security           | CWE-330     |
| Cleartext Transmission           | CWE-319     |

## Details Field Template

```
CWE-{ID}: {vulnerability class}
shape: {A|B|C}
control_missing: {control name}
evidence_lines: {file}:{line}, ...
exploit_precondition: {e.g. unauthenticated, any authenticated user}
```
