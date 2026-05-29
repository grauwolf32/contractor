---
description: Web application vulnerability testing playbooks for IDOR, SSRF, SSTI, and XXE. Each reference covers discovery, fingerprinting, exploitation payloads, bypass techniques, escalation paths, and remediation. Detail files live under `vulns/references/<class>.md` — read on demand.
---

# Web Vulnerability Testing Skill

Offensive testing checklists for four high-impact web vulnerability classes. Use this skill for authorized pentests, bug bounty work in scope, CTFs, and defensive review of the same surfaces.

## Operating Rules

1. **Authorization is a precondition.** Only apply these techniques against targets where you have explicit permission (signed scope, in-scope bug-bounty asset, owned lab/CTF). If scope is unclear, ask before sending payloads.
2. **Non-destructive PoCs.** Demonstrate impact with the lightest possible artifact: read `/etc/hostname`, run `id`, write a uniquely-named marker (`touch poc-<handle>.txt`). Never mutate other users' data, drop tables, exhaust quotas, or persist payloads beyond the test window.
3. **Stand up an OOB listener before blind testing.** Blind SSRF / blind XXE / blind SSTI are invisible without Burp Collaborator, Interactsh, or your own DNS+HTTP canary. Start the listener first; reuse one unique subdomain per finding so logs are attributable.
4. **Fingerprint before throwing engine-specific payloads.** For SSTI and XXE, polyglot first (`${{<%[%'"}}%\`, generic file-read DTD), identify the engine/parser, then load the targeted payload. Wrong-engine payloads waste signal and pollute logs.
5. **Stop probing once confirmed.** When a vector is proven, capture the request/response, document, and move on. Don't escalate into exfiltration or lateral movement unless that is explicitly in scope.
6. **Prefer read over write, GET over PUT/DELETE, lower IDs over higher.** Test horizontally (peer accounts) before vertically (admin), and always with accounts you control on both sides.

## Cross-Cutting Methodology

Most hunts in this skill share the same skeleton — load the per-class reference for vector-specific payloads at each step.

```text
1. Recon         enumerate endpoints, parameters, content types, file-upload sinks, API versions
2. Setup         multiple accounts (peer + privileged), intercepting proxy, OOB listener
3. Discovery     minimal/polyglot payload at every input; diff against baseline
4. Identify      fingerprint engine / parser / auth model / id format
5. Exploit       targeted payload to confirm impact (file read, id swap, OOB callback)
6. Bypass        if blocked, iterate on encoding, IP form, parameter shape, method, content-type
7. Escalate      chain only as scope allows (SSRF→IMDS→creds, SSTI→RCE, XXE→SSRF, IDOR→stored XSS)
8. Report        reproducible PoC, blast radius, fix recommendation
```

## Quick Triage

Pick the reference by the signal you observe in the target:

| Signal | Load |
| --- | --- |
| Object IDs in URL/body/cookies, ownership-bound resources, GraphQL `id:` args, mass-assignment-prone JSON, presigned URLs | `vulns/references/idor` |
| URL/host/webhook input, "fetch from URL", PDF/screenshot export, allow/deny-listed domains, internal IP filtering | `vulns/references/ssrf` |
| Reflected input that *evaluates* (e.g. `{{7*7}}` → `49`), template engine errors in stack traces | `vulns/references/ssti` |
| `application/xml`, SOAP envelope, SAML AuthnRequest, file upload accepting DOCX / SVG / EPUB / XLSX, JSON→XML conversion | `vulns/references/xxe` |

If two signals coexist (very common: SSRF + XXE, IDOR + SSTI), load both — chains are where these classes deliver the highest impact.

## Output Discipline

- For "give me a payload" prompts, output **one** payload that targets the identified engine/parser; only offer alternatives when explicitly asked or when fingerprint is genuinely ambiguous.
- Quote payloads verbatim — do **not** "clean up" entity references (`&#x25;`), backslash escapes, or hex-encoded segments. Those are load-bearing.
- When a payload assumes a Python subclass index (e.g. `__subclasses__()[40]`), say so — index varies between CPython versions and applications.
- For remediation answers, prefer concrete config snippets (parser flags, framework settings) over abstract advice.

## Reference Index

| File | Load when working on... |
| --- | --- |
| `vulns/references/idor` | Authorization tampering, object-id enumeration, GUID/UUID prediction, GraphQL/gRPC ID swaps, mass assignment, OAuth `state`/`code`, MFA endpoints, presigned-URL abuse, OPA/Cedar policy fuzzing |
| `vulns/references/ssrf` | URL fetchers, cloud metadata (AWS IMDSv2, GCP, Azure, Alibaba, Oracle, ECS, OpenStack), IP-format/encoding bypass, DNS rebinding, gopher/dict/file/jar protocols, K8s + service-mesh pivots |
| `vulns/references/ssti` | Engine fingerprint (Jinja2, Twig, FreeMarker, Pebble, Velocity, Mako, Thymeleaf, Razor, Liquid, Nunjucks, Handlebars, Blade, Groovy/GSP), sandbox escape, prototype-chain RCE in Node templates, CVEs 2024–2025 |
| `vulns/references/xxe` | XML parser exposed, SVG/DOCX/EPUB/SAML/XLSX upload, OOB exfil via external DTD, XInclude, CDATA bypass, PHP `php://filter` base64, K8s admission webhooks, CI/CD XML artifact parsing, parser hardening per language |
