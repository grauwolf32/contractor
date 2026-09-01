---
name: auth
description: "Discover authentication flows for explicitly authorized live-target testing, including session setup and controlled two-user checks."
compatibility: "Contractor adk@1 native Agent Skill disclosure; requires authorized-target HTTP operations; session, memory, and code-execution steps are conditional on their visible tools"
metadata:
  source-revision: 9c76b56cf7b83377fb1dd5e4a17440fa27b723f3
---

# Authentication discovery

Use this when a target endpoint requires auth and the finding does **not**
already give you credentials or a login flow. If the finding *does* supply
auth instructions, use those exactly — skip this.

This skill provides guidance only. It does not grant network access, create an
HTTP client, enable code execution, or select another skill. Use a named
operation below only when that exact operation is visible in the current Worker
invocation; otherwise return the missing capability instead of pretending the
step ran.

## Safety and authority

1. **Authorization and exact target scope are preconditions.** Send no request
   until the task identifies an authorized scheme, host, port, and applicable
   account scope. Redirects and discovered sibling services must remain inside
   that scope.
2. **Use controlled test identities only.** Never reuse, reset, or modify a real
   user's credentials or data. Create accounts only when account creation is
   permitted by the engagement.
3. **Prefer non-destructive proof and stop once confirmed.** Do not lock out
   users, exhaust OTPs, create unbounded accounts, or retain access beyond the
   requested check.
4. **Clean up when the target supports it.** Remove test accounts, revoke test
   tokens, and report any state that could not be removed.

## Optional shared memory

When both `write_memory` and `read_memory` are visible, persist only controlled
test credentials, tokens, cookies, and working URLs that the authorized task
needs. Use valid logical note names; there is no reserved `auth/` directory or
implicit sandbox bridge:

```
write_memory(name="auth_creds", memory="email=e@t.com password=Testpass1! token=eyJ...")
write_memory(name="auth_endpoints", memory="signup=/identity/api/auth/signup login=/identity/api/auth/login")
```

Read them back with `read_memory(name="auth_creds")` before re-authenticating.
If MemoryTools are absent, keep the minimum necessary values in the current
response context and do not claim durable persistence.

## Discovery patterns

The auth endpoint may not share the target's URL prefix. Try these **in order,
stop at the first that returns 200**:

Signup:
```
POST <BASE_URL>/identity/api/auth/signup  {"name":"Test","email":"exploit1@test.com","number":"1234567890","password":"Testpass1!"}
POST <BASE_URL>/api/auth/signup           {"name":"Test","email":"exploit1@test.com","number":"1234567890","password":"Testpass1!"}
POST <BASE_URL>/api/v1/register           {"username":"exploituser1","password":"Testpass1!","email":"exploit1@test.com"}
POST <BASE_URL>/users/v1/register         {"username":"exploituser1","password":"Testpass1!","email":"exploit1@test.com"}
```

Login (same email/password as signup):
```
POST <BASE_URL>/identity/api/auth/login   {"email":"exploit1@test.com","password":"Testpass1!"}
POST <BASE_URL>/api/auth/login            {"email":"exploit1@test.com","password":"Testpass1!"}
POST <BASE_URL>/api/v1/login              {"username":"exploituser1","password":"Testpass1!"}
POST <BASE_URL>/users/v1/login            {"username":"exploituser1","password":"Testpass1!"}
```

Rules:
- If signup returns **200**, go straight to login. If **409** (already exists),
  go straight to login anyway — do NOT try a second signup pattern.

## Pin the credential to the session (decision tree)

Inspect the login response and branch on what it hands you. The examples below
require a visible `http_session_set` operation; otherwise maintain session state
only through the semantics promised by the available authorized HTTP tool.

- **Bearer token in JSON body** (common fields: `token`, `auth_token`,
  `access_token`, `jwt`) →
  `http_session_set(auth={"kind":"bearer","token":"<token>"})`.
- **Session cookie, no token in body** → do nothing special: the http session
  keeps its own cookie jar, so the cookie set by the login response is replayed
  automatically on subsequent requests. For **state-changing** requests
  (POST/PUT/DELETE) you usually also need a CSRF token — harvest it from the
  login/HTML response body or a `/csrf` (a.k.a. `/api/csrf-token`) endpoint and
  echo it back via the `X-CSRF-Token` header (or the matching hidden form
  field). Common cookie names: `session`, `sid`, `connect.sid`, `JSESSIONID`,
  `laravel_session`.
- **Separate `refresh_token`** → optionally save it as `auth_refresh`, and use it to
  mint a fresh `access_token` when the current one expires (POST the refresh
  token to `/token/refresh` / `/oauth/token` `grant_type=refresh_token`) instead
  of re-running the whole login.
- **API key** (returned once, or supplied by the finding) →
  `http_session_set(headers={"X-API-Key": "<key>"})` (header name varies:
  `Authorization: ApiKey ...`, `X-Api-Key`, `api_key` query param).

## Two-user setup (for IDOR)

Register a **second** user (`exploit2@test.com`), but stay logged in as user 1.
Note user 2's resource IDs, then try to access them with user 1's token.
When MemoryTools are visible, keep the controlled test credentials in
`auth_user1` and `auth_user2`.

## After you obtain a JWT — try to forge it

Decode the header + payload (base64url, no signature needed). Branch on `alg`:

**HS\* (HMAC):** the signature is keyed by a shared secret — crack it offline.
When an explicitly visible code-execution operation provides the required
library or executable, run `pyjwt`/`jwt_tool` against a small common-secrets list
(`secret`, `password`, `changeme`, `jwt`, plus the app name / host). On a hit,
re-sign with elevated claims: `admin:true`, `role:admin`, or a swapped
`sub` / `user_id` pointing at another account.

**RS\* (RSA):** no secret to crack — attack the verification logic instead:
- **alg:none** — set header `alg` to `none`/`None` and strip the signature.
- **RS256 → HS256 key confusion** — sign with HS256 using the server's RSA
  **public** key as the HMAC secret. Fetch the key from `/jwks.json`,
  `/.well-known/jwks.json`, or extract it from the TLS cert.
- **kid abuse** — path traversal `kid:"../../dev/null"` signed with an empty
  key, or SQLi in `kid`.
- **jku / jwk header injection** — point at an attacker-hosted key.
- **null-signature** (CVE-2020-28042) — empty signature segment.

Forge only through a visible operation whose description confirms the required
environment; do not infer a container, dependency, filesystem, or network
route. If `http_request` is visible, replay the minimum confirming request and
cite the elevated response. If `write_memory` is visible, persist a successful
controlled-test token as `auth_forged_jwt`.

## OAuth / OIDC targets

If login redirects to an `/authorize` or `/oauth/*` endpoint, or you find
`/.well-known/openid-configuration`, test the flow itself:
1. **redirect_uri** — set it to an attacker / open-redirect / `localhost.evil.com`
   host to capture the leaked `code`/`token`.
2. **state** — missing or non-validated `state` → callback CSRF / forced
   account linking.
3. **code replay** — reuse an authorization code a second time.
4. **scope tampering** — downgrade/alter `scope` to bypass `redirect_uri`
   filters.
5. **Referer leak** — token/code leaking in the `Referer` header after callback.

If `write_memory` is visible, persist discovered `client_id` / `redirect_uri`
as `auth_oauth`.

## When login is multi-step / MFA-gated

If login returns "2FA required" / a `verify`-OTP step instead of a token, try
these before giving up:
- **Force-browse past it** — hit the post-auth endpoint directly with whatever
  partial session/token the first step gave you; the 2FA step may not be
  enforced server-side.
- **Code-validation bypass** — submit `code:null`, `000000`, an empty string, or
  the code as an array (`code:[123456]` / `code[]=123456`) — weak comparisons
  accept these.
- **OTP replay** — replay a previously *used* OTP; if it isn't invalidated after
  use it's still valid.
- **Leaked code** — check the OTP-submit response body, headers, and any inlined
  JS for the expected code echoed back.
- **Response tampering** — flip a `verified:false`→`true` flag or a `401`→`200`
  status on the verify response (works when the client trusts it).
- **Throttling** — if none work, check whether the OTP endpoint rate-limits at
  all; record an unthrottled endpoint without exhausting the code space. Use an
  iterative code-execution operation only when it is visible, expressly in
  scope, and bounded by the engagement.

## If you can't get valid creds, try to bypass

When signup/login won't yield creds, attack the auth check itself:
- **SQLi auth bypass** in the username/password fields: `admin' --`,
  `' OR 1=1-- `, `admin'/*`, `" OR ""="`.
- **Auth-flag parameter modification** — if the response or a follow-up request
  carries flags like `authenticated=1`, `role=admin`, `isAdmin=true`, flip them.
- **Type juggling** — `{"password": true}`, NoSQL operator injection
  `{"password": {"$ne": null}}` / `{"username": {"$gt": ""}}`, or array coercion
  `password[]=`.
- **Predictable sessions** — if session IDs are sequential/guessable, mint or
  guess another user's session.

Use an iterative code-execution operation only when it is visible and the
authorized scope permits that extraction. If `write_memory` is visible,
persist a working controlled-test bypass as `auth_bypass`.
