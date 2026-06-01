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
- Extract the token from the login response JSON (common field names:
  `token`, `auth_token`, `access_token`, `jwt`) and set it:
  `http_session_set(auth={"kind":"bearer","token":"<token>"})`.

## Two-user setup (for IDOR)

Register a **second** user (`exploit2@test.com`), but stay logged in as user 1.
Note user 2's resource IDs, then try to access them with user 1's token.
Keep both users' creds in memory under `auth/user1` and `auth/user2`.
