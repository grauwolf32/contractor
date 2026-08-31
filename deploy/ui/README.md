# Independent UI service

Build from the repository root so the image can consume both `ui/` and the
committed public OpenAPI contract:

```shell
podman build -f deploy/ui/Containerfile -t contractor-ui .
podman run --rm \
  -p 127.0.0.1:4173:4173 \
  -e CONTRACTOR_UI_API_BASE_URL=http://127.0.0.1:8080 \
  contractor-ui
```

For a containerized Server, use its browser-reachable HTTPS origin rather than
the container DNS name. Node puts that public origin into the non-secret
runtime config and derives the CSP HTTP/WebSocket `connect-src` entries from
it. Node does not receive an API token, session cookie, CSRF token, password,
or LLM credential and has no route that proxies `/v1`.

`compose.example.yml` is intentionally a standalone example. Rebuilding or
restarting this service does not rebuild or restart Contractor Server. In a
TLS deployment, use distinct same-site hosts such as `ui.contractor.example`
and `api.contractor.example`: the Server session cookie remains host-only to
the API, while `SameSite=Lax` still permits the UI's direct credentialed API
requests. The Server's browser-origin allowlist must contain the exact UI
origin, including its scheme and non-default port.

An independent release check is deliberately simple: start a Run, rebuild and
restart only `contractor-ui`, reload the Run/Operations route and confirm the
existing Server session and live execution remain available. No Server image,
configuration, database migration or Runtime Agent restart is part of a UI
release.
