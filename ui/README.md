# Contractor UI

This directory is an independently built React single-page application. The
Node process serves only the generated static files, `/runtime-config.json`,
and `/healthz`; it never proxies API requests or stores browser/model
credentials. The browser calls the configured Go Server origin directly.

The pinned toolchain is Node 24.20, Corepack 0.36 and pnpm 11.24. Install and
verify it from the repository root:

```shell
npm install --global corepack@0.36.0
make ui-verify
```

For frontend development, Vite serves source files on loopback. Supply a
matching runtime config through the production static service when testing the
built application:

```shell
corepack pnpm --dir ui build
CONTRACTOR_UI_API_BASE_URL=http://127.0.0.1:8080 \
  corepack pnpm --dir ui start
```

`CONTRACTOR_UI_API_BASE_URL` is required. It must be an origin-only HTTPS URL;
HTTP is accepted only for an IP-literal loopback address. Contractor Server
must list the UI origin (for example `http://127.0.0.1:4173`) in its exact
browser-origin allowlist. Production UI and API origins must also be HTTPS and
same-site so the Server's `SameSite=Lax` session cookie is eligible for direct
browser requests.

Optional static-service settings are:

- `CONTRACTOR_UI_HOST` (default `127.0.0.1`);
- `CONTRACTOR_UI_PORT` (default `4173`);
- `CONTRACTOR_UI_DIST_DIR` (default `ui/dist`).

The generated file at `src/api/generated/public.ts` comes only from
`api/openapi/contractor-public-v1.yaml`. Run `make ui-generate`; never edit the
file by hand. `make ui-generate-check` regenerates it and fails on a tracked
diff.

The `/artifacts` route manages UserScope Workflow inputs. Uploads are limited
to 16 MiB and use explicit create/CAS preconditions; the UI never retries a
PUT after a conflict or lost response. Version and lineage views retain exact
revisions. Inline preview is opt-in, capped at 256 KiB, restricted to a small
text media-type allowlist, and rendered as escaped text. Other payloads remain
available only through exact-revision download.
