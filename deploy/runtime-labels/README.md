# Runtime label deployment examples

[`runtime-configs.example.yaml`](runtime-configs.example.yaml) contains only
non-secret immutable documents. Publish one document at a time through
`POST /v1/operations/runtime-configs`, take the exact `ref` from the response,
then create `debug`, `agent-debug`, or `caido` through
`PUT /v1/operations/runtime-labels/{label}` with `If-None-Match: *`. Rebind with
the current quoted decimal `ETag` in `If-Match`; an active Run/allocation keeps
its already pinned configuration.

Create credential material directly from an operator secret source. Do not add
it to this directory. The public mutation accepts `otlp-headers@1`,
`http-proxy-basic@1`, or `http-proxy-bearer@1` and returns metadata only. Server
must start with its owner-only `--credential-master-key-file` before the first
Runtime credential is stored.

Each concurrently connected Runtime process needs a distinct CA-signed key and
certificate. Startup labels seed a previously unseen certificate principal;
later changes use the principal-label Operations endpoint. Runtime adapters
are all enabled by default. To expose a deliberate subset, repeat
`--runtime-adapter`, for example `--runtime-adapter http-proxy@1`. This narrows
the factory/probe surface and never makes an unavailable implementation pass.
