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
`http-proxy-basic@1`, `http-proxy-bearer@1`, or `caido-bearer@1` and returns
metadata only. Server must start with its owner-only
`--credential-master-key-file` before the first Runtime credential is stored.

For the `caido-analysis@1` example, create the referenced write-only credential
from an environment value and then clear the shell variable:

```sh
jq -n --arg token "$CAIDO_ACCESS_TOKEN" \
  '{credentialId:"caido-api",kind:"caido-bearer@1",material:{token:$token}}' |
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H 'Content-Type: application/json' \
    --data-binary @- \
    http://127.0.0.1:8080/v1/operations/runtime-credentials
unset CAIDO_ACCESS_TOKEN
```

The example forward route is unauthenticated. If the deployment requires proxy
authentication, create a separate `http-proxy-basic@1` or
`http-proxy-bearer@1` credential and add only its ID to `httpProxy.credential`.
The `caido-api` credential cannot be reused for that different credential kind.

Each concurrently connected Runtime process needs a distinct CA-signed key and
certificate. Startup labels seed a previously unseen certificate principal;
later changes use the principal-label Operations endpoint. Runtime adapters
are all enabled by default. To expose a deliberate subset, repeat
`--runtime-adapter`; a Runtime eligible for `security-analysis@1` needs both
`--runtime-adapter http-proxy@1` and `--runtime-adapter caido-graphql@1` when an
explicit subset is used. This narrows the factory/probe surface and never makes
an unavailable implementation pass.
