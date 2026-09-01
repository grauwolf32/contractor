# 11 — HTTP and Caido tools

Status: **Working agreement**

This document owns the Runtime tool contracts used for HTTP exploration and
Caido-assisted security workflows. The implementation is domain-specific, but
the architectural boundary is the same as for every Worker Toolset:

- AgentTemplate selects exact model-visible tools;
- Runtime labels select infrastructure routing and credentials;
- AllocationSpec carries resolved secret-bearing settings privately;
- a Runtime adapter creates allocation-scoped clients and erases them on
  release;
- the model never chooses an infrastructure endpoint, proxy credential or
  Caido bearer token.

The implementation ports useful behavior from `contractor-old`; it does not
preserve old unbounded responses, secret-bearing session artifacts or direct
environment-based infrastructure configuration.

## Two distinct facilities

`http-tools@1` sends model-described HTTP requests to application targets. It
can use the existing optional `http-proxy@1` adapter's `tool-http` route (for
example a Caido forward proxy) or a Runtime-owned direct transport when no such
route is resolved.

`caido@1` calls the trusted Caido control GraphQL API. That endpoint is not an
HTTP forward proxy and must not be represented by `httpProxy.proxyUrl`.
`caido-graphql@1` is a separate typed Runtime adapter.

```text
model -> http_request(target URL) -> optional tool-http forward proxy -> target

model -> caido_history/static operation -> caido-graphql@1 handle
                                      -> configured Caido /graphql endpoint
```

## Label-driven Caido configuration

RuntimeConfig gains one atomic Worker field:

```json
{
  "apiVersion": "contractor/v1alpha1",
  "kind": "RuntimeConfig",
  "metadata": {"name": "caido-lab", "version": "1"},
  "spec": {
    "worker": {
      "caido": {
        "adapter": "caido-graphql@1",
        "endpoint": "https://caido.internal:8443",
        "credential": "caido-lab-token",
        "caBundlePem": "-----BEGIN CERTIFICATE-----...",
        "requestTimeoutSeconds": 30
      }
    }
  }
}
```

The field follows existing atomic label merge rules. A higher layer replaces
the complete value or clears it with JSON `null`; fields from two configs are
never mixed. Default plus Run labels are resolved first and immutable Runtime
Agent labels remain the highest layer.

`endpoint` is an absolute `http` or `https` origin without userinfo, query or
fragment. Runtime appends exactly `/graphql`; a deployment may include a fixed
path prefix in the endpoint. HTTPS uses system trust plus optional bounded PEM
CA bundle. `requestTimeoutSeconds` is 1..120 and defaults to the allocation
request timeout.

`credential` is an optional ID of credential kind `caido-bearer@1`. Absence
means explicitly unauthenticated/guest access. The resolved token is a
`SecretString` only in private `RuntimeSettingsV2`; provenance carries the
credential ID/kind, never bytes.

`caido-graphql@1` advertises one private infrastructure handle
`caido-graphql-client`. Every `caido@1` tool requires that channel. Placement
must reject a selected Caido Toolset when effective labels do not resolve the
adapter and must choose an Agent advertising the adapter. Runtime repeats the
check during prepare. The adapter uses `trust_env=false` and is not implicitly
routed through `http-proxy@1`, avoiding proxy loops.

The adapter owns connection pooling, bounded request/response reads, bearer
header injection, TLS context, safe metrics and close/credential erasure. It
exposes a narrow `execute(operation, variables)` client where `operation` is
one implementation-defined static operation, never arbitrary model GraphQL.

## Generic HTTP Toolset

`http-tools@1` exports:

| Tool | Purpose |
|---|---|
| `http_request` | Send one bounded HTTP/HTTPS request and return metadata plus a small preview. |
| `http_read_body` | Read a bounded slice of a body stored as an ordinary Run artifact. |
| `http_history` | Return bounded allocation-session request summaries, oldest first. |
| `http_session_set` | Set allocation-memory cookies/default headers/auth. |
| `http_session_get` | Return the redacted session view. |
| `http_session_clear` | Erase cookies/headers/auth/history; retained response artifacts remain immutable. |

### Request contract

`http_request` accepts:

- `url`: absolute `http` or `https` URL, no userinfo or fragment;
- `method`: `GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS`;
- at most 64 bounded headers and query keys;
- `body_type`: `none|json|form|text`, with at most 1 MiB encoded request body;
- timeout 1..120 seconds, capped by allocation settings;
- `follow_redirects`, with at most 10 redirects and scheme validation at each
  hop.

Hop-by-hop headers, `Host`, `Content-Length`, proxy authentication and CR/LF
header injection are rejected. The Artifact API and Runtime private origins are
always denied. Other egress is intentionally the deployment's responsibility:
this Toolset exists to contact model-selected application targets. A resolved
`tool-http` proxy route is mandatory routing, not a hint; failure never falls
back to direct network.

Retries are bounded to idempotent methods by default and cover transport
failure plus `408`, `425`, `429`, `500`, `502`, `503`, `504`. A non-idempotent
request is not retried unless a future explicit idempotency contract is added.
The target's 4xx/5xx response is a valid response record, not an adapter
failure. This requires the proxy handle to expose response status rather than
collapsing it into transport failure.

The response record contains request ID/tag, method, final URL, status,
content type/length, safe response headers, body kind, at most 8192 characters
of text preview, exact body ArtifactRef when non-empty, truncation and elapsed
milliseconds. Authorization, cookies, set-cookie and proxy headers are removed
from returned headers and retained metrics.

The complete body is streamed with a 16 MiB hard limit into an ordinary
artifact in the allocation's logical namespace using media type
`application/vnd.contractor.http-body+json`. Text is stored as UTF-8 text;
arbitrary bytes use base64. Internal names are collision-resistant and the
exact returned ref is retained in allocation memory. `http_read_body` accepts
only a request ID created by that allocation, never an arbitrary ArtifactRef.

Session cookies/default headers/auth live only in allocation memory. They are
available to sequential A2A tasks on the same allocation and are erased on
release/abort/lease loss. In particular bearer/basic secret values are not
written to an artifact. `http_session_get` returns only `auth_kind`, redacted
sensitive headers and cookie names/count; cookie values are not model-visible
after being set.

History contains at most 128 summaries. IDs are monotonic for the allocation.
Tool calls serialize session mutation and request ID assignment; network waits
do not permit a second call to reuse state or an ID.

## Caido Toolset

`caido@1` retains these model-visible names:

| Tool | Static Caido operations |
|---|---|
| `caido_scope` | list scopes or create one bounded allow/deny scope |
| `caido_history` | query paginated proxy history using bounded HTTPQL text |
| `caido_request_detail` | read request/response metadata and bounded raw previews/artifacts |
| `caido_replay` | replay a stored request or one bounded supplied raw request |
| `caido_automate_run` | create/configure/start bounded Automate fuzzing |
| `caido_automate_results` | page one Automate result set |
| `caido_sitemap` | browse root or descendants with bounded depth/result count |
| `caido_workflow_list` | list Caido workflows, optionally by known kind |
| `caido_workflow_run` | run one convert/active workflow with bounded input |
| `caido_workflow_findings` | page findings created by workflows |

Every GraphQL document is a source-code constant selected by the tool method.
The model supplies variables only. Variable objects are strictly built and
bounded; unknown sort/strategy/kind/depth values fail before a request.

Common first limits:

| Resource | Limit |
|---|---:|
| one GraphQL response | 16 MiB |
| raw request supplied to replay/workflow | 1 MiB |
| inline raw preview | 8192 characters |
| history/results/findings page | 100 entries |
| scope allowlist plus denylist | 256 terms |
| HTTPQL/filter text | 8192 UTF-8 bytes |
| Automate targets | 32 |
| Automate payloads | 1000 and 1 MiB total |
| one payload/target | 8192 UTF-8 bytes |
| Automate workers | 1..50 |
| poll/wait | at most allocation deadline and 60 seconds |

Full raw Caido request/response and large workflow output are stored with the
same allocation Artifact client using reserved collision-resistant bindings;
tool results contain previews and exact refs. The old behavior of returning an
unbounded raw exchange inline is deliberately removed.

Replay requests receive an opaque allocation-derived `X-Request-Id` tag. HTTP
tool and Caido replay counters use distinct infixes so proxy history can
correlate traffic without exposing Run/Stage IDs. Tags are safe observability
identifiers, not authentication.

Known Caido domain failures become bounded tool results. Transport, JSON,
GraphQL and schema errors return a stable code and retryability without echoing
endpoint, bearer token, raw query, variables or arbitrary server text.

## Lifecycle, authority and metrics

Prepare order is adapter host first, selected Toolsets second, Worker last.
`caido@1` creation without its typed handle fails `caido_not_configured` and
does not create a ready Worker. HTTP direct/proxied clients and Caido sessions
are closed on finalize/release through their owning Toolset/adapter; secrets are
overwritten/dereferenced before the slot becomes idle.

AgentTemplate selection is the model authority boundary. Merely attaching a
`caido` or `debug` Run label configures infrastructure but adds no tools. A
Caido mutation is permitted only when its exact tool name is selected. No Skill
can add a tool that AgentTemplate omitted.

Metrics may retain tool/operation name, outcome, duration, status class,
request/response byte count, retry count and safe error code. They must not
retain URLs, query/filter text, headers, cookies, credentials, raw bodies,
payloads, findings or GraphQL data. Existing Toolset call counters and Runtime
adapter metrics remain the only durable metrics planes.

Stable errors include:

- `http_request_invalid`, `http_target_denied`, `http_request_failed`,
  `http_response_too_large`, `http_body_not_found`;
- `caido_not_configured`, `caido_request_invalid`, `caido_request_failed`,
  `caido_response_invalid`, `caido_response_too_large`.

## Initial acceptance

1. Strict Go/Python RuntimeConfig, private settings, credential and provenance
   fixtures round-trip `caido-graphql@1`; all secret values remain redacted.
2. Label precedence can retarget Caido for a Run or Runtime Agent without
   changing AgentTemplate or Runtime startup environment.
3. A selected Caido Toolset cannot reserve/prepare on an Agent lacking the
   adapter or when effective configuration omits it.
4. Generic HTTP direct and mandatory-proxy paths produce equivalent bounded
   records, preserve target 4xx/5xx, never proxy private Contractor traffic and
   never fall back after proxy failure.
5. Session auth is erased at allocation teardown and absent from artifacts,
   metrics, logs, reports and tool views.
6. Every Caido tool passes against a deterministic fake GraphQL server with
   exact variables, pagination/bounds, mutation and error fixtures.
7. The migrated `configs/skills/caido` package is assigned only to an
   AgentTemplate that selects its required Caido tools; native Skill loading
   introduces no additional execution authority.
8. Real process tests cover two Runtime Agents where only one advertises Caido,
   label-based placement, response loss/release, and subsequent clean slot
   reuse.

## Deliberately deferred

- arbitrary GraphQL queries or user-defined Caido schema extensions;
- automatic PAT-to-access-token exchange and refresh;
- Server-side HTTP allow/deny policy editor, DNS pinning or network sandbox;
- streaming multi-gigabyte downloads and binary artifact chunking;
- automatic Skill-to-Toolset dependency installation;
- UI-specific Caido panels or HTTP history browser;
- reconciling Caido findings into a separate Contractor domain model.

## Invariants

1. Forward proxy and Caido control API are different typed configurations.
2. Labels configure infrastructure but never grant model-visible tools.
3. Infrastructure endpoint/credential values never enter prompts or tool
   arguments.
4. Generic HTTP response status is data; transport/routing failure is an error.
5. Caido GraphQL is static-operation-only and bounded at every input/output.
6. Allocation teardown erases all session credentials and client handles.
