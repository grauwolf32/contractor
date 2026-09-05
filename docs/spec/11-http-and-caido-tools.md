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
never mixed. Default plus Run-selected Runtime labels are resolved first and
immutable Runtime Agent labels remain the highest layer.

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
- timeout 1..120 seconds, capped by allocation settings; when omitted, it is
  `min(allocation request timeout, 120 seconds)`;
- `follow_redirects`, with at most 10 redirects and scheme validation at each
  hop.

The initial callable shape intentionally remains compatible with the migrated
Skills: `http_request(url, method, headers, query, body, body_type, timeout,
follow_redirects)`. `http_session_set(cookies, headers, auth,
replace_cookies, replace_headers)` applies sparse updates; `auth` is exactly one
of `{"kind":"none"}`, `{"kind":"bearer","token":"..."}` or
`{"kind":"basic","username":"...","password":"..."}`. The redacted
session view contains only `auth_kind`, non-sensitive default-header values,
redaction markers for sensitive headers, cookie names/count and history count.

Hop-by-hop headers, `Host`, `Content-Length`, proxy authentication and CR/LF
header injection are rejected. Exact configured Contractor infrastructure
origins (LLM Gateway, Artifact API, telemetry collector, forward proxy and
Caido control API), `localhost` names and loopback/link-local/unspecified IP
literals are always denied. Other egress is intentionally the deployment's
responsibility: this Toolset exists to contact model-selected application
targets. A resolved `tool-http` proxy route is mandatory routing, not a hint;
failure never falls back to direct network.

### Egress and DNS boundary

The preceding checks are application-layer defense in depth, not a network
sandbox. The Runtime does not resolve a hostname before every call, pin its
addresses, reject RFC1918/ULA results, or prove that two names do not reach the
same service. Consequently DNS rebinding and a public hostname resolving to a
private address are outside the Toolset's guarantee. Every redirect is parsed
and checked again, and allocation auth/cookies are stripped on a cross-origin
hop, but the same DNS boundary applies to that new hostname.

When a deployment needs a closed target policy, it must assign a mandatory
`tool-http` route and enforce DNS/address/allowlist policy at that forward
proxy, and/or restrict the Runtime's network namespace. When no route is
resolved, the Runtime intentionally has the OS identity's direct egress. Both
paths use normal TLS certificate and endpoint-name verification; neither
supports an agent-selected `verify=false`. The proxy endpoint itself is
Control-Plane configuration and cannot be changed by a tool argument.

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
The JSON envelope itself must also fit the Artifact plane's 16 MiB payload
limit, so base64 expansion can make the effective binary-body limit smaller.
Bindings use the reserved `http.body.` prefix: generic model-visible Artifact
tools neither list nor read/write them. This is a Toolset authority boundary,
not a new Artifact API or storage class.

`http_read_body(request_id, offset, length)` returns at most 8192 text
characters or binary bytes. Offsets are characters for UTF-8 text and bytes
for binary bodies. `http_history(limit)` returns at most 128 summaries in
request-ID order and omits previews and body content.

Session cookies/default headers/auth live only in allocation memory. They are
available to sequential A2A tasks on the same allocation and are erased on
release/abort/lease loss. In particular bearer/basic secret values are not
written to an artifact. `http_session_get` returns only `auth_kind`, redacted
sensitive headers and cookie names/count; cookie values are not model-visible
after being set.

History contains at most 128 summaries. IDs are monotonic for the allocation.
Tool calls serialize session mutation and request ID assignment; network waits
do not permit a second call to reuse state or an ID. Sparse header/cookie/auth
updates are validated as one prospective state before mutation. Once reserved,
an ID is never reused after validation failure, transport ambiguity,
cancellation or an Artifact write whose response is lost. A body blob that was
committed before cancellation but never selected into the session remains an
unreferenced reserved binding; it cannot be recovered through
`http_read_body`, and ordinary Artifact tools cannot enumerate its prefix.
The httpx transport cookie jar is scratch state: allocation-owned cookies are
the authoritative copy and stale transport cookies are cleared before reuse.

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

### Caido compatibility boundary

The first version defines compatibility by GraphQL operation shape, not by a
claimed Caido semantic-version range. Runtime performs no introspection,
arbitrary query fallback or startup schema negotiation. A configured Caido
endpoint is compatible when it accepts the checked-in operation documents,
field selections, input/enumeration values and exact response identities used
by the selected tools. Missing or renamed fields, changed nullability, an
unknown response member or another schema drift becomes a bounded
`caido_response_invalid`/`caido_request_failed`; arbitrary GraphQL/server text
is discarded. It does not trigger a less constrained query.

This makes a Caido upgrade an explicit compatibility event. Deterministic
fixtures for all static operations and the repository compatibility gate must
pass before the supported deployment is updated. If a future version needs a
different document, it receives a reviewed adapter/tool revision rather than
runtime-generated GraphQL. Adapter capability discovery proves only that the
local implementation can construct the typed client; it deliberately does not
contact or introspect an infrastructure instance during Agent registration.

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

For request detail, the first implementation validates and decodes both Caido
Blobs before one mutation, then stores request plus response in a single
`application/vnd.contractor.caido-exchange+json` artifact. One atomic exchange
binding avoids a half-published pair if the second write were to fail. Text is
UTF-8 and arbitrary bytes are base64 inside the envelope; each returned preview
is at most 8192 characters (or 6144 binary bytes encoded as base64). A missing
request/session is an explicit bounded `not_found` domain result, while a
malformed partial response is `caido_response_invalid` and produces no selected
artifact.

Replay requests receive an opaque allocation-derived `X-Request-Id` tag. HTTP
tool and Caido replay counters use distinct infixes so proxy history can
correlate traffic without exposing Run/Stage IDs. Tags are safe observability
identifiers, not authentication.

`caido_scope(action="create")` requires a bounded non-empty name and accepts
at most 256 non-empty allow/deny terms in total. `caido_replay` accepts exactly
one source: an existing request ID, or UTF-8 raw request plus host/port/TLS.
Before the first mutation it replaces any existing `X-Request-Id` header in
the exact request bytes and consumes a fresh tag; the tag is not reused after
cancellation or an ambiguous response loss. Replay polling may return only
`completed`, `failed`, `timeout`, `started`, `rejected`, or `not_found` as
supported by the chosen path. `timeout` describes only the bounded local
observation window: it neither claims that Caido stopped nor starts a cleanup
mutation.

`caido_automate_run` reads one existing request, injects its own fresh tag,
then computes every placeholder against the exact tagged UTF-8 bytes submitted
to Caido. It rejects missing, duplicate, or overlapping targets. One simple
payload list may contain at most 1000 values and 1 MiB total; empty and control
character payload strings remain valid fuzz inputs. Strategy is exactly
`SEQUENTIAL|PARALLEL|MATRIX|ALL`, workers are 1..50, delay is 0..60000 ms,
redirects are disabled and Caido request retries are configured to zero.

`caido_workflow_run` dispatches by mutually exclusive input: `request_id`
starts an active workflow asynchronously, while non-empty UTF-8 `input` runs a
convert workflow. A convert result is returned inline only when it is bounded
UTF-8; a large text or any binary result is written to one exact reserved
`caido.output.*` artifact with a bounded preview in the tool result.

No Caido mutation is automatically retried. In particular, a transport error
after send has an unknown remote outcome and is returned as a non-retryable
bounded failure for that model-visible action;
the Worker may inspect Caido state before deciding whether another explicit
tool call is appropriate. Read-only polling queries also use no transport
retry in the first implementation. Every returned object has an exact
operation-specific shape, and identities returned for an ID-addressed query or
mutation must match the requested/session identity before they are exposed.

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

The checked-in `caido_analyst@1` template is the initial closed assignment. It
selects all ten `caido@1` operations, the bounded `http_request`,
`http_read_body` and `http_history` operations, text artifact read/write, and
the versionless `skills/caido` ref. `security-analysis@2` uses that template in
one passthrough Stage; `http_explorer@1` is the reusable HTTP-only template and
selects all six `http-tools@1` operations without a Caido Skill or adapter
requirement. Their Workflow, template, instructions and Skill contain no
endpoint, proxy route or credential setting. Deployment binds those separately
through RuntimeConfig labels.

This association is configuration, not Skill metadata semantics. Runtime does
not parse guidance to install Toolsets and a label does not make a Skill or tool
visible. A repository compatibility gate scans model-visible operation names in
the bundled Caido package and requires them in the exact checked-in template;
changing either side requires a reviewed config/Skill revision and a new
canonical package digest.

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
7. The migrated `configs/skills/caido` package is assigned only to
   `caido_analyst@1`, whose exact Toolset allowlist includes every HTTP/Caido
   operation named by the package; removing one fails the repository release
   gate and native Skill loading introduces no additional execution authority.
8. Real process tests cover two Runtime Agents where only one advertises Caido,
   label-based placement, response loss/release, and subsequent clean slot
   reuse.
9. A strict executable hardening matrix owns every input/response bound,
   proxy/redirect boundary, cancellation/lifecycle fault and retained-secret
   assertion; `make release-verify` composes that matrix with the real-process
   gate.

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
7. Application URL checks do not claim DNS or private-network isolation; a
   deployment requiring it uses mandatory proxy/network policy.
8. Reserved HTTP/Caido IDs and tags are monotonic and never reused after an
   ambiguous or cancelled operation.
