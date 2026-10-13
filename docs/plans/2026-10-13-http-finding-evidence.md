# HTTP materials selected by findings

Status: user-selected implementation scope, recorded on 2026-10-13. This is a
non-normative plan. Runtime finding-only retention is implemented; the UI
preview task is next.

## Agreed user outcome

1. An HTTP exchange participating in a successfully recorded finding becomes
   ordinary durable evidence. This includes unreviewed possible issues;
   confirmation by a human is not a prerequisite for retaining evidence.
2. The saved exchange always contains the actual outgoing request: method,
   URL, ordered headers and body. A received response contains its status,
   headers and available body. An absent or incomplete response has an explicit
   outcome; it is never fabricated or presented as a successful response.
3. `finding(...)` selects and attaches evidence through the existing finding
   intake and exact-reference mechanism. Selecting evidence never resends
   traffic. Selected HTTP artifacts remain available after buffer eviction and
   allocation release under the existing finding retention rules.
4. The recent shared HTTP/Caido Runtime buffer keeps at most 128 exchanges
   and 64 MiB of serialized captures, evicting the oldest on either bound.
   Tool reads and selection operate within its documented ownership and size
   bounds. Expired or foreign IDs fail explicitly.
5. Other proxied traffic is inspected in Caido. Caido history is external to
   the Run evidence contract; direct HTTP traffic appears there only when
   routed through Caido. This work does not change routing or promise that all
   requests pass through Caido.
6. The UI previews the selected request and available response, supports
   downloading the saved exchange and identifies its relationship to a finding.
   HTML is shown as source; binary bodies are identified and downloadable.

Full traffic export is recorded separately in
[proposal 0036](../proposals/0036-compressed-http-traffic-export.md). Complete
traffic retention, a ZIP download and independent bookmarking are outside the
current implementation scope.

## Gaps identified before implementation

- `http_request` currently writes a durable `http.body.*` artifact for every
  non-empty final response. Only outgoing requests are kept in the recent
  exchange buffer. Finding-only retention must cover response bodies too.
- `finding(request_id=...)` currently embeds the selected outgoing snapshot
  in the proposal and pins an existing response-body artifact. The selected
  exchange must become an ordinary exact evidence artifact usable by the UI.
- `caido_request_detail` and completed `caido_replay` currently write an
  exchange artifact during inspection. Inspection alone must not publish
  ordinary HTTP evidence under the new scope. The implementation must preserve
  bounded previews and an explicit way to select that captured source in the
  existing `finding` tool.
- Caido IDs and Runtime request IDs are different identities. Their source
  must be explicit; neither matching URLs nor request order establishes a
  relationship. The Runtime task defines the selection contract before changing
  tool schemas and updates all applicable finding facades and fixtures.
- The generic artifact UI currently lacks HTTP body and Caido exchange
  renderers. It must render selected HTTP evidence and fetch body bytes only
  for the selected preview or download, without scanning the traffic history.

## Implementation tasks

| Task | Outcome |
| --- | --- |
| [V376-001](../../tasks/done/v376-001-http-evidence-scope.yml) | Recorded this scope, preserved the deferred export proposal and prepared implementation tasks. |
| [V376-002](../../tasks/done/v376-002-finding-http-evidence.yml) | Implement finding-only HTTP/Caido evidence, recent response-body retention, exact references, failure handling and the owning specification changes. |
| [V376-003](../../tasks/v376-003-http-evidence-preview.yml) | Show and download selected request/response evidence in the UI with bounded reads and truthful finding links. |

The reference HTTP, finding, artifact and platform contracts change together
with their implementation. Historical plans remain historical. No compatibility
reader or reconstruction of missing requests from earlier Run artifacts is
required.
