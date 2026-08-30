# Public API contracts

`openapi/contractor-public-v1.yaml` is the single committed, already bundled
OpenAPI 3.1 contract for public HTTP endpoints. It deliberately has no external
`$ref`; the separately versioned WebSocket frame contract is linked as
`events/contractor-events-v1.schema.json` rather than embedded as an HTTP body.

Every operation carries `x-contractor-implementation`:

- `implemented` means the current Go Server exposes the operation and its
  fixture responses are checked against this contract;
- `planned` reserves the reviewed v1 shape but does not claim that the current
  Server routes it yet.

The distinction prevents generated clients from silently treating a design
placeholder as a deployed capability. A client must feature-detect through its
supported Server release/API compatibility policy until the marker changes to
`implemented`.

Within `/v1`, an additive change may introduce an optional request field, an
optional response field, or a new endpoint. Existing fields, statuses,
semantics, security requirements, and enum members cannot be removed, renamed,
narrowed, reinterpreted, or extended with a value that an exhaustive client
cannot handle. Changing a required field, making an optional field required,
changing a field type, or otherwise invalidating an existing conforming client
requires a new public API version and a separately versioned event subprotocol
when applicable.

Run `make verify-public-api` after either contract changes. The pinned Go module
validator checks OpenAPI structure, examples, formats and patterns, verifies
that the YAML is self-contained, validates the standalone Draft 2020-12 event
schema and examples, enforces the public/private and secret-free boundaries,
and validates every currently implemented handler's success and shared error
responses. No derived bundle is committed, so there is no second generated
contract that can drift from the YAML source of truth.
