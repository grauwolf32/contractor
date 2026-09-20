# Scan artifacts v1

This directory defines [HTTPRequestSet](http-request-set.schema.json) and
[ScanPlan](scan-plan.schema.json) structural schemas.

## HTTPRequestSet

`application/vnd.contractor.http-requests+json` is a bounded Artifact containing
neutral prepared HTTP requests. Its normative structural schema is
[http-request-set.schema.json](http-request-set.schema.json); the additional
semantic rules below are enforced by
[`internal/contracts/http_request_set.go`](../../../internal/contracts/http_request_set.go).
The [valid fixture](testdata/valid.json) is canonical JSON. Preparation policy,
including supported OpenAPI versions and coverage gap meanings, is specified in
[scan request preparation](../../../docs/spec/31-scan-request-preparation.md).

All fields shown in the schema are mandatory. Empty arrays and an empty request
body are present explicitly. Unknown, duplicate, differently cased and null
fields are rejected. JSON must contain one UTF-8 object; unpaired Unicode
surrogate escapes are rejected. `schemaVersion` is the integer `1`.

`source.artifact` is an exact ArtifactRef, including a nonblank revision.
`source.contentDigest` identifies the exact source bytes. `preparationDigest`
binds preparation policy, source identity and preparation options, as specified
by the preparation library. Both use `sha256:<64 lowercase hexadecimal digits>`.
The artifact validator checks their syntax; verifying them requires independent
access to the source bytes and preparation inputs.

Each request contains an uppercase supported method, absolute printable ASCII
HTTP(S) URL without credentials or fragment, lowercase HTTP header names sorted
lexicographically, and a UTF-8 text body. Header names are unique HTTP tokens;
values forbid C0 controls except horizontal tab and forbid DEL.
An explicit URL port must be 1 through 65,535. Percent escapes in both the path
and query must be well formed. Request bodies may be empty. Binary bodies are
outside this version's contract.

`contentDigest` is SHA-256 of RFC 8785 canonical JSON of **only** the four-field
request object. `id` is `request-` followed by the same 64 hexadecimal digits.
Validation verifies both against the request content. Requests are sorted by
unique ID; duplicate request content is represented once, retaining all origins.
Request identity excludes provenance, preparation options and coverage gaps.

Each origin identifies an OpenAPI operation with a JSON Pointer fragment such
as `#/paths/~1pets/get`. Origins are nonempty, unique and sorted per request;
the same operation cannot occur in multiple requests. Gaps contain a JSON
Pointer fragment and a bounded lowercase snake_case code. `#` denotes the
document root and is allowed for gaps. Gaps are sorted by pointer, then code,
with no duplicates. Pointers are at most 8,192 UTF-8 bytes.

Coverage counts all discovered operations, retained origins (`prepared`) and
their difference (`skipped`). `complete` is true exactly when there are no
skipped operations and no gaps. Empty input coverage may be complete, but an
invalid source document must fail preparation before producing an artifact.

Limits apply to UTF-8 **bytes**, not Unicode character counts. JSON Schema's
`maxLength` is a structural ceiling; the Go validator enforces byte ceilings,
aggregate limits, ordering, identity and coverage relationships:

| Item | Bound |
| --- | ---: |
| Complete JSON artifact | 4 MiB |
| Requests, operations, aggregate origins | 1,000 each |
| Gaps | 4,096 |
| URL | 8,192 bytes |
| Body | 64 KiB |
| Headers per request | 64 |
| Header name / value | 128 / 8,192 bytes |
| Sum of header name and value bytes | 32 KiB |

`DecodeHTTPRequestSet` validates untrusted bytes. `MarshalHTTPRequestSet`
validates a Go value and returns canonical bytes within the artifact bound.
`RequestContentDigest` validates one normalized request and computes its digest.
Validation never silently normalizes content.

This contract does not select injection points, bind credentials, authorize
network access or guarantee eligibility for any scanner. In particular it has
no SQLMap `testParameters`; `scan-plan@1` selects explicit parameters and
constructs the scanner-specific artifact, applying that adapter's validation.
Request bodies, headers and provenance can contain private data; consumers must
not log serialized artifacts or supplied values in diagnostics.

## ScanPlan

`application/vnd.contractor.scan-plan+json` contains the exact source ref and
content digest, normalized selection policy, preparation coverage/gaps, every
candidate and the bounded selected jobs. Its structural schema is
[scan-plan.schema.json](scan-plan.schema.json); cross-field identity, ordering,
provenance, scanner eligibility and budget checks belong to
[`internal/scanplan/plan_codec.go`](../../../internal/scanplan/plan_codec.go).

`BuildPlan` is pure. `MarshalPlan` validates Go values and emits canonical JSON;
`DecodePlan` validates strict input bytes. Plans are at most 4 MiB, with 4,000
candidates and 100 jobs. Time budgets sum configured Worker timeouts; they do
not bound the number of individual scanner network requests. Generated Artifact
revisions and physical allocations are excluded from semantic job identity.

[Scan planning](../../../docs/spec/32-scan-planning.md) owns the selection policy,
fixed Worker bindings, SQLMap/ffuf materialization, durable intent, aggregate
coverage and recovery behavior. Recovery preserves completed and unknown job
outcomes instead of silently rescanning them.
