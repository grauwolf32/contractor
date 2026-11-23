# Optional structured finding analysis

A finding can include an ordinary evidence document with media type
`application/vnd.contractor.finding-analysis+json`. The document describes
exact source locations, explicit paths and selected HTTP evidence. It does
not confirm a vulnerability or change Audit review.

Publish source and supporting evidence first. Author analysis with their exact
artifact revisions, then pass all those refs and the analysis ref to the
existing `finding` tool. Its `client_key` must equal the finding's client key.
The text Artifact writer validates the closed authoring shape and stores
canonical JSON for this media type. Other producers must publish canonical
RFC 8785 JSON themselves.

A minimal analysis without code locations is:

```json
{
  "schema": "contractor.finding.analysis.v1",
  "client_key": "candidate-1",
  "sources": [],
  "locations": [],
  "flows": [],
  "http_exchanges": []
}
```

For code, add an explicit `archive` or `file` source and locations addressing
that exact source. `file_digest` hashes file bytes, not the containing ZIP.
Paths are relative POSIX paths. Lines and Unicode code-point columns are
one-based; end columns are exclusive. Select `primary_location_id` explicitly.
After editing or inserting annotations, retain the edited source and use its
coordinates, or keep the original source and original coordinates. The
resolver does not guess offsets between snapshots.

Keep call reachability distinct from a data-flow claim. Every neighboring pair
of path steps needs an explicit edge or gap. Missing dispatch, argument binding
or physical coordinates must stay visible as a partial path; a sanitizer label
is a claim, not proof of safety. Copied tool output and annotation comments use
`producer-asserted` basis. Durable trusted code-relation capture is not supported.

A saved HTTP/Caido exchange uses `contractor-http-exchange/v1`; its exact
artifact must also be selected evidence. A response-body artifact uses
`http-response-body/v1` and cannot supply a missing request or status. The
projection omits query values, credentials, headers and bodies. Original
exchanges remain ordinary access-controlled evidence.

Collection publication resolves all dependencies against the selected receipt.
Malformed analysis, missing refs, a mismatched digest or invalid coordinates
rejects the entire publication without a partial output. See
[the owning contract](../spec/findings/findings.md#optional-typed-analysis-evidence)
for the complete shape, flow rules and limits.
