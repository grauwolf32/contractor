# Independent priority verdict v1

`verdict.schema.json` describes the six required model response fields. The
`internal/auditpriority` decoder is the authoritative executable boundary.
This directory does not register a Planner, public API, or model execution path.

The valid fixtures are canonical JSON plus a terminal newline for text files.
Invalid fixtures exercise structural constraints shared by the JSON Schema and
Go validator. Go tests also exercise constraints outside JSON Schema's data
model: duplicate JSON object keys, malformed Unicode, trailing input, raw bytes,
and item/evidence identity from the caller's exact frozen context.

JSON Schema `maxLength` counts Unicode code points. `x-maxUtf8Bytes` documents the
additional UTF-8 byte limit enforced by Go; `x-maxEncodedBytes` documents the
8 KiB raw response and canonical encoding limits. These annotations are not
standard JSON Schema assertions. No silent truncation or Unicode repair occurs.
Legitimate U+FFFD characters are accepted; malformed UTF-8 and lone UTF-16
surrogate escapes are rejected before decoding can replace them.

Evidence IDs and missing-context entries must be unique. Their arrays must be
present and non-null, including when empty. Text must contain a non-whitespace
character and cannot contain NUL. IDs use ASCII letters/digits initially and
then ASCII letters/digits, dots, underscores, colons or hyphens, up to 160 bytes.
The caller's complete evidence-ID set is unique and limited to 100 entries;
each verdict may refer to at most 16 of them.

`MarshalVerdict` validates structure and canonical size only. Acceptance requires
`ValidateVerdict` or `DecodeVerdict` with the assigned item key and context IDs.
All rejection diagnostics are the closed `priority_invalid_verdict` code and
never contain model/context text.
