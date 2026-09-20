# Shared toolset configuration fixtures

`cases.json` is consumed by Go and Python contract tests. Valid values are JCS
canonical bytes. `schemaValid` distinguishes JSON Schema expressible shape
checks from additional shared invariants (case-insensitive headers, reserved
headers and total byte limits). All invalid values must be rejected by both
language models. Origins are a Server projection, schema-checked in Python.
The recognizable secret is test data; validation/formatting must redact it.
