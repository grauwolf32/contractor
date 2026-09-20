# OpenAPI request preparation

`Prepare(sourceBytes, mediaType, exactArtifactRef, options)` is a pure library
entry point. Its caller obtains the source through the existing Artifact client,
then persists `contracts.MarshalHTTPRequestSet(result)` as
`contracts.HTTPRequestSetMediaType`. The package does not read files, resolve
remote references, call scanners or dispatch Workflows.

See the runnable [example](example_test.go), the
[normative preparation policy](../../docs/spec/31-scan-request-preparation.md)
and [RequestSet schema](../../api/scan/v1/http-request-set.schema.json).

An operation binding uses its exact source pointer:

```go
options := scanplan.Options{
    Server: "https://api.example.test/v1",
    Authentication: map[string]contracts.SecretString{
        "bearerAuth": contracts.NewSecretString(suppliedToken),
    },
    Operations: map[string]scanplan.OperationInput{
        "#/paths/~1pets~1{id}/get": {
            Parameters: map[string]any{"path:id": 7},
        },
        "#/paths/~1pets/post": {
            Body: &scanplan.BodyInput{
                MediaType: "application/json",
                Value: map[string]any{"name": "Milo"},
            },
        },
    },
    MaxRequests: 100,
}
```

Credentials must bind declared security scheme names. Unresolved requirements
produce explicit gaps. Go strings must contain valid UTF-8; body and parameter
values must be plain JSON-compatible values, not custom marshalers or byte
slices with implicit base64 encoding. Do not log prepared artifacts or bindings:
they contain the actual request data. Errors expose fixed codes; source pointers
and gap codes identify preparation problems without echoing supplied values.

Inspect `coverage` and `gaps` even when preparation returns no error. Whole-input
validation and resource failures return `*PreparationError`; a malformed input
does not become an empty successful result. `coverage.complete` describes the
bounded one-example-per-operation policy. It does not establish scan coverage or
absence of vulnerabilities. Selecting scanner parameters and dispatching the
result belong to V55-008.

Run `go test ./internal/scanplan/... ./internal/contracts/...` for codec, parser,
determinism, preparation, reference, authentication and limit checks.
