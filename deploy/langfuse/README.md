# Langfuse through generic OTLP

Server and Runtime use the same `otlp-http@1` exporter for Langfuse and
[Phoenix](../phoenix/README.md). They emit OpenTelemetry GenAI messages, system
instructions, model and usage attributes. Backend mapping belongs to the
collector; the exporters contain no Langfuse SDK, backend selector or
`langfuse.*` attributes.

## Connect an existing Langfuse

Create an Operations Runtime credential with ID `local-langfuse-otlp`, kind
`otlp-headers@1`, and this secret material:

```json
{
  "headers": {
    "Authorization": "Basic <base64(public-key:secret-key)>",
    "x-langfuse-ingestion-version": "4"
  }
}
```

Use your Langfuse project's public and secret keys. Keep the generated Basic
authorization in the credential store. The v4 header selects direct ingestion
so new observations appear in real time.

Publish [`runtime-config.example.yaml`](runtime-config.example.yaml), bind it
to a Runtime label, then select that label for a new Run as described in the
[Runtime label guide](../runtime-labels/README.md). Change the endpoint to your
actual Langfuse host; the complete trace path is `/api/public/otel/v1/traces`.
The loopback example assumes both Server and Runtime run on the host. A
containerized process needs an address reachable from its container.

Set `captureContent: true` to capture prompts and responses. Existing sensitive
tool suppression and redaction remain in force. Existing Runs retain their
pinned configuration.

OTLP/JSON decoding ignores unknown message fields, as required by the
[OTLP specification](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding).
A successful HTTP 200 response containing queue receipt metadata such as
`{"name":"otel-ingestion-job","id":"..."}` therefore succeeds without
matching that name. Known `partialSuccess` fields, malformed payloads,
response-size limits and failure status codes retain their validation. An
acknowledgement means acceptance by the receiving service, not durable backend
storage; the integration check below also reads the stored observations.

## Verify against a disposable backend

Requirements: Podman with a working connection, Go 1.26 on `PATH`, and the
locked Runtime environment (`cd runtime && uv sync --locked`). On macOS, start
the Podman machine first. Run from the repository root:

```sh
CONTAINER_CONNECTION=podman-machine-default-root sh deploy/langfuse/test-smoke.sh
```

The test pins **Langfuse 4.67.0** and all dependency images by digest. It starts
a separate pod with PostgreSQL, ClickHouse, Redis and SeaweedFS for S3. Only
Langfuse HTTP is published, on a random loopback port. All test state lives in
a temporary directory under `.local/`; containers and data are removed on
exit. It does not use the project's database, modify running services or
RuntimeConfig bindings, or call a model. This test stack is heavier than the
one-container Phoenix profile and is intended for interoperability checks.

The production Go and Python exporters must each report successful delivery.
The test then uses Langfuse's public Observations API v2 to verify generation
classification, model, system and user input, assistant output, and input,
output and cache-read token counts. Langfuse splits cached input out of its
`usageDetails.input`; the check verifies that split without double counting.
The shared emitter lives in [`../telemetry/emit.py`](../telemetry/emit.py) and
is also used by the Phoenix smoke test.

Sources: [native OTLP endpoint and mapping](https://langfuse.com/integrations/native/opentelemetry),
[v4 ingestion](https://langfuse.com/integrations/native/opentelemetry/migration-to-v4),
[official deployment](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml).
