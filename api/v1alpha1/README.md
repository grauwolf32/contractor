# Contractor private wire contracts v1alpha1

These JSON Schema documents describe Contractor-owned payloads exchanged
between the Go Server and Python Runtime Agent. Every standalone request and
response carries `"apiVersion": "contractor/v1alpha1"`, uses camelCase fields,
and rejects unknown fields. Reusable resolved values such as
`ResolvedLLMGatewayConfig` are embedded components rather than standalone
messages, so their version is carried by their exact digest-bearing ref.

A2A envelopes themselves are owned by the official A2A 1.0 SDKs. Contractor
places `StageContentRequest` and `StageContentResult` in an A2A DataPart with
media type `application/vnd.contractor.stage-content+json`.

The schemas are review artifacts and compatibility contracts. Go and Python
DTOs are maintained explicitly and are checked against shared golden fixtures
under `api/testdata/v1alpha1`.

`AllocationSpec.agentTemplate.modelPolicy` remains the immutable template
default covered by the AgentTemplate digest. The separate required
`AllocationSpec.modelPolicy` is the effective Worker policy pinned by the Run;
Runtime validates its own digest and uses it without mutating or re-signing the
AgentTemplate. `RuntimeSettings.llmGatewayToken` may be the empty string for an
explicitly unauthenticated Gateway.

`llm-gateway-config-manifest.schema.json` describes the strict non-secret YAML
manifest after YAML-to-JSON conversion. `llm-gateway-config.schema.json`
describes its normalized, digest-bearing resolved value on private wires.
