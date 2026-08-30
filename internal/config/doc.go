// Package config loads Contractor's operator-owned YAML manifests and text
// resources into a validated, dependency-resolved snapshot.
//
// Loading is deliberately all-or-nothing. Callers receive a Snapshot only
// after every LLMGatewayConfig, ModelPolicy, AgentTemplate, Workflow,
// descriptor reference, and instruction resource has passed validation.
package config
