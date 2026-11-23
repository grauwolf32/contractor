// Package config loads Contractor's operator-owned YAML manifests and text
// resources into a validated, dependency-resolved snapshot, and owns the
// managed publication and discovery of that catalog.
//
// Loading is deliberately all-or-nothing. Callers receive a Snapshot only
// after every LLMGatewayConfig, ModelPolicy, AgentTemplate, Workflow,
// descriptor reference, and instruction resource has passed validation.
//
// The values a Snapshot hands out are defined in internal/config/resolved
// (the resolved Workflow model and the code descriptor registry) and in
// internal/config/auditprofile (AuditProfiles). Only the packages that load,
// hold or publish the catalog import this package (TestOnlyCatalogOwnersImportTheLoader).
package config
