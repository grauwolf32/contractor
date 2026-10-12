// Package auditprofile is the AuditProfile model and its rules: modes,
// inventories, Workflow roles and their mappings, execution limits and
// interaction policies. Resolve turns an authored spec into the immutable
// ResolvedAuditProfile an Audit retains, and DecodeResolvedAuditProfileSnapshot
// checks a retained one again without the catalog.
//
// The catalog loader in internal/config decodes AuditProfile manifests and
// calls Resolve; the Audit packages read retained profiles through this
// package and not through the loader. The JSON encoding of
// ResolvedAuditProfile is the persisted profile snapshot that SQL reads
// through the members named in this package.
package auditprofile
