// Package artifactpolicy defines shared purpose-specific Artifact binding rules.
package artifactpolicy

import "strings"

const (
	MemoryArtifactPrefix               = "memory."
	AuditManagedProjectNamespacePrefix = "audit-"
	AuditStandardCatalogNamespace      = "audit-standards"
	FindingProposalNamespace           = "finding-proposals"
	// RunSystemNamespace is owned by trusted Control Plane services. Its
	// bindings are not part of the Worker or owner-facing Artifact library.
	RunSystemNamespace        = "contractor-system"
	RunRepeatRequestName      = "repeat-request"
	RunRepeatRequestMediaType = "application/vnd.contractor.run-repeat-request+json"
)

func IsAuditManagedProjectNamespace(namespace string) bool {
	return strings.HasPrefix(namespace, AuditManagedProjectNamespacePrefix)
}

func IsAuditStandardCatalogNamespace(namespace string) bool {
	return strings.HasPrefix(namespace, AuditStandardCatalogNamespace)
}

// IsPurposeReservedNamespace reports whether a Run Namespace is owned by a
// trusted purpose-specific flow rather than by a Stage Agent binding.
func IsPurposeReservedNamespace(namespace string) bool {
	switch namespace {
	case "inputs", "outputs", "skills", FindingProposalNamespace, RunSystemNamespace:
		return true
	default:
		return false
	}
}

func IsRunSystemNamespace(namespace string) bool {
	return namespace == RunSystemNamespace
}

// IsReservedMemoryBinding reports whether a RunScope binding belongs only to
// MemoryTools. Purpose-reserved Namespaces retain their independent policy.
func IsReservedMemoryBinding(namespace, name string) bool {
	if IsPurposeReservedNamespace(namespace) {
		return false
	}
	return strings.HasPrefix(name, MemoryArtifactPrefix)
}
