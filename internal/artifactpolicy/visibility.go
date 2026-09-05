// Package artifactpolicy defines shared purpose-specific Artifact binding rules.
package artifactpolicy

import "strings"

const (
	MemoryArtifactPrefix               = "memory."
	AuditManagedProjectNamespacePrefix = "audit-"
)

func IsAuditManagedProjectNamespace(namespace string) bool {
	return strings.HasPrefix(namespace, AuditManagedProjectNamespacePrefix)
}

// IsPurposeReservedNamespace reports whether a Run Namespace is owned by a
// trusted purpose-specific flow rather than by a Stage Agent binding.
func IsPurposeReservedNamespace(namespace string) bool {
	switch namespace {
	case "inputs", "outputs", "skills":
		return true
	default:
		return false
	}
}

// IsReservedMemoryBinding reports whether a RunScope binding belongs only to
// MemoryTools. Purpose-reserved Namespaces retain their independent policy.
func IsReservedMemoryBinding(namespace, name string) bool {
	if IsPurposeReservedNamespace(namespace) {
		return false
	}
	return strings.HasPrefix(name, MemoryArtifactPrefix)
}
