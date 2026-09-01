// Package artifactpolicy defines shared purpose-specific Artifact binding rules.
package artifactpolicy

import "strings"

const MemoryArtifactPrefix = "memory."

// IsReservedMemoryBinding reports whether a RunScope binding belongs only to
// MemoryTools. Purpose-reserved Namespaces retain their independent policy.
func IsReservedMemoryBinding(namespace, name string) bool {
	switch namespace {
	case "inputs", "outputs", "skills":
		return false
	default:
		return strings.HasPrefix(name, MemoryArtifactPrefix)
	}
}
