// Package artifactpolicy defines shared purpose-specific Artifact binding rules.
package artifactpolicy

import (
	"slices"
	"strings"
)

const (
	MemoryArtifactPrefix = "memory."
	// HTTPBodyArtifactPrefix names the HTTP response bodies the Runtime HTTP
	// toolset writes (ART-R24).
	HTTPBodyArtifactPrefix             = "http.body."
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

// Model visibility (ART-R17, ART-R19, ART-R24, ART-R109): the generic
// model-visible Artifact tools (run-artifacts@1) never disclose these
// bindings. The private paged listing excludes them in storage before its
// limit applies. The Runtime applies the same rule in artifact_visibility.py;
// both sides are checked against api/testdata/v1alpha1/artifact-list-cases.json.

// ModelHiddenNamespaces returns the Run Namespaces hidden from model tools
// entirely.
func ModelHiddenNamespaces() []string {
	return []string{"skills", RunSystemNamespace}
}

// ModelHiddenNamePrefixes returns the binding-name prefixes hidden from model
// tools outside ModelNamePrefixExemptNamespaces.
func ModelHiddenNamePrefixes() []string {
	return []string{MemoryArtifactPrefix, HTTPBodyArtifactPrefix}
}

// ModelNamePrefixExemptNamespaces returns the purpose-reserved Namespaces whose
// own contract governs their names (ART-R22), so a hidden name prefix does not
// apply there.
func ModelNamePrefixExemptNamespaces() []string {
	return []string{"inputs", "outputs"}
}

// IsModelHiddenBinding reports whether generic model tools must not disclose a
// RunScope binding.
func IsModelHiddenBinding(namespace, name string) bool {
	if slices.Contains(ModelHiddenNamespaces(), namespace) {
		return true
	}
	if slices.Contains(ModelNamePrefixExemptNamespaces(), namespace) {
		return false
	}
	for _, prefix := range ModelHiddenNamePrefixes() {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}
