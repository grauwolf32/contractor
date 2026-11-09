// Package policy defines shared purpose-specific Artifact binding rules.
package policy

import (
	"slices"
	"strings"
)

const (
	// ProjectSitemapNamespace is writable only by the authorized sitemap reducer.
	ProjectSitemapNamespace = "sitemap"
	MemoryArtifactPrefix    = "memory."
	// HTTPBodyArtifactPrefix names the HTTP response bodies the Runtime HTTP
	// toolset writes (ART-R24).
	HTTPBodyArtifactPrefix             = "http.body."
	AuditManagedProjectNamespacePrefix = "audit-"
	AuditStandardCatalogNamespace      = "audit-standards"
	HTTPExchangeNamespace              = "http-exchanges"
	FindingProposalNamespace           = "finding-proposals"
	FindingMaterialsNamespace          = "finding-materials"
	FindingHistoryNamespace            = "finding-history"
	HistoricalAuditsNamespace          = "historical-audits"
	HistoricalProjectNamespacePrefix   = "historical-"
	// AuditContextNamespace holds frozen, authorized Run-local Audit context.
	// Models may read its exact refs; only the trusted materializer writes it.
	AuditContextNamespace = "audit-context"
)

func IsAuditManagedProjectNamespace(namespace string) bool {
	return strings.HasPrefix(namespace, AuditManagedProjectNamespacePrefix)
}

// IsPurposeReservedProjectNamespace reports Project namespaces whose trusted
// domain publisher owns writes. Generic reads and exact input forks stay ordinary.
func IsPurposeReservedProjectNamespace(namespace string) bool {
	return IsAuditManagedProjectNamespace(namespace) || namespace == ProjectSitemapNamespace ||
		namespace == FindingMaterialsNamespace || namespace == FindingHistoryNamespace || namespace == HistoricalAuditsNamespace
}

// IsAuditStandardCatalogNamespace reports the one UserScope Namespace reserved
// for curated Audit standard packages (ART-R25). It is an exact name, not a
// prefix: audit-standards-notes is an ordinary user Namespace.
func IsAuditStandardCatalogNamespace(namespace string) bool {
	return namespace == AuditStandardCatalogNamespace
}

// PurposeReservedNamespaces returns the Run Namespaces owned by a trusted
// purpose-specific flow rather than by a Stage Agent binding (ART-R114). The
// Runtime's contracts.artifacts.PURPOSE_RESERVED_NAMESPACES is the same list;
// api/testdata/v1alpha1/memory-reservation-cases.json pins both.
func PurposeReservedNamespaces() []string {
	return []string{"inputs", "outputs", "skills", FindingProposalNamespace, AuditContextNamespace, HTTPExchangeNamespace}
}

// IsPurposeReservedNamespace reports whether a Run Namespace is one of
// PurposeReservedNamespaces.
func IsPurposeReservedNamespace(namespace string) bool {
	return slices.Contains(PurposeReservedNamespaces(), namespace)
}

func IsReservedMemoryBinding(namespace, name string) bool {
	if IsPurposeReservedNamespace(namespace) {
		return false
	}
	return strings.HasPrefix(name, MemoryArtifactPrefix)
}

// Model visibility (ART-R19, ART-R24, ART-R109): the generic
// model-visible Artifact tools (run-artifacts@1) never disclose these
// bindings. The private paged listing excludes them in storage before its
// limit applies. The Runtime applies the same rule in artifact_visibility.py;
// both sides are checked against api/testdata/v1alpha1/artifact-list-cases.json.

// ModelHiddenNamespaces returns the Run Namespaces hidden from model tools
// entirely.
func ModelHiddenNamespaces() []string {
	return []string{"skills"}
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
