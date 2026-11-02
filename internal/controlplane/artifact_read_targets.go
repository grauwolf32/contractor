package controlplane

import (
	"slices"

	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// projectArtifactReadTargets composes concrete read authority from the pinned
// configuration. Today there is at most one target with two operations; the
// generic Artifact handler does not interpret toolset identities or policies.
func projectArtifactReadTargets(grant AllocationGrant) []ProjectArtifactReadTarget {
	if grant.SitemapPolicy == nil || grant.SitemapPolicy.Validate() != nil {
		return nil
	}
	operations := []string{}
	for _, operation := range []string{contracts.SitemapListTool, contracts.SitemapReadTool} {
		if slices.Contains(grant.SitemapTools, operation) {
			operations = append(operations, operation)
		}
	}
	if len(operations) == 0 {
		return nil
	}
	return []ProjectArtifactReadTarget{{Namespace: artifactpolicy.ProjectSitemapNamespace, Name: "map", Operations: operations}}
}
