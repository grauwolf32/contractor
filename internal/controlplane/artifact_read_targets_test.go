package controlplane

import (
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestProjectArtifactReadsDeriveOnlyPinnedReaders(t *testing.T) {
	grant := AllocationGrant{SitemapTools: contracts.SitemapToolNames()}
	if got := projectArtifactReadTargets(grant); len(got) != 0 {
		t.Fatal("missing policy granted Project reads")
	}
	grant.SitemapPolicy = &contracts.ProjectSitemapPolicy{Version: contracts.SitemapAccessVersion, CollectObservations: true}
	for _, operation := range contracts.SitemapToolNames() {
		grant.SitemapTools = []string{operation}
		got := projectArtifactReadTargets(grant)
		reader := slices.Contains([]string{contracts.SitemapListTool, contracts.SitemapReadTool}, operation)
		if reader && (len(got) != 1 || len(got[0].Operations) != 1 || got[0].Operations[0] != operation) || !reader && len(got) != 0 {
			t.Fatalf("operation %s: %+v", operation, got)
		}
	}
}
