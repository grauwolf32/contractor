package evalservice

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestAuditOutputEligibilityMatchesCollectedArtifacts(t *testing.T) {
	for _, test := range []struct {
		name, slot, media, state string
	}{
		{"machine-report", "report", "application/json", "eligible"},
		{"coverage", "coverage", "application/json", "eligible"},
		{"findings", "findings", "application/json", "eligible"},
		{"summary", "summary", "text/markdown", "eligible"},
		{"report-is-not-markdown", "report", "text/markdown", "unsupported"},
		{"summary-is-not-json", "summary", "application/json", "unsupported"},
		{"coverage-is-not-yaml", "coverage", "application/yaml", "unsupported"},
		{"unknown-output", "unknown", "application/json", "unsupported"},
	} {
		t.Run(test.name, func(t *testing.T) {
			c := evaldomain.Case{Outputs: map[string]evaldomain.Output{
				"evidence": {Required: true, MediaTypes: []string{test.media}},
			}}
			variant := evaldomain.Variant{Kind: "audit", OutputMapping: map[string]string{"evidence": test.slot}}
			snapshot := BindingSnapshot{Audit: &config.ResolvedAuditProfile{}}
			got := caseEligibility(c, variant, snapshot, nil)
			if got.State != test.state {
				t.Fatalf("output %s (%s): got %+v, want %s", test.slot, test.media, got, test.state)
			}
		})
	}
}
