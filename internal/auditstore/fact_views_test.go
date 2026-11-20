package auditstore

import (
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"testing"
)

func TestSharedFactPresentationAndNeedsReview(t *testing.T) {
	var fixture struct {
		Views []struct {
			Name               string
			State              *FindingState
			AnalystVerdict     *AnalystVerdict
			AnalystSeverity    *FindingSeverity
			SeveritySuggestion string
			Severity           string
			NeedsReview        bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "finding-fact-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, c := range fixture.Views {
		t.Run(c.Name, func(t *testing.T) {
			row := ProjectFindingSummary{State: c.State, AnalystVerdict: c.AnalystVerdict, AnalystSeverity: c.AnalystSeverity, SeveritySuggestion: c.SeveritySuggestion}
			if string(row.PresentationSeverity()) != c.Severity || row.NeedsReview() != c.NeedsReview || row.Matches(nil, nil, true, nil) != c.NeedsReview {
				t.Fatalf("view: %+v", row)
			}
		})
	}
}
