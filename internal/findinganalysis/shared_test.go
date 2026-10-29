package findinganalysis

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedFindingAnalysisCases(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name     string
			Valid    bool
			Document string
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "finding-analysis-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("empty fixture")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			if _, err := decode([]byte(tc.Document)); (err == nil) != tc.Valid {
				t.Fatalf("valid=%v: %v", tc.Valid, err)
			}
		})
	}
}
