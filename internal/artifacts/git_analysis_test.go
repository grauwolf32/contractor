package artifacts_test

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedGitAnalysisSummaries(t *testing.T) {
	var fixture struct {
		AnalysisSummaries []struct {
			Name  string
			Value json.RawMessage
			Valid bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "git-import-modes.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, test := range fixture.AnalysisSummaries {
		t.Run(test.Name, func(t *testing.T) {
			var value artifacts.GitAnalysis
			err := json.Unmarshal(test.Value, &value)
			if (err == nil) != test.Valid {
				t.Fatalf("valid=%v want %v: %v", err == nil, test.Valid, err)
			}
		})
	}
	for _, raw := range []string{`null`, `{}`, `{"policy":"source-analysis-v1","policy":"source-analysis-v1","reportPath":".contractor-import-report.json","includedFiles":1,"excludedEntries":0}`, `{"Policy":"source-analysis-v1","reportPath":".contractor-import-report.json","includedFiles":1,"excludedEntries":0}`} {
		var value artifacts.GitAnalysis
		if json.Unmarshal([]byte(raw), &value) == nil {
			t.Fatalf("invalid read accepted: %s", raw)
		}
	}
}
