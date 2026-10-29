package findingexport

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func TestSharedFindingExportRequestCases(t *testing.T) {
	var fixture struct {
		Limits struct{ MaxFindings, MaxContentBytes, MaxDiagnosticsBytes int }
		Cases  []struct {
			Name                     string
			Request                  json.RawMessage
			Valid, IncludeDuplicates bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "public", "finding-export-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Limits.MaxFindings != auditdomain.MaximumCollectionEntries || fixture.Limits.MaxContentBytes != MaxBytes || fixture.Limits.MaxDiagnosticsBytes != MaxDiagnosticsBytes {
		t.Fatal("export limits drifted from shared fixture")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			var request Request
			err := strictjson.DecodeNonNull(tc.Request, &request)
			if err == nil {
				_, _, err = request.Normalize("audit-one")
			}
			if (err == nil) != tc.Valid {
				t.Fatalf("valid=%t: %v", tc.Valid, err)
			}
			if tc.Valid && request.IncludeDuplicates != tc.IncludeDuplicates {
				t.Fatal("duplicate default changed")
			}
		})
	}
}
