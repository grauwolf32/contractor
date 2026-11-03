package config

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedAuditCommonProgramsLoadRealCatalog(t *testing.T) {
	var fixtures struct {
		Cases []struct {
			Name     string `json:"name"`
			Document string `json:"document"`
			Valid    bool   `json:"valid"`
		} `json:"cases"`
		LoaderCases []struct {
			Name     string `json:"name"`
			Document string `json:"document"`
			Valid    bool   `json:"valid"`
		} `json:"loader_cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-common", "programs.json"), &fixtures); err != nil {
		t.Fatal(err)
	}
	snapshot := mustLoad(t, copyCoreFixture(t, "audit-common-catalog"))
	fixtures.Cases = append(fixtures.Cases, fixtures.LoaderCases...)
	for _, c := range fixtures.Cases {
		t.Run(c.Name, func(t *testing.T) {
			err := ValidateAuditProgram(snapshot, []byte(c.Document))
			if (err == nil) != c.Valid {
				t.Fatalf("valid=%v error=%v", c.Valid, err)
			}
			if err != nil {
				classified, ok := apierror.Of(err)
				if !ok || classified.ErrorKind() != apierror.AuditDocumentInvalid {
					t.Fatalf("expected audit_document_invalid, got %v", err)
				}
			}
		})
	}
}
