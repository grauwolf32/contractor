package config

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
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

func TestSharedAuditCommonLimitsRetainPublishedCeilings(t *testing.T) {
	var fixture struct {
		Limits map[string]int `json:"limits"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-common", "boundaries.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	published := map[string]int{
		"max_tasks":               auditprofile.MaxAuditItemsTotal,
		"max_iterations":          auditprofile.MaxAuditRounds,
		"max_tasks_per_iteration": auditprofile.MaxAuditItemsPerRound,
		"batch_size":              auditprofile.MaxAuditBatchSize,
		"max_submitted_runs":      auditprofile.MaxAuditSubmittedRuns,
		"max_attempts":            auditprofile.MaxAuditRunAttempts,
		"deadline_seconds":        auditprofile.MaxAuditDeadlineSeconds,
		"max_evidence_bytes":      int(auditprofile.MaxAuditEvidenceBytes),
	}
	for key, want := range published {
		if fixture.Limits[key] != want {
			t.Fatalf("%s=%d want existing ceiling %d", key, fixture.Limits[key], want)
		}
	}
}
