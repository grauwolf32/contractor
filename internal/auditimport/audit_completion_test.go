package auditimport

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/config"
)

// The Python collector consumes this same data through its production tool and
// canonical encoder. These are task-local acceptance checks, not DB integration.
func TestAuditCompletionSharedTaskLocalValidation(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "api", "testdata", "audit-completion", "task-local-validation.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct {
		Tasks map[string]auditdomain.ItemTask `json:"tasks"`
		Cases []struct {
			Name   string `json:"name"`
			Task   string `json:"task"`
			Valid  bool   `json:"valid"`
			Result struct {
				Assessment string                 `json:"assessment"`
				Summary    string                 `json:"summary"`
				Completed  []string               `json:"completed"`
				Gaps       []string               `json:"gaps"`
				Evidence   []auditdomain.Evidence `json:"evidence"`
			} `json:"result"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures.Cases) == 0 {
		t.Fatal("empty shared acceptance fixture")
	}
	for _, fixture := range fixtures.Cases {
		t.Run(fixture.Name, func(t *testing.T) {
			task := fixtures.Tasks[fixture.Task]
			if _, err := auditdomain.EncodeItemTask(task); err != nil {
				t.Fatalf("invalid trusted task fixture: %v", err)
			}
			evidence := map[string]validatedEvidence{}
			ids := []string{}
			for index, value := range fixture.Result.Evidence {
				id := fmt.Sprintf("ev-%03d", index)
				value.ID = id
				evidence[id] = validatedEvidence{value: value}
				ids = append(ids, id)
			}
			result := auditdomain.CheckResult{
				ItemKey: task.ItemKey, SubjectKey: task.SubjectKey,
				Assessment: fixture.Result.Assessment, Summary: fixture.Result.Summary,
				EvidenceIDs: ids, Proposals: []auditdomain.ProposalSelection{},
				Coverage: auditdomain.ResultCoverage{
					Requested: expectedCoverage(task), Completed: fixture.Result.Completed,
					Gaps: fixture.Result.Gaps,
				},
			}
			_, resultErr := auditdomain.EncodeCheckResultSet(auditdomain.CheckResultSet{
				Schema:                  auditdomain.CheckResultsSchema,
				ExecutionManifestDigest: "sha256:" + strings.Repeat("a", 64),
				Results:                 []auditdomain.CheckResult{result},
			})
			if resultErr == nil {
				_, resultErr = semanticCoverage(config.AuditModeCustomChecklist, task, result, evidence)
			}
			if (resultErr == nil) != fixture.Valid {
				t.Fatalf("accepted=%v, want=%v (error=%v)", resultErr == nil, fixture.Valid, resultErr)
			}
		})
	}
}
