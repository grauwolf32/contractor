package privateartifacts

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// historyErrorCase is one error of
// api/testdata/v1alpha1/artifact-history-cases.json.
type historyErrorCase struct {
	Code       string `json:"code"`
	Status     int    `json:"status"`
	Retryable  bool   `json:"retryable"`
	RetryAfter bool   `json:"retryAfter"`
	Resend     bool   `json:"resend"`
	MemoryCode string `json:"memoryCode"`
}

// TestSharedArtifactHistoryPrivateErrors answers each shared history outcome
// through the private Artifact API, which the Runtime decodes with the same
// fixture (runtime/tests/test_shared_artifact_history.py): protected history
// and the Run revision limit are permanent 409s, and pending maintenance is a
// retryable 503 with the policy's Retry-After (ART-R126, ART-R127, ART-R118).
func TestSharedArtifactHistoryPrivateErrors(t *testing.T) {
	var cases struct {
		Errors []historyErrorCase `json:"errors"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "artifact-history-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	causes := map[string]error{
		"artifact_history_protected":           &artifacts.HistoryProtectedError{},
		"artifact_history_maintenance_pending": &artifacts.HistoryMaintenanceError{Maintained: true, Removed: 64},
		"artifact_limit_exceeded":              fmt.Errorf("write artifact: %w", artifacts.ErrRunRevisionLimit),
	}
	if len(cases.Errors) != len(causes) {
		t.Fatalf("shared history errors = %d, want %d", len(cases.Errors), len(causes))
	}
	for _, test := range cases.Errors {
		t.Run(test.Code, func(t *testing.T) {
			cause, ok := causes[test.Code]
			if !ok {
				t.Fatalf("no Server error answers %s", test.Code)
			}
			repository := &rejectingWriteRepository{memoryRepository: newMemoryRepository(), err: cause}
			handler, err := NewHandler(Dependencies{Transfers: artifactpgtest.NewStorage(),
				Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(repository),
				NewRequestID: func() (string, error) { return "artifact-history", nil },
			})
			if err != nil {
				t.Fatal(err)
			}
			response := putArtifact(t, handler, "builder", "report", "*", []byte("report"))
			assertPrivateErrorEnvelope(t, response, test.Status, test.Code, test.Retryable)
			if retryAfter := response.Header().Get("Retry-After"); (retryAfter == "1") != test.RetryAfter || (retryAfter != "" && !test.RetryAfter) {
				t.Fatalf("Retry-After = %q, want present %t", retryAfter, test.RetryAfter)
			}
		})
	}
}
