package memory

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedArtifactHistoryMemoryCodes maps each shared history outcome of
// api/testdata/v1alpha1/artifact-history-cases.json to its model-facing
// Memory code, as the Runtime adapter does (MEM-R72): history pressure never
// reports a full Namespace.
func TestSharedArtifactHistoryMemoryCodes(t *testing.T) {
	var cases struct {
		Errors []struct {
			Code       string `json:"code"`
			Retryable  bool   `json:"retryable"`
			MemoryCode string `json:"memoryCode"`
		} `json:"errors"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "artifact-history-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	causes := map[string]error{
		"artifact_history_protected":           &artifacts.HistoryProtectedError{},
		"artifact_history_maintenance_pending": &artifacts.HistoryMaintenanceError{Maintained: true},
		"artifact_limit_exceeded":              fmt.Errorf("write artifact: %w", artifacts.ErrRunRevisionLimit),
	}
	for _, test := range cases.Errors {
		cause, ok := causes[test.Code]
		if !ok {
			t.Fatalf("no store error answers %s", test.Code)
		}
		mapped := mapStoreError(cause)
		var bounded *ToolError
		if !errors.As(mapped, &bounded) || bounded.Code != test.MemoryCode {
			t.Errorf("%s maps to %v, want %s", test.Code, mapped, test.MemoryCode)
		}
	}
}
