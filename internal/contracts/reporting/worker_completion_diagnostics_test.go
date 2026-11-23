package reporting

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedWorkerCompletionDiagnostics gives every case of the shared fixture
// the same verdict as the Python models and the JSON Schema: a valid object
// is kept exactly, and any invalid one, including an unknown kind or phase
// (AWC-R115), rejects the whole report or live Worker State.
func TestSharedWorkerCompletionDiagnostics(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "audit-completion", "diagnostics.json")
	var cases []struct {
		Name                string
		Value               json.RawMessage
		Valid               bool
		SemanticOnly        bool
		JSONWhitespaceBytes int
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, item := range cases {
		seen[item.Name] = true
		t.Run(item.Name, func(t *testing.T) {
			item.Value = json.RawMessage(strings.Replace(string(item.Value), "{", "{"+strings.Repeat(" ", item.JSONWhitespaceBytes), 1))
			liveRaw := `{"counters":{},"toolCalls":[],"errors":[],"finalOutcome":null,"truncated":false,"completion":` + string(item.Value) + `}`
			var live WorkerAllocationState
			liveErr := contracts.DecodePrivate([]byte(liveRaw), &live)
			if liveErr == nil {
				liveErr = live.validate()
			}
			if (liveErr == nil) != item.Valid {
				t.Fatalf("live valid=%v want=%v: %v", liveErr == nil, item.Valid, liveErr)
			}
			raw := `{"reportId":"report-1","complete":true,"metrics":{"tools":{}},"toolCalls":[],"errors":[],"truncated":false,"completion":` + string(item.Value) + `}`
			var report ExecutionReport
			err := contracts.DecodePrivate([]byte(raw), &report)
			if err == nil {
				err = report.Validate()
			}
			if (err == nil) != item.Valid {
				t.Fatalf("valid=%v want=%v: %v", err == nil, item.Valid, err)
			}
			if strings.HasPrefix(item.Name, "unknown-") {
				for _, problem := range []error{err, liveErr} {
					var typed *contracts.PrivateProtocolError
					if !errors.As(problem, &typed) || typed.Class != contracts.PrivateProtocolErrorSchema {
						t.Fatalf("unknown kind or phase is not a private schema error: %v", problem)
					}
				}
			}
			if !item.Valid {
				return
			}
			if report.Completion == nil || live.Completion == nil {
				t.Fatal("valid diagnostics were dropped")
			}
			for _, value := range []*WorkerCompletionDiagnostics{report.Completion, live.Completion} {
				encoded, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				contractstest.AssertSemanticJSONEqual(t, encoded, item.Value)
			}
		})
	}
	for _, name := range []string{"unknown-kind", "unknown-phase"} {
		if !seen[name] {
			t.Fatalf("shared fixture lost its %s case", name)
		}
	}
}

func TestCompletionOmissionAndStrictDuplicateFields(t *testing.T) {
	var report ExecutionReport
	if err := json.Unmarshal([]byte(`{"reportId":"old","complete":true,"metrics":{"tools":{}},"toolCalls":[],"errors":[],"truncated":false}`), &report); err != nil {
		t.Fatal(err)
	}
	if report.Completion != nil {
		t.Fatal("legacy report acquired completion")
	}
	encoded, _ := json.Marshal(report)
	if strings.Contains(string(encoded), "completion") {
		t.Fatal("legacy omission changed")
	}
	if _, err := decodeWorkerCompletionDiagnostics([]byte(`{"kind":"audit-check-results@1","phase":"published","phase":"failed"}`)); err == nil {
		t.Fatal("accepted duplicate phase")
	}
}
