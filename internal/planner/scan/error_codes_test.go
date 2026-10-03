package scan

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/planner"
)

type workerErrorCodeCases struct {
	Outcomes map[string]string `json:"outcomes"`
	Codes    []struct {
		ErrorCode string          `json:"errorCode"`
		ExitCode  json.RawMessage `json:"exitCode"`
		Outcome   string          `json:"outcome"`
	} `json:"codes"`
}

func readWorkerErrorCodeCases(t *testing.T) workerErrorCodeCases {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "api", "scan", "v1", "testdata", "worker-error-codes.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases workerErrorCodeCases
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	return cases
}

func TestObservationStatusClassifiesSharedWorkerErrorCodes(t *testing.T) {
	t.Parallel()
	cases := readWorkerErrorCodeCases(t)
	if len(cases.Outcomes) != len(outcomeCodes) {
		t.Fatalf("shared outcomes = %v, want %v", cases.Outcomes, outcomeCodes)
	}
	for outcome, code := range cases.Outcomes {
		if outcomeCodes[outcome] != code {
			t.Fatalf("outcome %s code = %q, want shared %q", outcome, outcomeCodes[outcome], code)
		}
	}
	seen := map[string]bool{}
	for _, entry := range cases.Codes {
		seen[entry.ErrorCode] = true
		errorCode, _ := json.Marshal(entry.ErrorCode)
		observation := map[string]json.RawMessage{
			"status":              json.RawMessage(`"failed"`),
			"exitCode":            entry.ExitCode,
			"errorCode":           errorCode,
			"stdoutTruncated":     json.RawMessage(`false`),
			"stderrTruncated":     json.RawMessage(`false`),
			"outputLimitExceeded": json.RawMessage(`false`),
		}
		status, code := observationStatus(observation)
		if status != entry.Outcome || code != cases.Outcomes[entry.Outcome] {
			t.Errorf("%s = (%s, %s), want (%s, %s)", entry.ErrorCode, status, code, entry.Outcome, cases.Outcomes[entry.Outcome])
		}
	}
	for code := range workerErrorOutcomes {
		if !seen[code] {
			t.Errorf("Go classifies %s, which the shared table does not list", code)
		}
	}
}

func TestObservationStatusRejectsUnknownWorkerErrorCode(t *testing.T) {
	t.Parallel()
	status, code := observationStatus(map[string]json.RawMessage{
		"status":    json.RawMessage(`"failed"`),
		"exitCode":  json.RawMessage(`1`),
		"errorCode": json.RawMessage(`"scanner_exploded"`),
	})
	if status != planner.ScanJobIncomplete || code != "scan_report_invalid" {
		t.Fatalf("unknown errorCode = (%s, %s)", status, code)
	}
}
