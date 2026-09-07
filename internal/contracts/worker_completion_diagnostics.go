package contracts

import (
	"bytes"
	"encoding/json"
)

// WorkerCompletionDiagnostics describes the latest invocation's local work.
// Published is a technical fact, never accepted Audit evidence/coverage.
type WorkerCompletionDiagnostics struct {
	Kind          string `json:"kind"`
	Phase         string `json:"phase"`
	AcceptedCount int    `json:"acceptedCount"`
	TotalCount    int    `json:"totalCount"`
	ReminderCount int    `json:"reminderCount"`
	FailureCode   string `json:"failureCode,omitempty"`
}

func knownCompletionPhase(phase string) bool {
	switch phase {
	case "collecting", "sealed", "publishing", "published", "failed":
		return true
	}
	return false
}

func (d WorkerCompletionDiagnostics) Validate() error {
	if d.Kind != AuditCheckResultsV1 || !knownCompletionPhase(d.Phase) ||
		d.TotalCount < 1 || d.TotalCount > 64 || d.AcceptedCount < 0 || d.AcceptedCount > d.TotalCount ||
		d.ReminderCount < 0 || d.ReminderCount > 2 {
		return invalidf("invalid Worker completion diagnostics")
	}
	if (d.Phase == "sealed" || d.Phase == "publishing" || d.Phase == "published") && d.AcceptedCount != d.TotalCount {
		return invalidf("inconsistent Worker completion counts")
	}
	if (d.Phase == "failed") != (d.FailureCode != "") || d.FailureCode != "" && !validWorkerFailureCode(d.FailureCode) {
		return invalidf("inconsistent Worker completion failure")
	}
	return nil
}

// Unknown optional kinds/phases are discarded instead of being interpreted as
// successful completion. Known data is decoded strictly and remains bounded.
func decodeWorkerCompletionDiagnostics(data []byte) (*WorkerCompletionDiagnostics, error) {
	if len(data) == 0 || bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return nil, nil
	}
	if len(data) > 4096 || rejectDuplicateJSONKeys(data) != nil {
		return nil, invalidf("invalid Worker completion diagnostics JSON")
	}
	var header struct {
		Kind  string `json:"kind"`
		Phase string `json:"phase"`
	}
	if err := json.Unmarshal(data, &header); err != nil {
		return nil, err
	}
	if header.Kind != "" && header.Phase != "" && (header.Kind != AuditCheckResultsV1 || !knownCompletionPhase(header.Phase)) {
		return nil, nil
	}
	var value WorkerCompletionDiagnostics
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, err
	}
	for _, key := range []string{"kind", "phase", "acceptedCount", "totalCount", "reminderCount"} {
		if raw, ok := fields[key]; !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return nil, invalidf("missing Worker completion field")
		}
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	if raw, ok := fields["failureCode"]; ok && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) && value.FailureCode == "" {
		return nil, invalidf("empty Worker completion failure code")
	}
	if err := value.Validate(); err != nil {
		return nil, err
	}
	return &value, nil
}

func (r *ExecutionReport) UnmarshalJSON(data []byte) error {
	type plain ExecutionReport
	var value plain
	wire := struct {
		*plain
		Completion json.RawMessage `json:"completion"`
	}{plain: &value}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return err
	}
	if err := decoder.Decode(&wire); err != nil {
		return err
	}
	completion, err := decodeWorkerCompletionDiagnostics(wire.Completion)
	if err != nil {
		return err
	}
	*r = ExecutionReport(value)
	r.Completion = completion
	return nil
}
