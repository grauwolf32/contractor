package reporting

import (
	"bytes"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
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
	if d.Kind != contracts.AuditCheckResultsV1 || !knownCompletionPhase(d.Phase) ||
		d.TotalCount < 1 || d.TotalCount > 64 || d.AcceptedCount < 0 || d.AcceptedCount > d.TotalCount ||
		d.ReminderCount < 0 || d.ReminderCount > 2 {
		return contracts.Invalidf("invalid Worker completion diagnostics")
	}
	if (d.Phase == "sealed" || d.Phase == "publishing" || d.Phase == "published") && d.AcceptedCount != d.TotalCount {
		return contracts.Invalidf("inconsistent Worker completion counts")
	}
	if (d.Phase == "failed") != (d.FailureCode != "") || d.FailureCode != "" && !contracts.ValidWorkerFailureCode(d.FailureCode) {
		return contracts.Invalidf("inconsistent Worker completion failure")
	}
	return nil
}

// decodeWorkerCompletionDiagnostics strictly decodes the optional completion
// object of a report or live Worker State. An unknown kind or phase is invalid
// like any other malformed value (AWC-R115): it fails the containing message
// with contracts.ErrValidation instead of being dropped.
func decodeWorkerCompletionDiagnostics(data []byte) (*WorkerCompletionDiagnostics, error) {
	if len(data) == 0 || bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return nil, nil
	}
	var fields map[string]json.RawMessage
	if len(data) > 4096 || strictjson.Decode(data, &fields) != nil {
		return nil, contracts.Invalidf("invalid Worker completion diagnostics JSON")
	}
	for _, key := range []string{"kind", "phase", "acceptedCount", "totalCount", "reminderCount"} {
		if raw, ok := fields[key]; !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return nil, contracts.Invalidf("missing Worker completion field")
		}
	}
	var value WorkerCompletionDiagnostics
	if err := strictjson.Decode(data, &value); err != nil {
		return nil, err
	}
	if raw, ok := fields["failureCode"]; ok && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) && value.FailureCode == "" {
		return nil, contracts.Invalidf("empty Worker completion failure code")
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
	if err := strictjson.Decode(data, &wire); err != nil {
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
