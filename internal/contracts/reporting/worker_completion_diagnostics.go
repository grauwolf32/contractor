package reporting

import (
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

// WorkerCompletionDiagnostics describes the latest invocation's local work.
// Published is a technical fact, never accepted Audit evidence/coverage.
type WorkerCompletionDiagnostics struct {
	Kind          string `json:"kind" strictjson:"required"`
	Phase         string `json:"phase" strictjson:"required"`
	AcceptedCount int    `json:"acceptedCount" strictjson:"required"`
	TotalCount    int    `json:"totalCount" strictjson:"required"`
	ReminderCount int    `json:"reminderCount" strictjson:"required"`
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

// UnmarshalJSON strictly decodes the optional completion object of a report or
// live Worker State, so every decoder of either applies the same checks.
func (d *WorkerCompletionDiagnostics) UnmarshalJSON(data []byte) error {
	value, err := decodeWorkerCompletionDiagnostics(data)
	if err != nil {
		return err
	}
	*d = value
	return nil
}

// decodeWorkerCompletionDiagnostics decodes one bounded completion object. An
// unknown kind or phase is invalid like any other malformed value (AWC-R115):
// it fails the containing message with contracts.ErrValidation instead of
// being dropped.
func decodeWorkerCompletionDiagnostics(data []byte) (WorkerCompletionDiagnostics, error) {
	type plain WorkerCompletionDiagnostics
	var wire struct {
		plain
		// FailureCode tells an explicit empty code from an omitted one.
		FailureCode *string `json:"failureCode"`
	}
	if err := strictjson.DecodeNonNull(data, &wire); err != nil {
		return WorkerCompletionDiagnostics{}, err
	}
	value := WorkerCompletionDiagnostics(wire.plain)
	if wire.FailureCode != nil {
		if *wire.FailureCode == "" {
			return WorkerCompletionDiagnostics{}, contracts.Invalidf("empty Worker completion failure code")
		}
		value.FailureCode = *wire.FailureCode
	}
	size, err := contracts.ResultJSONSize(value)
	if err != nil || size > 4096 {
		return WorkerCompletionDiagnostics{}, contracts.Invalidf("Worker completion diagnostics exceed 4096 bytes")
	}
	if err := value.Validate(); err != nil {
		return WorkerCompletionDiagnostics{}, err
	}
	return value, nil
}
