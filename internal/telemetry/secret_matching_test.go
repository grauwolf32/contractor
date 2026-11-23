package telemetry

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
)

// A short credential is redacted only as a complete string, long ones
// anywhere and longest first, and error codes are never partially redacted
// (OBS-R31, OBS-R32, LIM-OBS-3).
func TestNormalizationMatchesSecretsByTheSharedRule(t *testing.T) {
	t.Parallel()
	const (
		username = "admin"
		password = "admin-0123456789-password-xyz" // starts with the username
		token    = "tok-0123456789abc"
	)
	policy := NewPolicy(username, password, token)
	stop := "administrator stopped after " + token
	finished := time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC)
	report := reporting.AllocationFinalReport{
		ReportID: "allocation-final-1", AllocationID: "allocation-1",
		StartedAt: finished.Add(-time.Minute), FinishedAt: finished,
		Worker: reporting.ExecutionReport{
			ReportID: "worker-1", Complete: true,
			ToolCalls: []reporting.ToolCallRecord{{
				CallID: "tool-1", Tool: "read_source_file", Outcome: reporting.ToolCallFailed,
				Error:     &reporting.ExecutionError{Code: "read_failed", Message: "cannot read main.go"},
				Arguments: map[string]any{"path": "data/main.go", "user": username, "note": "login " + password},
			}},
			Errors: []reporting.ExecutionError{
				{Code: "tool_call_failed", Message: "cannot read main.go"},
				{Code: "admin_rejected", Message: "administrator login failed"},
				{Code: "bad_" + token, Message: username},
			},
		},
		Runtime: reporting.RuntimeReport{Complete: true, StopReason: &stop},
	}
	normalized, err := policy.NormalizeAllocationReport(report)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(normalized)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "0123456789") {
		t.Fatal("normalized report keeps part of a long secret")
	}
	arguments := normalized.Worker.ToolCalls[0].Arguments
	if arguments["path"] != "data/main.go" || arguments["user"] != redactedValue ||
		arguments["note"] != "login "+redactedValue {
		t.Fatalf("arguments were not matched by the shared rule")
	}
	if failure := normalized.Worker.ToolCalls[0].Error; failure.Code != "read_failed" ||
		failure.Message != "cannot read main.go" {
		t.Fatal("tool call error was not matched by the shared rule")
	}
	errors := normalized.Worker.Errors
	if errors[0].Code != "tool_call_failed" || errors[0].Message != "cannot read main.go" ||
		errors[1].Code != "admin_rejected" || errors[1].Message != "administrator login failed" ||
		errors[2].Code != "tool_call_failed" || errors[2].Message != redactedValue {
		t.Fatal("error codes or messages were not matched by the shared rule")
	}
	if *normalized.Runtime.StopReason != "administrator stopped after "+redactedValue {
		t.Fatal("stop reason was not matched by the shared rule")
	}
}

// The Planner span filters drop an attribute under the same rule as the
// Runtime exporter's contains_private_value: equal to a secret, or holding a
// long one (OBS-R31).
func TestPlannerSpanFiltersUseTheSharedSecretRule(t *testing.T) {
	t.Parallel()
	secrets := []string{"admin", "tok-0123456789abc"}
	if safePlannerString("administrator", secrets) != "administrator" ||
		safePlannerString("admin", secrets) != "" ||
		safePlannerString("x-tok-0123456789abc", secrets) != "" {
		t.Fatal("Planner span attribute filter does not follow the shared rule")
	}
	labels := plannerRunMetadataLabelAttributes(map[string]string{
		"team": "administrators", "owner": "admin", "ticket": "tok-0123456789abc-1",
	}, secrets)
	if len(labels) != 1 || labels["contractor.run.label.team"] != "administrators" {
		t.Fatal("Planner run label filter does not follow the shared rule")
	}
}
