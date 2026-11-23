package runstore

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestScanRunEventExposesOnlyClosedActionCode(t *testing.T) {
	for _, code := range []string{"scan_owner_acquired", "scan_plan_initialized", "scan_job_started", "scan_job_finished"} {
		data := map[string]any{
			"stageExecutionId": "stage-1", "sessionId": "session-1", "invocationId": "invocation-1",
		}
		activity := map[string]any{"kind": code, "author": "scan_planner", "functionCalls": []string{}, "functionResults": []string{}}
		data["activity"] = activity
		encode := func() RunEventAppend {
			encoded, err := json.Marshal(data)
			if err != nil {
				t.Fatal(err)
			}
			return RunEventAppend{EventID: "event-1", EventSchemaVersion: contracts.APIVersion, Kind: RunEventPlannerActivity, Data: encoded}
		}
		if err := validateRunEventAppend(encode()); err != nil {
			t.Fatal(err)
		}
		data["workerName"] = "private-worker-value"
		if err := validateRunEventAppend(encode()); err == nil {
			t.Fatal("scan event accepted extra semantic data")
		}
		delete(data, "workerName")
		activity["functionCalls"] = []string{"private_function"}
		if err := validateRunEventAppend(encode()); err == nil {
			t.Fatal("scan activity accepted function payload")
		}
		activity["functionCalls"] = []string{}
		activity["kind"] = "arbitrary_private_value"
		if err := validateRunEventAppend(encode()); err == nil {
			t.Fatal("scan event accepted arbitrary code")
		}
	}
}
