package runstore

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRunEventClosedSchemasAcceptEveryFixedKind(t *testing.T) {
	for kind := range validRunEventKinds {
		t.Run(string(kind), func(t *testing.T) {
			if err := validateRunEventAppend(validPlannerRunEvent(t, kind)); err != nil {
				t.Fatalf("valid %s event: %v", kind, err)
			}
		})
	}
}

func TestPlannerRunEventClosedSchemasRejectUnknownOrInconsistentData(t *testing.T) {
	tests := []struct {
		name  string
		event RunEventAppend
	}{
		{
			name: "unknown top-level field",
			event: RunEventAppend{
				EventID: "event-unknown", EventSchemaVersion: contracts.APIVersion,
				Kind: RunEventPlannerStarted,
				Data: json.RawMessage(`{
					"stageExecutionId":"stage-1","sessionId":"session-1",
					"invocationId":"invocation-1","physicalEndpoint":"https://secret.invalid"
				}`),
			},
		},
		{
			name: "unknown nested plan field",
			event: RunEventAppend{
				EventID: "event-plan", EventSchemaVersion: contracts.APIVersion,
				Kind: RunEventPlannerPlanChanged,
				Data: json.RawMessage(`{
					"stageExecutionId":"stage-1","sessionId":"session-1","invocationId":"invocation-1",
					"plan":{"revision":1,"subtasks":[{"id":"0","objective":"inspect",
					"instructions":"read","status":"pending","rawPrompt":"secret"}],"currentSubtaskId":"0"}
				}`),
			},
		},
		{
			name: "dispatch details differ from plan",
			event: func() RunEventAppend {
				value := validPlannerRunEvent(t, RunEventPlannerDispatchStarted)
				var data map[string]any
				if err := json.Unmarshal(value.Data, &data); err != nil {
					t.Fatal(err)
				}
				data["workerName"] = "wrong-worker"
				value.Data = mustRunEventJSON(t, data)
				return value
			}(),
		},
		{
			name: "unsupported schema",
			event: func() RunEventAppend {
				value := validPlannerRunEvent(t, RunEventPlannerStarted)
				value.EventSchemaVersion = "contractor/v99"
				return value
			}(),
		},
		{
			name: "lifecycle event contains unknown field",
			event: RunEventAppend{
				EventID: "event-lifecycle", EventSchemaVersion: contracts.APIVersion,
				Kind: RunEventLifecycleChanged,
				Data: json.RawMessage(`{"runId":"run-1","resource":"run","state":"running","reason":"raw"}`),
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := validateRunEventAppend(test.event); err == nil {
				t.Fatal("invalid closed-schema event was accepted")
			}
		})
	}
}

func TestPlannerRunEventIdentityMustMatchDurableSession(t *testing.T) {
	event := validPlannerRunEvent(t, RunEventPlannerStarted)
	if err := validatePlannerRunEventIdentity(
		event, "stage-1", "session-1", "invocation-1",
	); err != nil {
		t.Fatalf("matching identity: %v", err)
	}
	if err := validatePlannerRunEventIdentity(
		event, "stage-1", "session-1", "another-invocation",
	); err == nil {
		t.Fatal("mismatched Run event identity was accepted")
	}
}

func TestPublicRunEventGrammarRejectsValuesOutsideTheWebSocketSchema(t *testing.T) {
	for _, mutate := range []func(*plannerRunEventData){
		func(data *plannerRunEventData) { data.SessionID = "unsafe session" },
		func(data *plannerRunEventData) { data.SubtaskID = "00" },
		func(data *plannerRunEventData) { data.CallID = "dispatch-1" },
		func(data *plannerRunEventData) { data.WorkerName = "unsafe worker" },
	} {
		event := validPlannerRunEvent(t, RunEventPlannerDispatchSelected)
		var data plannerRunEventData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			t.Fatal(err)
		}
		mutate(&data)
		event.Data = mustRunEventJSON(t, data)
		if err := validateRunEventAppend(event); err == nil {
			t.Fatal("event outside the public WebSocket grammar was accepted")
		}
	}

	lifecycle := WorkflowRunEvent{
		RunID: "run-1", EventID: "event-lifecycle", EventSchemaVersion: contracts.APIVersion,
		Kind: RunEventLifecycleChanged,
		Data: json.RawMessage(`{"runId":"another-run","resource":"run","state":"running"}`),
	}
	if _, err := EncodePublicRunEventData(lifecycle); err == nil {
		t.Fatal("lifecycle data for another Run was accepted for public delivery")
	}
}

func TestPlannerRunEventAllowsFailedFinishBeforeFirstSubtask(t *testing.T) {
	event := validPlannerRunEvent(t, RunEventPlannerFinishRequested)
	var data plannerRunEventData
	if err := json.Unmarshal(event.Data, &data); err != nil {
		t.Fatal(err)
	}
	zero := uint64(0)
	data.PlanRevision = &zero
	data.Outcome = string(contracts.StageFailed)
	event.Data = mustRunEventJSON(t, data)
	if err := validateRunEventAppend(event); err != nil {
		t.Fatalf("failed finish before the first subtask: %v", err)
	}

	data.Outcome = string(contracts.StageSucceeded)
	event.Data = mustRunEventJSON(t, data)
	if err := validateRunEventAppend(event); err == nil {
		t.Fatal("succeeded finish before the first subtask was accepted")
	}
}

func validPlannerRunEvent(t *testing.T, kind RunEventKind) RunEventAppend {
	t.Helper()
	if kind == RunEventLifecycleChanged {
		return RunEventAppend{
			EventID: "event-lifecycle", EventSchemaVersion: contracts.APIVersion,
			Kind: kind, Data: mustRunEventJSON(t, lifecycleRunEventData{
				RunID: "run-1", Resource: "stageExecution",
				StageExecutionID: "stage-1", State: "running",
			}),
		}
	}
	data := plannerRunEventData{
		StageExecutionID: "stage-1", SessionID: "session-1", InvocationID: "invocation-1",
	}
	pending := &plannerRunPlanProjection{
		Revision: 1, CurrentSubtaskID: "0",
		Subtasks: []plannerRunSubtask{{
			ID: "0", Objective: "Inspect", Instructions: "Read the source", Status: "pending",
		}},
	}
	revision := uint64(1)
	switch kind {
	case RunEventPlannerStarted, RunEventPlannerRequestRecorded:
	case RunEventPlannerActivity:
		data.Activity = &plannerRunActivity{
			Kind: "adk_event", Author: "router_planner",
			FunctionCalls: []string{"execute_current_subtask"}, FunctionResults: []string{},
			InputTokens: 10, OutputTokens: 4,
		}
	case RunEventPlannerPlanChanged:
		data.Plan = pending
	case RunEventPlannerCurrentChanged:
		data.PlanRevision = &revision
		data.SubtaskID = "0"
	case RunEventPlannerDispatchSelected:
		data.PlanRevision = &revision
		data.SubtaskID = "0"
		data.CallID = "dispatch-0001"
		data.WorkerName = "reviewer"
	case RunEventPlannerDispatchStarted:
		data.SubtaskID = "0"
		data.CallID = "dispatch-0001"
		data.WorkerName = "reviewer"
		data.Plan = &plannerRunPlanProjection{
			Revision: 2, CurrentSubtaskID: "0",
			Subtasks: []plannerRunSubtask{{
				ID: "0", Objective: "Inspect", Instructions: "Read the source", Status: "running",
			}},
			ActiveDispatch: &plannerRunActiveDispatch{
				CallID: "dispatch-0001", SubtaskID: "0", WorkerName: "reviewer",
			},
		}
	case RunEventPlannerDispatchCompleted:
		data.SubtaskID = "0"
		data.CallID = "dispatch-0001"
		data.WorkerName = "reviewer"
		data.Outcome = "succeeded"
		data.Plan = &plannerRunPlanProjection{
			Revision: 3,
			Subtasks: []plannerRunSubtask{{
				ID: "0", Objective: "Inspect", Instructions: "Read the source", Status: "succeeded",
			}},
		}
	case RunEventPlannerFinishRequested:
		data.PlanRevision = &revision
		data.Outcome = string(contracts.StageSucceeded)
	case RunEventPlannerCompleted:
		data.Outcome = string(contracts.StageSucceeded)
	case RunEventPlannerFailed:
		data.Outcome = "failure"
		data.Code = "planner_gateway_unavailable"
	default:
		t.Fatalf("test has no event fixture for %q", kind)
	}
	return RunEventAppend{
		EventID: "event-" + string(kind), EventSchemaVersion: contracts.APIVersion,
		Kind: kind, Data: mustRunEventJSON(t, data),
	}
}

func mustRunEventJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}
