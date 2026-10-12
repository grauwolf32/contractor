package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
)

// TestSharedCompletionDiagnosticsDecideRuntimeResponses pins what AWC-R115
// means for the Server's Runtime clients. Each shared case gets the verdict of
// the contract tests: a final or abort report and a live Worker State keep a
// valid completion object exactly, and an invalid one, including an unknown
// kind or phase, rejects the whole lifecycle response or State read.
func TestSharedCompletionDiagnosticsDecideRuntimeResponses(t *testing.T) {
	var cases []struct {
		Name  string
		Value json.RawMessage
		Valid bool
	}
	fixture := contractstest.ReadFile(t, "api", "testdata", "audit-completion", "diagnostics.json")
	if err := json.Unmarshal(fixture, &cases); err != nil {
		t.Fatal(err)
	}
	final, err := json.Marshal(reporting.AllocationFinalResponse{
		APIVersion: contracts.APIVersion, Report: testExecutionReport("allocation_1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	state := contractstest.ReadFixture(t, "valid", "agent-state-snapshot.json")
	template := testTemplate(t)
	const etag = `"contractor-agent-state-v1-7"`
	for _, item := range cases {
		t.Run(item.Name, func(t *testing.T) {
			finalBody := withCompletion(t, final, item.Value, "report", "worker")
			stateBody := withCompletion(t, state, item.Value, "state", "metrics")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				if request.Method == http.MethodGet {
					w.Header().Set("Cache-Control", "private, no-cache")
					w.Header().Set("ETag", etag)
					_, _ = w.Write(stateBody)
					return
				}
				_, _ = w.Write(finalBody)
			}))
			defer server.Close()
			client, _ := NewRuntimeControlClient(server.Client())
			reservation := testReservation(
				"allocation_1", "builder", server.URL, server.URL, template, time.Now().Add(time.Minute).UTC(),
			)
			deadline := time.Now().Add(time.Minute)
			finalized, finalizeErr := client.Finalize(context.Background(), reservation, "finalization_1", deadline)
			aborted, abortErr := client.Abort(
				context.Background(), reservation, "abort_1",
				contracts.TerminationError{Code: "run_cancelled", Message: "cancelled"}, deadline,
			)
			read, readErr := client.ReadWorkerState(
				context.Background(), stateWorkerHandle(server.URL, "allocation_1"), "",
			)
			if !item.Valid {
				var typed *WorkerStateReadError
				if finalizeErr == nil || abortErr == nil || !errors.As(readErr, &typed) ||
					typed.Code != "worker_state_response_invalid" || typed.Retryable {
					t.Fatalf("invalid completion accepted: finalize %v, abort %v, State %#v",
						finalizeErr, abortErr, readErr)
				}
				return
			}
			if finalizeErr != nil || abortErr != nil || readErr != nil || read.Snapshot == nil {
				t.Fatalf("valid completion rejected: finalize %v, abort %v, State %v",
					finalizeErr, abortErr, readErr)
			}
			for _, completion := range []*reporting.WorkerCompletionDiagnostics{
				finalized.Worker.Completion, aborted.Worker.Completion,
				read.Snapshot.State.Metrics.Completion,
			} {
				if completion == nil {
					t.Fatal("valid completion diagnostics were dropped")
				}
				encoded, err := json.Marshal(completion)
				if err != nil {
					t.Fatal(err)
				}
				contractstest.AssertSemanticJSONEqual(t, encoded, item.Value)
			}
		})
	}
}

// withCompletion returns document with the raw completion object placed in
// the object at path.
func withCompletion(t *testing.T, document []byte, completion json.RawMessage, path ...string) []byte {
	t.Helper()
	var root map[string]any
	if err := json.Unmarshal(document, &root); err != nil {
		t.Fatal(err)
	}
	target := root
	for _, key := range path {
		next, ok := target[key].(map[string]any)
		if !ok {
			t.Fatalf("document has no object at %v", path)
		}
		target = next
	}
	target["completion"] = completion
	encoded, err := json.Marshal(root)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}
