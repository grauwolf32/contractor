package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/controlplane"
)

func TestOperationsSnapshotAndPagesAreCoherentStableAndObserved(t *testing.T) {
	fixture := newHandlerFixture(t)
	now := time.Date(2026, 8, 31, 1, 0, 0, 0, time.UTC)
	lease := now.Add(time.Minute)
	allocationID := "allocation-operations"
	template, err := fixture.configs.Snapshot().AgentTemplate("artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	gateway, err := fixture.configs.Snapshot().LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	snapshot := controlplane.OperationsSnapshot{
		Cursor: controlplane.OperationsCursor{
			Generation: "operations-generation-fixed", Revision: 42,
		},
		RuntimeAgents: []controlplane.RuntimeAgentObservation{
			{
				InstanceID: "runtime-z", SoftwareVersion: "0.1.0",
				SupportedRuntimes: []string{"adk@1"},
				SupportedToolsets: []controlplane.RuntimeToolsetCapability{{
					Ref: "run-artifacts@1", Tools: []string{"write_artifact", "read_artifact"},
				}},
				SupportedSandboxProfiles: []string{"local-workdir@1"},
				ObservedState:            "idle", SlotState: controlplane.SlotIdle,
				LastAcceptedHeartbeat: &now, ConfirmedLeaseUntil: &lease,
			},
			{
				InstanceID: "runtime-a", SoftwareVersion: "0.1.0",
				SupportedRuntimes: []string{"adk@1"},
				SupportedToolsets: []controlplane.RuntimeToolsetCapability{{
					Ref: "run-artifacts@1", Tools: []string{"read_artifact"},
				}},
				SupportedSandboxProfiles: []string{"local-workdir@1"},
				ObservedState:            "fenced", SlotState: controlplane.SlotFenced,
				LastAcceptedHeartbeat: &now, ConfirmedLeaseUntil: &lease,
				CurrentAllocationID: &allocationID, AuthoritativeAllocationID: &allocationID,
				ReconciliationReason: &controlplane.SafeReason{
					Code: "reconciliation_required", Retryable: true,
				},
			},
		},
		Allocations: []controlplane.AllocationObservation{{
			AllocationID: allocationID, RunID: "run-operations",
			StageExecutionID: "stage-operations", RuntimeAgentInstanceID: "runtime-a",
			LogicalWorker: "builder", AgentTemplate: template.Ref,
			ExecutionConfig: controlplane.AllocationExecutionConfig{
				ModelPolicy: template.ModelPolicy.Ref, LLMGateway: gateway.Ref,
			},
			AuthoritativePhase: controlplane.AllocationAborting,
			ObservedPhase:      controlplane.AllocationObservedFenced,
			Reason:             &controlplane.SafeReason{Code: "planner_cancelled", Retryable: true},
			Metrics:            controlplane.MetricsSummary{ReportsComplete: false},
		}},
	}
	fixture.operations.set(snapshot)

	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, authenticatedRequest(
		http.MethodGet, "/v1/operations/snapshot", bytes.NewReader(nil),
	))
	if response.Code != http.StatusOK {
		t.Fatalf("Operations snapshot = %d: %s", response.Code, response.Body.String())
	}
	var full operationsSnapshotResponse
	if err := json.Unmarshal(response.Body.Bytes(), &full); err != nil {
		t.Fatal(err)
	}
	if full.Cursor.Generation != snapshot.Cursor.Generation || full.Cursor.Revision != "42" ||
		len(full.RuntimeAgents) != 2 || full.RuntimeAgents[0].InstanceID != "runtime-a" ||
		full.RuntimeAgents[0].ObservedState != "fenced" ||
		len(full.RuntimeAgents[0].SupportedToolsets) != 1 ||
		len(full.RuntimeAgents[0].SupportedToolsets[0].Tools) != 1 ||
		full.RuntimeAgents[0].AuthoritativeAllocationID == nil ||
		full.RuntimeAgents[0].CurrentAllocationID == nil ||
		len(full.Allocations) != 1 ||
		full.Allocations[0].ObservedPhase != controlplane.AllocationObservedFenced {
		t.Fatalf("Operations response collapsed or reordered facts: %+v", full)
	}
	if snapshot.RuntimeAgents[0].SupportedToolsets[0].Tools[0] != "write_artifact" {
		t.Fatal("Operations response normalization mutated the source snapshot")
	}

	firstResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(firstResponse, authenticatedRequest(
		http.MethodGet, "/v1/operations/runtime-agents?limit=1", bytes.NewReader(nil),
	))
	if firstResponse.Code != http.StatusOK {
		t.Fatalf("Runtime Agent first page = %d: %s", firstResponse.Code, firstResponse.Body.String())
	}
	var first runtimeAgentPageResponse
	if err := json.Unmarshal(firstResponse.Body.Bytes(), &first); err != nil {
		t.Fatal(err)
	}
	if len(first.Items) != 1 || first.Items[0].InstanceID != "runtime-a" ||
		!first.Page.HasMore || first.Page.NextCursor == nil {
		t.Fatalf("Runtime Agent first page = %+v", first)
	}

	secondResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(secondResponse, authenticatedRequest(
		http.MethodGet,
		"/v1/operations/runtime-agents?limit=1&cursor="+url.QueryEscape(*first.Page.NextCursor),
		bytes.NewReader(nil),
	))
	var second runtimeAgentPageResponse
	if secondResponse.Code != http.StatusOK || json.Unmarshal(secondResponse.Body.Bytes(), &second) != nil ||
		len(second.Items) != 1 || second.Items[0].InstanceID != "runtime-z" || second.Page.HasMore ||
		second.Items[0].SupportedToolsets[0].Tools[0] != "read_artifact" {
		t.Fatalf("Runtime Agent second page = %d: %+v %s", secondResponse.Code, second, secondResponse.Body.String())
	}

	snapshot.Cursor.Revision++
	fixture.operations.set(snapshot)
	stale := httptest.NewRecorder()
	fixture.handler.ServeHTTP(stale, authenticatedRequest(
		http.MethodGet,
		"/v1/operations/runtime-agents?limit=1&cursor="+url.QueryEscape(*first.Page.NextCursor),
		bytes.NewReader(nil),
	))
	if stale.Code != http.StatusBadRequest {
		t.Fatalf("stale Operations cursor = %d: %s", stale.Code, stale.Body.String())
	}

	allocations := httptest.NewRecorder()
	fixture.handler.ServeHTTP(allocations, authenticatedRequest(
		http.MethodGet, "/v1/operations/allocations", bytes.NewReader(nil),
	))
	if allocations.Code != http.StatusOK {
		t.Fatalf("allocation page = %d: %s", allocations.Code, allocations.Body.String())
	}
}

func TestOperationsRoutesRejectMutationAndUnexpectedQuery(t *testing.T) {
	fixture := newHandlerFixture(t)
	for _, test := range []struct {
		method string
		target string
		status int
	}{
		{http.MethodPost, "/v1/operations/snapshot", http.StatusMethodNotAllowed},
		{http.MethodDelete, "/v1/operations/runtime-agents", http.StatusMethodNotAllowed},
		{http.MethodPost, "/v1/operations/allocations", http.StatusMethodNotAllowed},
		{http.MethodGet, "/v1/operations/snapshot?force=true", http.StatusBadRequest},
		{http.MethodGet, "/v1/operations/allocations?unknown=true", http.StatusBadRequest},
	} {
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, authenticatedRequest(
			test.method, test.target, bytes.NewReader(nil),
		))
		if response.Code != test.status {
			t.Errorf("%s %s = %d, want %d: %s", test.method, test.target, response.Code, test.status, response.Body.String())
		}
	}
}
