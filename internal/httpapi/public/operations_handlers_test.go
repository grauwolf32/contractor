package public

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/configload/configloadtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/controlplane/controlplanetest"
)

func TestOperationsSnapshotAndPagesAreCoherentStableAndObserved(t *testing.T) {
	fixture := newHandlerFixture(t)
	now := time.Date(2026, 8, 31, 1, 0, 0, 0, time.UTC)
	lease := now.Add(time.Minute)
	allocationID := "allocation-operations"
	template, err := configloadtest.AgentTemplate(fixture.configs.Snapshot(), "artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	gateway, err := fixture.configs.Snapshot().LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	policy, err := fixture.configs.Snapshot().ModelPolicy("worker@1")
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
				ModelPolicy: policy.Ref, LLMGateway: gateway.Ref,
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

	// The last returned ID disappeared and an earlier ID joined. Continue
	// past the saved key and report the new snapshot revision.
	earlier := snapshot.RuntimeAgents[0]
	earlier.InstanceID = "runtime-0"
	snapshot.RuntimeAgents[1] = earlier
	snapshot.Cursor.Revision++
	fixture.operations.set(snapshot)
	secondResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(secondResponse, authenticatedRequest(
		http.MethodGet,
		"/v1/operations/runtime-agents?limit=1&cursor="+url.QueryEscape(*first.Page.NextCursor),
		bytes.NewReader(nil),
	))
	var second runtimeAgentPageResponse
	if secondResponse.Code != http.StatusOK || json.Unmarshal(secondResponse.Body.Bytes(), &second) != nil ||
		len(second.Items) != 1 || second.Items[0].InstanceID != "runtime-z" || second.Page.HasMore || second.Cursor.Revision != "43" ||
		second.Items[0].SupportedToolsets[0].Tools[0] != "read_artifact" {
		t.Fatalf("Runtime Agent second page = %d: %+v %s", secondResponse.Code, second, secondResponse.Body.String())
	}

	continuation := url.QueryEscape(*first.Page.NextCursor)
	for _, path := range []string{
		"allocations?cursor=" + continuation,
		"runtime-agents?cursor=" + continuation + "AA",
		"runtime-agents?cursor=invalid",
	} {
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, "/v1/operations/"+path, bytes.NewReader(nil)))
		if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), `"invalid_request"`) {
			t.Fatalf("invalid Operations cursor accepted for %s: %d %s", path, response.Code, response.Body.String())
		}
	}
	for _, generation := range []string{snapshot.Cursor.Generation, "operations-generation-restarted"} {
		snapshot.Cursor.Generation = generation
		snapshot.RuntimeAgents = snapshot.RuntimeAgents[1:]
		fixture.operations.set(snapshot)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, "/v1/operations/runtime-agents?cursor="+continuation, bytes.NewReader(nil)))
		if generation == "operations-generation-restarted" {
			if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), `"invalid_request"`) {
				t.Fatalf("old generation continuation = %d %s", response.Code, response.Body.String())
			}
		} else if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"items":[],"page":{"hasMore":false}`) {
			t.Fatalf("empty last page = %d %s", response.Code, response.Body.String())
		}
	}

	allocations := httptest.NewRecorder()
	fixture.handler.ServeHTTP(allocations, authenticatedRequest(
		http.MethodGet, "/v1/operations/allocations", bytes.NewReader(nil),
	))
	if allocations.Code != http.StatusOK {
		t.Fatalf("allocation page = %d: %s", allocations.Code, allocations.Body.String())
	}
}

func TestOperationsLivePagesContinueAcrossRegistryHeartbeats(t *testing.T) {
	nextID := 0
	registry, err := controlplane.NewRegistry(controlplane.RegistryOptions{
		MonotonicNow: func() time.Duration { return 0 },
		NewID: func(prefix string) (string, error) {
			nextID++
			return fmt.Sprintf("%s%02d", prefix, nextID), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid",
		newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(services *testServices) { services.Operations = registry },
	)
	template, err := configloadtest.AgentTemplate(fixture.configs.Snapshot(), "artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	policy, err := fixture.configs.Snapshot().ModelPolicy("worker@1")
	if err != nil {
		t.Fatal(err)
	}
	gateway, err := fixture.configs.Snapshot().LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	heartbeat := func(principal, instance string, seq uint64) {
		t.Helper()
		_, err := registry.HeartbeatAuthenticated(principal, control.AgentHeartbeat{
			APIVersion: contracts.APIVersion, InstanceID: instance,
			HeartbeatSeq: seq, EchoedAckSeq: seq - 1, ObservedState: control.AgentIdle,
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	var instances, allocations []string
	for _, suffix := range []string{"a", "b", "c"} {
		instance, principal := "runtime-"+suffix, strings.Repeat(suffix, 64)
		_, err := registry.RegisterAuthenticated(controlplane.AuthenticatedPrincipal{
			RuntimeAgentID: principal, LabelRevision: 1, Labels: []string{},
		}, control.AgentRegistration{
			APIVersion: contracts.APIVersion, InstanceID: instance, SoftwareVersion: "0.1.0",
			StartedAt:  time.Date(2026, 11, 17, 0, 0, 0, 0, time.UTC),
			ControlURL: "https://" + instance + ".example:9443", A2AURL: "https://" + instance + ".example:9444",
			SupportedRuntimes: []string{"adk@1"}, SupportedSandboxProfiles: []string{"local-workdir@1"},
			SupportedRuntimeAdapters: []runtimesettings.RuntimeAdapterRef{},
			SupportedToolsets:        []control.ToolsetCapability{{Ref: "run-artifacts@1", Tools: []string{"list_artifacts", "read_artifact", "write_artifact"}}},
			ObservedState:            control.AgentIdle,
		})
		if err != nil {
			t.Fatal(err)
		}
		heartbeat(principal, instance, 1)
		heartbeat(principal, instance, 2)
		reserved, err := controlplanetest.Reserve(registry, controlplane.ReservationRequest{
			RunID: "run-" + suffix, StageExecutionID: "stage-" + suffix,
			Bindings: []controlplane.BindingRequirement{{
				LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: template,
				WorkerSessionMode: contracts.WorkerSessionIsolated,
				ContextManagement: contracts.EffectiveContextManagement(
					contracts.WorkerContextManagement{Strategy: contracts.ContextStrategyTemplate}, template),
				ExecutionConfig: controlplane.AllocationExecutionConfig{ModelPolicy: policy.Ref, LLMGateway: gateway.Ref},
			}},
		})
		if err != nil {
			t.Fatal(err)
		}
		instances = append(instances, instance)
		allocations = append(allocations, reserved[0].Grant.AllocationID)
	}
	sequence := uint64(2)
	for _, list := range []struct {
		kind, idKey string
		want        []string
	}{{"runtime-agents", "instanceId", instances}, {"allocations", "allocationId", allocations}} {
		var seen []string
		next := ""
		for index := range list.want {
			query := "/v1/operations/" + list.kind + "?limit=1"
			if next != "" {
				query += "&cursor=" + url.QueryEscape(next)
			}
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, query, bytes.NewReader(nil)))
			var page struct {
				Cursor operationsCursorResponse
				Items  []map[string]json.RawMessage
				Page   pageInfoResponse
			}
			if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil || response.Code != http.StatusOK || len(page.Items) != 1 {
				t.Fatalf("Operations %s page %d: %d %s (%v)", list.kind, index+1, response.Code, response.Body.String(), err)
			}
			var id string
			if err := json.Unmarshal(page.Items[0][list.idKey], &id); err != nil {
				t.Fatal(err)
			}
			seen = append(seen, id)
			wantCursor := registry.SnapshotOperations().Cursor
			if page.Cursor.Generation != wantCursor.Generation || page.Cursor.Revision != strconv.FormatUint(wantCursor.Revision, 10) ||
				page.Page.HasMore != (index < len(list.want)-1) {
				t.Fatalf("unexpected Operations cursor or page: %+v %+v", page.Cursor, page.Page)
			}
			if page.Page.HasMore {
				if page.Page.NextCursor == nil {
					t.Fatal("missing continuation")
				}
				next = *page.Page.NextCursor
				sequence++
				heartbeat(strings.Repeat("c", 64), "runtime-c", sequence)
			}
		}
		if !slices.Equal(seen, list.want) {
			t.Fatalf("%s IDs = %v, want %v", list.kind, seen, list.want)
		}
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
