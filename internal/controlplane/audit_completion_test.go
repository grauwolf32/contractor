package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

func completionBinding(t *testing.T) BindingRequirement {
	template := testTemplate(t)
	template.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "audit-results", Version: "2"}, Tools: []string{"read_audit_task", "submit_result", "submit_results", "evidence"}}}
	binding := testBinding(t, "checker", "check", template)
	task := "context-r1"
	binding.CompletionContract = &contracts.WorkerCompletionContract{Kind: contracts.AuditCheckResultsV1,
		Context:        contracts.ArtifactRef{Namespace: "inputs", Name: "context", Revision: &task},
		ResultArtifact: contracts.ArtifactRef{Namespace: "check", Name: "result"}}
	return binding
}

func TestAuditCompletionMixedFleetAndReservationReplay(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	for _, id := range []string{"a-old", "b-tools-only", "c-capable"} {
		registration := testRegistration(id)
		if id != "a-old" {
			registration.SupportedToolsets = append(registration.SupportedToolsets, control.ToolsetCapability{Ref: "audit-results@2", Tools: []string{"read_audit_task", "submit_result", "submit_results", "evidence"}})
		}
		if id == "c-capable" {
			registration.Capabilities = &contracts.RuntimeCompletionCapabilities{CompletionContracts: []string{contracts.AuditCheckResultsV1}}
		}
		principal := inProcessPrincipal(id)
		if _, err := registry.RegisterAuthenticated(principal, registration); err != nil {
			t.Fatal(err)
		}
		for _, beat := range []control.AgentHeartbeat{heartbeat(id, 1, 0), heartbeat(id, 2, 1)} {
			if _, err := registry.HeartbeatAuthenticated(principal.RuntimeAgentID, beat); err != nil {
				t.Fatal(err)
			}
		}
	}
	binding := completionBinding(t)
	request := ReservationRequest{RunID: "run", StageExecutionID: "stage", Bindings: []BindingRequirement{binding}}
	reservations, err := registry.ReserveAll(request)
	if err != nil || len(reservations) != 1 {
		t.Fatalf("reserve: %v %v", reservations, err)
	}
	if reservations[0].Grant.RuntimeInstanceID != "c-capable" {
		t.Fatal("unsupported Runtime selected")
	}
	again, err := registry.ReserveAll(request)
	if err != nil || !reflect.DeepEqual(again, reservations) {
		t.Fatal("replay changed reservation", err)
	}
	*reservations[0].CompletionContract.Context.Revision = "caller-mutation"
	reservations[0].CompletionCapabilities.CompletionContracts[0] = "changed"
	stored, err := registry.GetReservation(again[0].Grant.AllocationID)
	if err != nil || !reflect.DeepEqual(stored, again[0]) {
		t.Fatal("reservation was not deeply cloned", err)
	}
	request.StageExecutionID = "next-stage"
	if _, err := registry.ReserveAll(request); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("unsupported fleet: %v", err)
	}
	request.Bindings[0].CompletionContract = nil
	if _, err := registry.ReserveAll(request); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("untrusted v2 toolset: %v", err)
	}
}

func TestAuditCompletionDirectPrepareRejectsUnsupportedBeforeHTTP(t *testing.T) {
	client, err := NewRuntimeControlClient(&http.Client{})
	if err != nil {
		t.Fatal(err)
	}
	binding := completionBinding(t)
	reservation := Reservation{Grant: AllocationGrant{Namespace: binding.Namespace}, AgentTemplate: binding.AgentTemplate, CompletionContract: binding.CompletionContract,
		ContextManagement: binding.ContextManagement}
	if _, err := client.Prepare(context.Background(), reservation, runtimesettings.WorkerExecutionSettings{}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("direct prepare: %v", err)
	}
}

func TestAuditCompletionPrepareSendsPinnedContract(t *testing.T) {
	binding := completionBinding(t)
	lease := time.Now().Add(time.Minute).UTC().Truncate(time.Microsecond)
	var received control.PrepareAllocationRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(control.PrepareAllocationResponse{APIVersion: contracts.APIVersion, WorkerHandle: contracts.WorkerHandle{
			AllocationID: "allocation", AgentTemplateRef: binding.AgentTemplate.Ref, WorkerRuntimeRef: binding.AgentTemplate.Runtime, LeaseExpiresAt: lease,
			AgentCard: testAgentCard("checker", "allocation", serverURL(r)+"/private/v1/allocations/allocation/a2a")}})
	}))
	defer server.Close()
	reservation := testReservation("allocation", "checker", server.URL, server.URL, binding.AgentTemplate, lease)
	reservation.Grant.Namespace = binding.Namespace
	reservation.CompletionContract = binding.CompletionContract
	reservation.CompletionCapabilities = &contracts.RuntimeCompletionCapabilities{CompletionContracts: []string{contracts.AuditCheckResultsV1}}
	client, err := NewRuntimeControlClient(server.Client())
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Prepare(context.Background(), reservation, runtimesettings.WorkerExecutionSettings{ModelPolicy: testWorkerPolicy(t), RuntimeSettings: testRuntimeSettings(), ResolvedRuntimeConfigProvenance: testRuntimeProvenance()})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(received.Spec.CompletionContract, binding.CompletionContract) {
		t.Fatal("wire contract changed")
	}
}

func TestCompletionCapabilitySnapshotsCannotChangePlacement(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	registration := testRegistration("capable")
	registration.Capabilities = &contracts.RuntimeCompletionCapabilities{CompletionContracts: []string{contracts.AuditCheckResultsV1}}
	registration.SupportedToolsets = append(registration.SupportedToolsets, control.ToolsetCapability{Ref: "audit-results@2", Tools: []string{"read_audit_task", "submit_result", "submit_results", "evidence"}})
	returned, err := registry.Register(registration)
	if err != nil {
		t.Fatal(err)
	}
	registerReadyWith(t, registry, registration)
	returned.Registration.Capabilities.CompletionContracts[0] = "changed-registration-result"
	candidates := registry.PlacementCandidates()
	if got := candidates[0].Registration.Capabilities.CompletionContracts[0]; got != contracts.AuditCheckResultsV1 {
		t.Fatalf("registration result changed frozen capability: %s", got)
	}
	candidates[0].Registration.Capabilities.CompletionContracts[0] = "changed-placement-result"
	request := ReservationRequest{RunID: "run-clone", StageExecutionID: "stage-clone", Bindings: []BindingRequirement{completionBinding(t)}}
	reservations, err := registry.ReserveAll(request)
	if err != nil || len(reservations) != 1 {
		t.Fatalf("snapshot changed completion placement: count=%d err=%v", len(reservations), err)
	}
}
