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
)

func completionBinding(t *testing.T) BindingRequirement {
	template := testTemplate(t)
	template.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "audit-results", Version: "2"}, Tools: []string{"read_audit_task", "submit_check_result"}}}
	binding := testBinding(t, "checker", "check", template)
	task, manifest := "task-r1", "manifest-r1"
	binding.CompletionContract = &contracts.WorkerCompletionContract{Kind: contracts.AuditCheckResultsV1,
		Task:              contracts.ArtifactRef{Namespace: "inputs", Name: "task", Revision: &task},
		ExecutionManifest: contracts.ArtifactRef{Namespace: "inputs", Name: "manifest", Revision: &manifest},
		ResultArtifact:    contracts.ArtifactRef{Namespace: "check", Name: "result"}}
	return binding
}

func TestAuditCompletionMixedFleetAndReservationReplay(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	for _, id := range []string{"a-old", "b-tools-only", "c-capable"} {
		registration := testRegistration(id)
		if id != "a-old" {
			registration.SupportedToolsets = append(registration.SupportedToolsets, contracts.ToolsetCapability{Ref: "audit-results@2", Tools: []string{"read_audit_task", "submit_check_result"}})
		}
		if id == "c-capable" {
			registration.Capabilities = &contracts.RuntimeCompletionCapabilities{CompletionContracts: []string{contracts.AuditCheckResultsV1}}
		}
		principal := inProcessPrincipal(id)
		if _, err := registry.RegisterAuthenticated(principal, registration); err != nil {
			t.Fatal(err)
		}
		for _, beat := range []contracts.AgentHeartbeat{heartbeat(id, 1, 0), heartbeat(id, 2, 1)} {
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
	*reservations[0].CompletionContract.Task.Revision = "caller-mutation"
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
	reservation := Reservation{Grant: AllocationGrant{Namespace: binding.Namespace}, AgentTemplate: binding.AgentTemplate, CompletionContract: binding.CompletionContract}
	if _, err := client.Prepare(context.Background(), reservation, contracts.WorkerExecutionSettings{}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("direct prepare: %v", err)
	}
}

func TestAuditCompletionPrepareSendsPinnedContract(t *testing.T) {
	binding := completionBinding(t)
	lease := time.Now().Add(time.Minute).UTC().Truncate(time.Microsecond)
	var received contracts.PrepareAllocationRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(contracts.PrepareAllocationResponse{APIVersion: contracts.APIVersion, WorkerHandle: contracts.WorkerHandle{
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
	_, err = client.Prepare(context.Background(), reservation, contracts.WorkerExecutionSettings{ModelPolicy: binding.AgentTemplate.ModelPolicy, RuntimeSettings: testRuntimeSettings(), ResolvedRuntimeConfigProvenance: testRuntimeProvenance()})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(received.Spec.CompletionContract, binding.CompletionContract) {
		t.Fatal("wire contract changed")
	}
}
