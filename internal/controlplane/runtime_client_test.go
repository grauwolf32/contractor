package controlplane

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRuntimeControlClientPrepareSendsExactResolvedAllocation(t *testing.T) {
	template := testTemplate(t)
	settings := testRuntimeSettings()
	lease := time.Date(2026, 8, 29, 13, 0, 0, 0, time.UTC)
	var received contracts.PrepareAllocationRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.URL.Path != "/private/v1/allocations/allocation_1/prepare" {
			t.Errorf("unexpected request %s %s", request.Method, request.URL.Path)
		}
		decoder := json.NewDecoder(request.Body)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&received); err != nil {
			t.Errorf("decode prepare request: %v", err)
		}
		response := contracts.PrepareAllocationResponse{
			APIVersion: contracts.APIVersion,
			WorkerHandle: contracts.WorkerHandle{
				AllocationID: "allocation_1", AgentTemplateRef: template.Ref,
				WorkerRuntimeRef: template.Runtime, LeaseExpiresAt: lease,
				AgentCard: testAgentCard(
					"builder", "allocation_1",
					serverURL(request)+"/private/v1/allocations/allocation_1/a2a",
				),
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	reservation := testReservation("allocation_1", "builder", server.URL, server.URL, template, lease)
	client, err := NewRuntimeControlClient(server.Client())
	if err != nil {
		t.Fatal(err)
	}

	handle, err := client.Prepare(context.Background(), reservation, settings)
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}
	if handle.AllocationID != reservation.Grant.AllocationID || received.Spec.LeaseExpiresAt != lease {
		t.Fatalf("handle/request = (%+v, %+v)", handle, received.Spec)
	}
	if received.Spec.AgentTemplate.Ref != template.Ref ||
		received.Spec.RuntimeSettings.LLMGatewayToken.Reveal() != settings.LLMGatewayToken.Reveal() {
		t.Fatalf("prepare request lost resolved inputs: %+v", received.Spec)
	}
}

func TestRuntimeControlClientPrepareRejectsSecretBearingHandle(t *testing.T) {
	template := testTemplate(t)
	settings := testRuntimeSettings()
	lease := time.Now().Add(time.Minute).UTC()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		response := contracts.PrepareAllocationResponse{
			APIVersion: contracts.APIVersion,
			WorkerHandle: contracts.WorkerHandle{
				AllocationID: "allocation_1", AgentTemplateRef: template.Ref,
				WorkerRuntimeRef: template.Runtime, LeaseExpiresAt: lease,
				AgentCard: testAgentCard(
					settings.LLMGatewayToken.Reveal(), "allocation_1", serverURL(request)+"/a2a",
				),
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	client, _ := NewRuntimeControlClient(server.Client())
	reservation := testReservation("allocation_1", "builder", server.URL, server.URL, template, lease)

	_, err := client.Prepare(context.Background(), reservation, settings)
	if err == nil || bytes.Contains([]byte(err.Error()), []byte(settings.LLMGatewayToken.Reveal())) {
		t.Fatalf("secret-bearing WorkerHandle error = %v", err)
	}
}

func TestValidateA2AAgentCardRequiresMutualTLS(t *testing.T) {
	registeredURL := "https://runtime.example"
	endpoint := "https://runtime.example/private/v1/allocations/allocation_1/a2a"
	card := testAgentCard("builder", "allocation_1", endpoint)
	delete(card, "securityRequirements")

	err := validateA2AAgentCard(card, "allocation_1", registeredURL)
	if err == nil || err.Error() != "Runtime Agent A2A Agent Card does not require mutual TLS" {
		t.Fatalf("validateA2AAgentCard error = %v", err)
	}
}

func TestValidateA2AAgentCardRejectsAnotherPathOnRegisteredOrigin(t *testing.T) {
	registeredURL := "https://runtime.example"
	card := testAgentCard(
		"builder", "allocation_1", "https://runtime.example/unrelated/allocation_1/a2a",
	)

	err := validateA2AAgentCard(card, "allocation_1", registeredURL)
	if err == nil || err.Error() != "Runtime Agent returned an A2A Agent Card for another endpoint" {
		t.Fatalf("validateA2AAgentCard error = %v", err)
	}
}

func TestPrepareAllCleansEveryReservationAfterPartialFailure(t *testing.T) {
	runtime := &recordingRuntime{prepareFailure: map[string]error{"allocation_2": errors.New("synthetic failure")}}
	registry := &recordingAllocationRegistry{}
	controller, err := NewRuntimeBatchController(runtime, registry, RuntimeBatchOptions{
		Now:            func() time.Time { return time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC) },
		NewID:          func(prefix string) (string, error) { return prefix + "cleanup", nil },
		CleanupTimeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	template := testTemplate(t)
	lease := time.Now().Add(time.Minute)
	reservations := []Reservation{
		testReservation("allocation_1", "first", "https://first.example", "https://first.example", template, lease),
		testReservation("allocation_2", "second", "https://second.example", "https://second.example", template, lease),
	}

	handles, err := controller.PrepareAll(context.Background(), reservations, testRuntimeSettings())
	if err == nil || handles != nil {
		t.Fatalf("PrepareAll = (%+v, %v), want nil/error", handles, err)
	}
	for _, allocationID := range []string{"allocation_1", "allocation_2"} {
		if !slices.Contains(runtime.aborted, allocationID) || !slices.Contains(runtime.released, allocationID) ||
			!slices.Contains(registry.fenced, allocationID) || !slices.Contains(registry.released, allocationID) {
			t.Fatalf("allocation %s was not fully cleaned: runtime=%+v registry=%+v", allocationID, runtime, registry)
		}
	}
}

func TestFinalizeAllFencesBeforeRuntimeAndReleaseRetainsFailedGrant(t *testing.T) {
	runtime := &recordingRuntime{releaseFailure: map[string]error{"allocation_2": errors.New("unavailable")}}
	registry := &recordingAllocationRegistry{}
	controller, err := NewRuntimeBatchController(runtime, registry, RuntimeBatchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	template := testTemplate(t)
	lease := time.Now().Add(time.Minute)
	reservations := []Reservation{
		testReservation("allocation_1", "first", "https://first.example", "https://first.example", template, lease),
		testReservation("allocation_2", "second", "https://second.example", "https://second.example", template, lease),
	}

	reports, err := controller.FinalizeAll(
		context.Background(), reservations, "finalization_1", time.Now().Add(time.Minute),
	)
	if err != nil || len(reports) != 2 {
		t.Fatalf("FinalizeAll = (%+v, %v)", reports, err)
	}
	if !slices.Equal(registry.fenced, []string{"allocation_1", "allocation_2"}) {
		t.Fatalf("fences = %v", registry.fenced)
	}
	if err := controller.ReleaseAll(context.Background(), reservations); err == nil {
		t.Fatal("ReleaseAll accepted a Runtime Agent release failure")
	}
	if !slices.Equal(registry.released, []string{"allocation_1"}) {
		t.Fatalf("registry releases = %v, want only confirmed Runtime release", registry.released)
	}
}

func TestRuntimeControlClientFinalizeAndReleaseProtocol(t *testing.T) {
	template := testTemplate(t)
	lease := time.Now().Add(time.Minute).UTC()
	var operations []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		operations = append(operations, request.URL.Path)
		if stringsHasSuffix(request.URL.Path, "/release") {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		response := contracts.AllocationFinalResponse{
			APIVersion: contracts.APIVersion,
			Report: contracts.ExecutionReport{
				AllocationID: "allocation_1", StartedAt: time.Now().Add(-time.Second).UTC(),
				FinishedAt: time.Now().UTC(), Complete: true, Counters: map[string]int64{},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	client, _ := NewRuntimeControlClient(server.Client())
	reservation := testReservation("allocation_1", "builder", server.URL, server.URL, template, lease)
	if _, err := client.Finalize(context.Background(), reservation, "finalization_1", time.Now().Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	if err := client.Release(context.Background(), reservation); err != nil {
		t.Fatal(err)
	}
	if len(operations) != 2 {
		t.Fatalf("operations = %v", operations)
	}
}

type recordingRuntime struct {
	mu             sync.Mutex
	prepared       []string
	finalized      []string
	aborted        []string
	released       []string
	prepareFailure map[string]error
	releaseFailure map[string]error
}

func (r *recordingRuntime) Prepare(
	_ context.Context, reservation Reservation, _ contracts.RuntimeSettings,
) (contracts.WorkerHandle, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	allocationID := reservation.Grant.AllocationID
	r.prepared = append(r.prepared, allocationID)
	if err := r.prepareFailure[allocationID]; err != nil {
		return contracts.WorkerHandle{}, err
	}
	return contracts.WorkerHandle{AllocationID: allocationID}, nil
}

func (r *recordingRuntime) Finalize(
	_ context.Context, reservation Reservation, _ string, _ time.Time,
) (contracts.ExecutionReport, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.finalized = append(r.finalized, reservation.Grant.AllocationID)
	return testExecutionReport(reservation.Grant.AllocationID), nil
}

func (r *recordingRuntime) Abort(
	_ context.Context,
	reservation Reservation,
	_ string,
	_ contracts.TerminationError,
	_ time.Time,
) (contracts.ExecutionReport, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.aborted = append(r.aborted, reservation.Grant.AllocationID)
	return testExecutionReport(reservation.Grant.AllocationID), nil
}

func (r *recordingRuntime) Release(_ context.Context, reservation Reservation) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	allocationID := reservation.Grant.AllocationID
	r.released = append(r.released, allocationID)
	return r.releaseFailure[allocationID]
}

type recordingAllocationRegistry struct {
	fenced   []string
	released []string
}

func (r *recordingAllocationRegistry) SetWriteFence(allocationID string) error {
	r.fenced = append(r.fenced, allocationID)
	return nil
}

func (r *recordingAllocationRegistry) Release(allocationID string) error {
	r.released = append(r.released, allocationID)
	return nil
}

func testReservation(
	allocationID, logicalName, controlURL, a2aURL string,
	template contracts.ResolvedAgentTemplate,
	lease time.Time,
) Reservation {
	return Reservation{
		Grant: AllocationGrant{
			AllocationID: allocationID, RuntimeInstanceID: "runtime_" + logicalName,
			RunID: "run_1", StageExecutionID: "stage_1", LogicalAgentName: logicalName,
			Namespace: logicalName, ReadPolicy: ReadCurrentRun, WritePolicy: WriteInputsAndIntermediates,
		},
		ControlURL: controlURL, A2AURL: a2aURL, AgentTemplate: template, LeaseExpiresAt: lease,
	}
}

func testRuntimeSettings() contracts.RuntimeSettings {
	return contracts.RuntimeSettings{
		LLMGatewayURL: "https://llm.example/v1", LLMGatewayToken: contracts.NewSecretString("recognizable-runtime-secret"),
		ArtifactAPIURL: "https://control.example/private/v1", RequestTimeoutSeconds: 30,
	}
}

func testExecutionReport(allocationID string) contracts.ExecutionReport {
	now := time.Now().UTC()
	return contracts.ExecutionReport{
		AllocationID: allocationID, StartedAt: now.Add(-time.Second), FinishedAt: now,
		Complete: true, Counters: map[string]int64{},
	}
}

func serverURL(request *http.Request) string {
	return "http://" + request.Host
}

func testAgentCard(name, allocationID, endpoint string) map[string]any {
	return map[string]any{
		"name": name,
		"supportedInterfaces": []any{map[string]any{
			"url": endpoint, "protocolBinding": "JSONRPC", "protocolVersion": "1.0",
			"tenant": allocationID,
		}},
		"defaultInputModes":  []any{stageContentMediaType},
		"defaultOutputModes": []any{stageContentMediaType},
		"skills": []any{map[string]any{
			"id": "contractor_stage_content", "inputModes": []any{stageContentMediaType},
			"outputModes": []any{stageContentMediaType},
		}},
		"securitySchemes": map[string]any{
			"mutualTLS": map[string]any{"mtlsSecurityScheme": map[string]any{}},
		},
		"securityRequirements": []any{map[string]any{
			"schemes": map[string]any{"mutualTLS": map[string]any{}},
		}},
	}
}

func stringsHasSuffix(value, suffix string) bool {
	return len(value) >= len(suffix) && value[len(value)-len(suffix):] == suffix
}

func (r *recordingRuntime) String() string {
	return fmt.Sprintf("prepared=%v aborted=%v released=%v", r.prepared, r.aborted, r.released)
}
