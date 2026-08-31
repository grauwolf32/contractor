package controlplane

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestOperationsSnapshotKeepsObservedAndAuthoritativeAllocationFactsDistinct(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-operations")
	template := testTemplate(t)
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-operations", StageExecutionID: "stage-operations",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", template)},
	})
	if err != nil {
		t.Fatal(err)
	}
	allocationID := reservations[0].Grant.AllocationID

	reserved := registry.SnapshotOperations()
	if len(reserved.RuntimeAgents) != 1 || len(reserved.Allocations) != 1 ||
		reserved.RuntimeAgents[0].SlotState != SlotReserved ||
		reserved.RuntimeAgents[0].ObservedState != contracts.AgentIdle ||
		reserved.RuntimeAgents[0].AuthoritativeAllocationID == nil ||
		*reserved.RuntimeAgents[0].AuthoritativeAllocationID != allocationID ||
		reserved.RuntimeAgents[0].CurrentAllocationID != nil ||
		reserved.Allocations[0].AuthoritativePhase != AllocationPreparing ||
		reserved.Allocations[0].ObservedPhase != AllocationObservedAbsent ||
		reserved.Allocations[0].Metrics.ReportsComplete {
		t.Fatalf("reserved Operations snapshot = %+v", reserved)
	}
	if reserved.RuntimeAgents[0].SoftwareVersion != "0.1.0" ||
		reserved.RuntimeAgents[0].ConfirmedLeaseUntil == nil {
		t.Fatalf("Runtime observation lacks reported version/lease: %+v", reserved.RuntimeAgents[0])
	}

	if err := registry.SetAllocationPhase(allocationID, AllocationActive, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Heartbeat(contracts.AgentHeartbeat{
		APIVersion: contracts.APIVersion, InstanceID: "agent-operations",
		HeartbeatSeq: 3, EchoedAckSeq: 2, ObservedState: contracts.AgentAllocated,
		AllocationID: &allocationID,
	}); err != nil {
		t.Fatal(err)
	}
	active := registry.SnapshotOperations()
	if active.RuntimeAgents[0].SlotState != SlotBusy ||
		active.Allocations[0].AuthoritativePhase != AllocationActive ||
		active.Allocations[0].ObservedPhase != AllocationObservedPrepared {
		t.Fatalf("active Operations snapshot = %+v", active)
	}

	if err := registry.SetWriteFence(allocationID); err != nil {
		t.Fatal(err)
	}
	reason := SafeReason{Code: "planner_cancelled", Retryable: true}
	if err := registry.SetAllocationPhase(allocationID, AllocationAborting, &reason); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Heartbeat(contracts.AgentHeartbeat{
		APIVersion: contracts.APIVersion, InstanceID: "agent-operations",
		HeartbeatSeq: 4, EchoedAckSeq: 3, ObservedState: contracts.AgentFenced,
		AllocationID: &allocationID,
	}); err != nil {
		t.Fatal(err)
	}
	fenced := registry.SnapshotOperations()
	agent := fenced.RuntimeAgents[0]
	allocation := fenced.Allocations[0]
	if agent.ObservedState != contracts.AgentFenced || agent.SlotState != SlotFenced ||
		agent.CurrentAllocationID == nil || *agent.CurrentAllocationID != allocationID ||
		agent.AuthoritativeAllocationID == nil || *agent.AuthoritativeAllocationID != allocationID ||
		agent.ReconciliationReason == nil ||
		allocation.AuthoritativePhase != AllocationAborting ||
		allocation.ObservedPhase != AllocationObservedFenced || allocation.Reason == nil ||
		allocation.Reason.Code != "planner_cancelled" {
		t.Fatalf("fenced Operations snapshot collapsed authorities: %+v", fenced)
	}
	if fenced.Cursor.Generation != reserved.Cursor.Generation ||
		fenced.Cursor.Revision <= reserved.Cursor.Revision {
		t.Fatalf("snapshot cursor did not advance monotonically: reserved=%+v fenced=%+v", reserved.Cursor, fenced.Cursor)
	}
}

func TestOperationsSnapshotReflectsCompleteHeterogeneousAssignment(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	specialist := testRegistration("agent-a-specialist-operations")
	specialist.SupportedToolsets = append(specialist.SupportedToolsets, contracts.ToolsetCapability{
		Ref: "likec4@1", Tools: []string{"validate_likec4"},
	})
	registerReadyWith(t, registry, specialist)
	registerReady(t, registry, "agent-b-generalist-operations")
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-heterogeneous-operations", StageExecutionID: "stage-heterogeneous-operations",
		Bindings: []BindingRequirement{
			testBinding(t, "a-generic", "generic", testTemplate(t)),
			testBinding(t, "b-validator", "validator", likeC4ValidationTemplate(t)),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	snapshot := registry.SnapshotOperations()
	if len(snapshot.RuntimeAgents) != 2 || len(snapshot.Allocations) != 2 {
		t.Fatalf("heterogeneous Operations cardinality = %+v", snapshot)
	}
	assigned := make(map[string]string, len(reservations))
	for _, reservation := range reservations {
		assigned[reservation.Grant.LogicalAgentName] = reservation.Grant.RuntimeInstanceID
	}
	for _, observation := range snapshot.RuntimeAgents {
		if observation.SlotState != SlotReserved || observation.AuthoritativeAllocationID == nil {
			t.Fatalf("heterogeneous reserved agent observation = %+v", observation)
		}
	}
	if assigned["a-generic"] != "agent-b-generalist-operations" ||
		assigned["b-validator"] != "agent-a-specialist-operations" {
		t.Fatalf("heterogeneous Operations assignment = %+v", assigned)
	}
}

func TestOperationsSnapshotStoresOnlyFinalReportAggregates(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	registerReady(t, registry, "agent-report")
	template := testTemplate(t)
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-report", StageExecutionID: "stage-report",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", template)},
	})
	if err != nil {
		t.Fatal(err)
	}
	allocationID := reservations[0].Grant.AllocationID
	modelCalls, toolCalls, toolFailures := int64(2), int64(3), int64(1)
	exhausted := "tool_calls"
	now := time.Now().UTC()
	report := contracts.AllocationFinalReport{
		ReportID: "allocation-final-report", AllocationID: allocationID,
		StartedAt: now.Add(-time.Second), FinishedAt: now,
		Worker: contracts.ExecutionReport{
			ReportID: "worker-report", Complete: true,
			Metrics: contracts.ExecutionMetrics{
				ModelCalls: &modelCalls,
				Tools: map[string]contracts.ToolMetrics{
					"write_artifact": {Calls: &toolCalls, Failed: &toolFailures},
				},
				WorkerBudget: &contracts.WorkerBudgetMetrics{
					MaxModelCalls: 4, MaxToolCalls: 3, MaxTotalTokens: 100,
					ObservedModelCalls: 2, ObservedToolCalls: 3, ObservedTotalTokens: 50,
					Exhausted: &exhausted,
				},
			},
			ToolCalls: []contracts.ToolCallRecord{},
			Errors: []contracts.ExecutionError{{
				Code: "provider_error", Message: "must-not-appear-provider-body",
			}},
		},
		Runtime: contracts.RuntimeReport{Complete: true},
	}
	if err := registry.RecordAllocationReport(allocationID, report); err != nil {
		t.Fatal(err)
	}
	snapshot := registry.SnapshotOperations()
	metrics := snapshot.Allocations[0].Metrics
	if !metrics.ReportsComplete || metrics.ModelCalls != 2 || metrics.ToolCalls != 3 ||
		metrics.ToolFailures != 1 || metrics.ErrorCount != 1 ||
		snapshot.Allocations[0].ExhaustedDimension == nil ||
		*snapshot.Allocations[0].ExhaustedDimension != exhausted {
		t.Fatalf("safe allocation aggregate = %+v", snapshot.Allocations[0])
	}
	encoded, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "must-not-appear-provider-body") ||
		strings.Contains(string(encoded), "provider_error") {
		t.Fatalf("Operations snapshot leaked report detail: %s", encoded)
	}
	beforeOverflow := snapshot.Cursor
	maximum := int64(math.MaxInt64)
	overflow := report
	overflow.ReportID = "allocation-final-overflow"
	overflow.Worker.ReportID = "worker-overflow"
	overflow.Worker.Metrics = contracts.ExecutionMetrics{Tools: map[string]contracts.ToolMetrics{
		"first": {Calls: &maximum}, "second": {Calls: &maximum},
	}}
	if err := registry.RecordAllocationReport(allocationID, overflow); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("overflowing aggregate error = %v", err)
	}
	afterOverflow := registry.SnapshotOperations()
	if afterOverflow.Cursor != beforeOverflow || afterOverflow.Allocations[0].Metrics != metrics {
		t.Fatalf("overflowing aggregate mutated snapshot: %+v", afterOverflow)
	}
}

func TestOperationsSnapshotRetiresSupersededAgentAfterAllocationRelease(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	oldRegistration := testRegistration("agent-old-operations")
	registerReadyWith(t, registry, oldRegistration)
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-old-operations", StageExecutionID: "stage-old-operations",
		Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))},
	})
	if err != nil {
		t.Fatal(err)
	}
	restarted := testRegistration("agent-new-operations")
	restarted.ControlURL = oldRegistration.ControlURL
	restarted.A2AURL = oldRegistration.A2AURL
	registerReadyWith(t, registry, restarted)
	whileOwned := registry.SnapshotOperations()
	if len(whileOwned.RuntimeAgents) != 2 {
		t.Fatalf("superseded allocation owner disappeared before release: %+v", whileOwned.RuntimeAgents)
	}
	if err := registry.Release(reservations[0].Grant.AllocationID); err != nil {
		t.Fatal(err)
	}
	released := registry.SnapshotOperations()
	if len(released.RuntimeAgents) != 1 || released.RuntimeAgents[0].InstanceID != restarted.InstanceID {
		t.Fatalf("released superseded agent remained current: %+v", released.RuntimeAgents)
	}
}

func TestOperationsSnapshotIsStableOrderedAndMutationFree(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	capable := testRegistration("agent-z")
	capable.SupportedRuntimes = []string{"python@1", "adk@1"}
	capable.SupportedSandboxProfiles = []string{"remote@1", "local-workdir@1"}
	capable.SupportedToolsets[0].Tools = []string{"write_artifact", "read_artifact"}
	registerReadyWith(t, registry, capable)
	minimal := testRegistration("agent-a")
	minimal.SupportedToolsets = []contracts.ToolsetCapability{}
	registerReadyWith(t, registry, minimal)
	first := registry.SnapshotOperations()
	second := registry.SnapshotOperations()
	if first.Cursor.Generation == "" || first.Cursor != second.Cursor ||
		len(first.RuntimeAgents) != 2 || first.RuntimeAgents[0].InstanceID != "agent-a" ||
		first.RuntimeAgents[1].InstanceID != "agent-z" ||
		len(first.RuntimeAgents[0].SupportedToolsets) != 0 ||
		first.RuntimeAgents[1].SupportedRuntimes[0] != "adk@1" ||
		first.RuntimeAgents[1].SupportedSandboxProfiles[0] != "local-workdir@1" ||
		first.RuntimeAgents[1].SupportedToolsets[0].Tools[0] != "read_artifact" {
		t.Fatalf("stable ordered snapshot = first %+v second %+v", first, second)
	}
	first.RuntimeAgents[0].SoftwareVersion = "mutated"
	first.RuntimeAgents[1].SupportedRuntimes[0] = "mutated@1"
	first.RuntimeAgents[1].SupportedToolsets[0].Tools[0] = "mutated_tool"
	third := registry.SnapshotOperations()
	if third.RuntimeAgents[0].SoftwareVersion != "0.1.0" ||
		third.RuntimeAgents[1].SupportedRuntimes[0] != "adk@1" ||
		third.RuntimeAgents[1].SupportedToolsets[0].Tools[0] != "read_artifact" ||
		third.Cursor != second.Cursor || third.Validate() != nil {
		t.Fatalf("caller mutated authoritative snapshot: %+v", third)
	}
}

func TestRuntimeAgentCapabilityValidationFailsClosed(t *testing.T) {
	valid := func() RuntimeAgentObservation {
		return RuntimeAgentObservation{
			InstanceID: "agent-capabilities", SoftwareVersion: "0.1.0",
			SupportedRuntimes: []string{"adk@1"},
			SupportedToolsets: []RuntimeToolsetCapability{{
				Ref: "run-artifacts@1", Tools: []string{"read_artifact"},
			}},
			SupportedSandboxProfiles: []string{"local-workdir@1"},
			ObservedState:            contracts.AgentIdle, SlotState: SlotIdle,
		}
	}
	tests := []struct {
		name   string
		mutate func(*RuntimeAgentObservation)
	}{
		{"missing-runtime", func(agent *RuntimeAgentObservation) {
			agent.SupportedRuntimes = nil
		}},
		{"duplicate-runtime", func(agent *RuntimeAgentObservation) {
			agent.SupportedRuntimes = []string{"adk@1", "adk@1"}
		}},
		{"malformed-toolset", func(agent *RuntimeAgentObservation) {
			agent.SupportedToolsets[0].Ref = "probe-secret/path"
		}},
		{"empty-tool-list", func(agent *RuntimeAgentObservation) {
			agent.SupportedToolsets[0].Tools = nil
		}},
		{"duplicate-tool", func(agent *RuntimeAgentObservation) {
			agent.SupportedToolsets[0].Tools = []string{"read_artifact", "read_artifact"}
		}},
		{"oversized-tool-list", func(agent *RuntimeAgentObservation) {
			agent.SupportedToolsets[0].Tools = make([]string, maximumRuntimeToolsPerToolset+1)
			for index := range agent.SupportedToolsets[0].Tools {
				agent.SupportedToolsets[0].Tools[index] = fmt.Sprintf("tool_%d", index)
			}
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			agent := valid()
			test.mutate(&agent)
			err := agent.Validate()
			if err == nil || strings.Contains(err.Error(), "probe-secret") {
				t.Fatalf("invalid capability validation error = %v", err)
			}
		})
	}
}

func TestOperationsReplayIsOrderedBoundedAndNotifiesWithoutBlocking(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	initial := registry.SnapshotOperations().Cursor
	updates, cancel := registry.SubscribeOperations()
	defer cancel()
	if err := registry.InvalidateOperations(OperationsConfiguration, "config-v1"); err != nil {
		t.Fatal(err)
	}
	select {
	case <-updates:
	case <-time.After(time.Second):
		t.Fatal("Operations watcher was not notified")
	}
	changes, current, err := registry.ReplayOperations(initial)
	if err != nil || len(changes) != 1 || changes[0].Cursor.Revision != initial.Revision+1 ||
		changes[0].Resource != OperationsConfiguration || changes[0].ResourceID != "config-v1" ||
		changes[0].OccurredAt.IsZero() || current != changes[0].Cursor {
		t.Fatalf("Operations replay = (%+v, %+v, %v)", changes, current, err)
	}
	if changes, _, err := registry.ReplayOperations(current); err != nil || len(changes) != 0 {
		t.Fatalf("current Operations replay = (%+v, %v)", changes, err)
	}
	wrongGeneration := current
	wrongGeneration.Generation = "operations-another-process"
	if _, _, err := registry.ReplayOperations(wrongGeneration); !errors.Is(err, ErrOperationsGeneration) {
		t.Fatalf("generation mismatch error = %v", err)
	}
	future := current
	future.Revision++
	if _, _, err := registry.ReplayOperations(future); !errors.Is(err, ErrOperationsCursor) {
		t.Fatalf("future cursor error = %v", err)
	}

	old := current
	for range operationsHistoryLimit + 1 {
		if err := registry.InvalidateOperations(OperationsCredential, ""); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := registry.ReplayOperations(old); !errors.Is(err, ErrOperationsCursor) {
		t.Fatalf("expired Operations cursor error = %v", err)
	}
	beforeInvalid := registry.SnapshotOperations().Cursor
	if err := registry.InvalidateOperations(OperationsResource("raw"), "secret/value"); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("invalid Operations resource error = %v", err)
	}
	if afterInvalid := registry.SnapshotOperations().Cursor; afterInvalid != beforeInvalid {
		t.Fatalf("invalid Operations change advanced cursor: before=%+v after=%+v", beforeInvalid, afterInvalid)
	}

	corrupt := newTestRegistry(t, newTestClock())
	corruptStart := corrupt.SnapshotOperations().Cursor
	for range 3 {
		if err := corrupt.InvalidateOperations(OperationsCredential, ""); err != nil {
			t.Fatal(err)
		}
	}
	corrupt.mu.Lock()
	corrupt.operationsHistory = append(corrupt.operationsHistory[:1], corrupt.operationsHistory[2:]...)
	corrupt.mu.Unlock()
	if _, _, err := corrupt.ReplayOperations(corruptStart); !errors.Is(err, ErrOperationsGap) {
		t.Fatalf("Operations sequence gap error = %v", err)
	}
}
