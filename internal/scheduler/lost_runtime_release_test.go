package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// Terminal recovery uses the real Registry and batch controller while the
// Scheduler store retains the durable release markers for this test.
type lostRuntimeReleaseAllocator struct {
	Allocator
	registry *controlplane.InMemoryRegistry
}

type unavailableReleaseRuntime struct {
	deferredLeaseRuntime
	releaseCalls int
}

func (r *unavailableReleaseRuntime) Release(context.Context, controlplane.Reservation) error {
	r.releaseCalls++
	return errors.New("Runtime unavailable")
}

func (a lostRuntimeReleaseAllocator) GetGrant(allocationID string) (controlplane.AllocationGrant, error) {
	return a.registry.GetGrant(allocationID)
}

func (a lostRuntimeReleaseAllocator) GetReservation(allocationID string) (controlplane.Reservation, error) {
	return a.registry.GetReservation(allocationID)
}

func TestTerminalReleaseRecoveryMarksExpiredUnreachableRuntimeReleased(t *testing.T) {
	h := newSchedulerHarness(t)
	execution := h.persistedExecution(t, runstore.StageSucceeded)
	h.store.stages = []runstore.StageExecution{execution}
	h.store.run.State = runstore.RunSucceeded
	now := h.clock.now
	var elapsed time.Duration
	registry, err := controlplane.NewRegistry(controlplane.RegistryOptions{
		Now:          func() time.Time { return now },
		MonotonicNow: func() time.Duration { return elapsed },
	})
	if err != nil {
		t.Fatal(err)
	}
	registration := control.AgentRegistration{
		APIVersion: contracts.APIVersion, InstanceID: "unreachable-runtime", SoftwareVersion: "0.1.0",
		StartedAt: now, ControlURL: "https://unreachable.example:9443", A2AURL: "https://unreachable.example:9444",
		InitialLabels: []string{}, SupportedRuntimeAdapters: []contracts.RuntimeAdapterRef{},
		SupportedRuntimes: []string{"adk@1"},
		SupportedToolsets: []control.ToolsetCapability{{
			Ref: "run-artifacts@1", Tools: []string{"list_artifacts", "read_artifact", "write_artifact"},
		}},
		SupportedSandboxProfiles: []string{"local-workdir@1"}, ObservedState: control.AgentIdle,
	}
	if _, err := registry.Register(registration); err != nil {
		t.Fatal(err)
	}
	for sequence := uint64(1); sequence <= 2; sequence++ {
		if _, err := registry.Heartbeat(control.AgentHeartbeat{
			APIVersion: contracts.APIVersion, InstanceID: registration.InstanceID,
			HeartbeatSeq: sequence, EchoedAckSeq: sequence - 1, ObservedState: control.AgentIdle,
		}); err != nil {
			t.Fatal(err)
		}
	}
	stage := h.workflow.Stages[h.workflow.EntryStage]
	binding := stage.Agents["builder"]
	reservations, err := registry.ReserveAll(controlplane.ReservationRequest{
		RunID: execution.RunID, StageExecutionID: execution.StageExecutionID,
		Bindings: []controlplane.BindingRequirement{{
			LogicalAgentName: "builder", Namespace: binding.Namespace,
			WorkerSessionMode: stage.Session, AgentTemplate: binding.Template,
			ResolvedSkills:  []contracts.ResolvedSkill{},
			ExecutionConfig: allocationExecutionConfig(stage, "builder"),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	reservation := reservations[0]
	h.store.allocations = []runstore.StageAllocation{{
		AllocationID: reservation.Grant.AllocationID, StageExecutionID: execution.StageExecutionID,
		LogicalAgentName: reservation.Grant.LogicalAgentName, Namespace: reservation.Grant.Namespace,
		RuntimeAgentInstanceID: reservation.Grant.RuntimeInstanceID,
		AgentTemplateRef:       reservation.AgentTemplate.Ref, WorkerRuntimeRef: reservation.AgentTemplate.Runtime,
	}}
	runtime := &unavailableReleaseRuntime{}
	batch, err := controlplane.NewRuntimeBatchController(runtime, registry, controlplane.RuntimeBatchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	h.scheduler.allocator = lostRuntimeReleaseAllocator{Allocator: h.allocator, registry: registry}
	h.scheduler.workers = batch
	now = now.Add(61 * time.Second)
	elapsed += 61 * time.Second

	worked, err := h.scheduler.recoverTerminalRelease(context.Background())
	if !worked || err != nil {
		t.Fatalf("lost Runtime terminal release recovery = (%t, %v)", worked, err)
	}
	if _, err := registry.GetGrant(reservation.Grant.AllocationID); !errors.Is(err, controlplane.ErrAllocationNotFound) {
		t.Fatalf("lost Runtime grant after recovery = %v", err)
	}
	if runtime.releaseCalls != 1 || h.store.allocations[0].ReleaseAttemptedAt == nil ||
		h.store.allocations[0].ReleaseCompletedAt == nil {
		t.Fatalf("durable release markers = %+v", h.store.allocations[0])
	}
	attempted := *h.store.allocations[0].ReleaseAttemptedAt
	worked, err = h.scheduler.recoverTerminalRelease(context.Background())
	if worked || err != nil || runtime.releaseCalls != 1 || !h.store.allocations[0].ReleaseAttemptedAt.Equal(attempted) {
		t.Fatalf("repeated terminal release after completion = (%t, %v), allocation=%+v", worked, err, h.store.allocations[0])
	}
}
