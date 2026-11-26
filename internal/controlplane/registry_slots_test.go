package controlplane

import (
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func slotRequest(t *testing.T, stage string, count int) ReservationRequest {
	t.Helper()
	bindings := make([]BindingRequirement, count)
	for index := range count {
		name := fmt.Sprintf("worker%d", index)
		bindings[index] = testBinding(t, name, name, testTemplate(t))
	}
	return ReservationRequest{RunID: "run-slots", StageExecutionID: stage, Bindings: bindings}
}

func TestRuntimeSlotsReleaseRequiresAddressedAcknowledgement(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	registration := testRegistration("runtime-four")
	registration.AllocationSlots = 4
	registerReadyWith(t, registry, registration)
	placed, err := registry.ReserveAll(slotRequest(t, "stage-four", 4))
	if err != nil {
		t.Fatal(err)
	}
	observations := make([]control.ObservedAllocation, 4)
	for index, item := range placed {
		if item.Grant.RuntimeInstanceID != registration.InstanceID {
			t.Fatal("assignment escaped process")
		}
		observations[index] = control.ObservedAllocation{AllocationID: item.Grant.AllocationID, ObservedState: control.AgentAllocated}
	}
	beat := heartbeat(registration.InstanceID, 3, 2)
	beat.ObservedState = control.AgentAllocated
	beat.Allocations = observations
	if _, err := registry.Heartbeat(beat); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.ReserveAll(slotRequest(t, "stage-fifth", 1)); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("fifth allocation: %v", err)
	}
	if err := registry.SetWriteFence(placed[0].Grant.AllocationID); err != nil {
		t.Fatal(err)
	}
	if err := registry.Release(placed[0].Grant.AllocationID); err != nil {
		t.Fatal(err)
	}
	beat.HeartbeatSeq = 4
	beat.EchoedAckSeq = 3
	beat.Allocations = observations[1:]
	response, err := registry.Heartbeat(beat)
	if err != nil {
		t.Fatal(err)
	}
	if len(response.AllocationActions) != 1 || response.AllocationActions[0].AllocationID != placed[0].Grant.AllocationID || response.AllocationActions[0].Action != control.ActionRelease {
		t.Fatalf("release reply: %+v", response)
	}
	if _, err := registry.ReserveAll(slotRequest(t, "stage-before-echo", 1)); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("unconfirmed release reused: %v", err)
	}
	beat.HeartbeatSeq = 5
	beat.EchoedAckSeq = 9999
	if _, err := registry.Heartbeat(beat); err != nil {
		t.Fatal(err)
	}
	if registry.SnapshotOperations().RuntimeAgents[0].AvailableSlots != 0 {
		t.Fatal("unissued acknowledgement released capacity")
	}
	beat.HeartbeatSeq = 6
	beat.EchoedAckSeq = 4
	if _, err := registry.Heartbeat(beat); err != nil {
		t.Fatal(err)
	}
	snapshot := registry.SnapshotOperations()
	if err := snapshot.Validate(); err != nil {
		t.Fatal(err)
	}
	if agent := snapshot.RuntimeAgents[0]; agent.AvailableSlots != 1 || agent.BusySlots != 3 || agent.AllocationSlots != 4 {
		t.Fatalf("capacity: %+v", agent)
	}
	if _, err := registry.ReserveAll(slotRequest(t, "stage-fifth", 1)); err != nil {
		t.Fatal(err)
	}
	for _, item := range placed[1:] {
		grant, err := registry.GetGrant(item.Grant.AllocationID)
		if err != nil || grant.Lost || grant.WriteFenced {
			t.Fatalf("neighbor lost: %+v %v", grant, err)
		}
	}
}

func TestRuntimeSlotsMixedCapacityAndAtomicStage(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	for _, capacity := range []int{1, 2, 4} {
		registration := testRegistration(fmt.Sprintf("runtime-%d", capacity))
		registration.AllocationSlots = capacity
		registerReadyWith(t, registry, registration)
	}
	if _, err := registry.ReserveAll(slotRequest(t, "stage-too-large", 8)); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatal(err)
	}
	if len(registry.SnapshotOperations().Allocations) != 0 {
		t.Fatal("partial Stage reservation")
	}
	placed, err := registry.ReserveAll(slotRequest(t, "stage-seven", 7))
	if err != nil {
		t.Fatal(err)
	}
	counts := map[string]int{}
	for _, item := range placed {
		counts[item.Grant.RuntimeInstanceID]++
	}
	for _, capacity := range []int{1, 2, 4} {
		if counts[fmt.Sprintf("runtime-%d", capacity)] != capacity {
			t.Fatalf("mixed capacity: %+v", counts)
		}
	}
}

func TestRuntimeSlotsCapacityFrozenPerIncarnationAndLeaseLossFansOut(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registration := testRegistration("runtime-original")
	registration.AllocationSlots = 4
	registerReadyWith(t, registry, registration)
	placed, err := registry.ReserveAll(slotRequest(t, "stage-four", 4))
	if err != nil {
		t.Fatal(err)
	}
	changed := registration.Clone()
	changed.AllocationSlots = 1
	if _, err := registry.Register(changed); !errors.Is(err, ErrRegistrationConflict) {
		t.Fatalf("capacity drift: %v", err)
	}
	clock.Advance(61 * time.Second)
	losses := registry.PollAllocationLosses()
	if len(losses) != 4 {
		t.Fatalf("lease loss did not cover all allocations: %+v", losses)
	}
	changed.InstanceID = "runtime-restarted"
	changed.StartedAt = changed.StartedAt.Add(time.Minute)
	if _, err := registry.Register(changed); err != nil {
		t.Fatalf("restart with smaller capacity: %v", err)
	}
	for _, item := range placed {
		if removed, err := registry.ReleaseLost(item.Grant.AllocationID); err != nil || !removed {
			t.Fatalf("release predecessor: %t %v", removed, err)
		}
	}
	for _, beat := range []control.AgentHeartbeat{heartbeat(changed.InstanceID, 1, 0), heartbeat(changed.InstanceID, 2, 1)} {
		if _, err := registry.Heartbeat(beat); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := registry.ReserveAll(slotRequest(t, "stage-restarted", 2)); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("restart exceeded reduced capacity: %v", err)
	}
	if _, err := registry.ReserveAll(slotRequest(t, "stage-restarted-one", 1)); err != nil {
		t.Fatal(err)
	}
}

func TestRuntimeSlotsPersistWholeMultiWorkerStageInOneProcess(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPlacementPool(t, ctx)
	fixture := newPlacementFixture(t, ctx, pool, nil)
	registration := fixture.candidateRegistration(nil, nil)
	registration.AllocationSlots = 4
	fixture.registerProcess(t, ctx, "runtime-four", "4", nil, registration)
	request := fixture.request()
	binding := request.Bindings[0]
	request.Bindings = nil
	for index := range 5 {
		item := binding
		item.LogicalAgentName, item.Namespace = fmt.Sprintf("worker%d", index), fmt.Sprintf("worker%d", index)
		request.Bindings = append(request.Bindings, item)
	}
	if _, err := fixture.allocator.ReserveAllContext(ctx, request); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("oversized Stage: %v", err)
	}
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	allocations, err := store.ListStageAllocations(ctx, fixture.stageExecutionID)
	if err != nil || len(allocations) != 0 || len(fixture.registry.SnapshotOperations().Allocations) != 0 {
		t.Fatalf("partial Stage leaked: %+v %v", allocations, err)
	}
	request.Bindings = request.Bindings[:4]
	placed, err := fixture.allocator.ReserveAllContext(ctx, request)
	if err != nil || len(placed) != 4 {
		t.Fatalf("four-Worker Stage: %+v %v", placed, err)
	}
	allocations, err = store.ListStageAllocations(ctx, fixture.stageExecutionID)
	if err != nil || len(allocations) != 4 {
		t.Fatalf("durable Stage: %+v %v", allocations, err)
	}
	ids := map[string]bool{}
	for _, allocation := range allocations {
		if allocation.RuntimeAgentInstanceID != "runtime-four" || ids[allocation.AllocationID] || allocation.RuntimeConfiguration == nil {
			t.Fatalf("invalid durable placement: %+v", allocation)
		}
		ids[allocation.AllocationID] = true
	}
}
