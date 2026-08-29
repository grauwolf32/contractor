package controlplane

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestConfirmedLeaseRequiresEchoOfIssuedAck(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registration := testRegistration("agent-1")
	if _, err := registry.Register(registration); err != nil {
		t.Fatal(err)
	}

	first := heartbeat("agent-1", 1, 0)
	response1, err := registry.Heartbeat(first)
	if err != nil || response1.AckSeq != 1 {
		t.Fatalf("first heartbeat = (%+v, %v)", response1, err)
	}
	afterFirst, _ := registry.GetAgent("agent-1")
	if !afterFirst.ConfirmedLeaseExpiresAt.IsZero() {
		t.Fatalf("unconfirmed first heartbeat established lease %s", afterFirst.ConfirmedLeaseExpiresAt)
	}

	clock.Advance(5 * time.Second)
	if _, err := registry.Heartbeat(heartbeat("agent-1", 2, 1)); err != nil {
		t.Fatal(err)
	}
	confirmed, _ := registry.GetAgent("agent-1")
	wantExpiry := clock.Now().Add(time.Minute)
	if confirmed.LastConfirmedAckSeq != 1 || !confirmed.ConfirmedLeaseExpiresAt.Equal(wantExpiry) {
		t.Fatalf("confirmed lease = ack %d, expiry %s; want ack 1, expiry %s", confirmed.LastConfirmedAckSeq, confirmed.ConfirmedLeaseExpiresAt, wantExpiry)
	}

	clock.Advance(5 * time.Second)
	replayed, err := registry.Heartbeat(heartbeat("agent-1", 2, 1))
	if err != nil || replayed.AckSeq != 2 {
		t.Fatalf("replayed heartbeat = (%+v, %v)", replayed, err)
	}
	if _, err := registry.Heartbeat(heartbeat("agent-1", 1, 0)); err != nil {
		t.Fatalf("cached out-of-order heartbeat: %v", err)
	}
	if _, err := registry.Heartbeat(heartbeat("agent-1", 3, 1)); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Heartbeat(heartbeat("agent-1", 4, 999)); err != nil {
		t.Fatal(err)
	}
	unchanged, _ := registry.GetAgent("agent-1")
	if !unchanged.ConfirmedLeaseExpiresAt.Equal(wantExpiry) || unchanged.LastConfirmedAckSeq != 1 {
		t.Fatalf("old/unknown ack advanced lease: %+v", unchanged)
	}

	clock.Advance(5 * time.Second)
	if _, err := registry.Heartbeat(heartbeat("agent-1", 5, 3)); err != nil {
		t.Fatal(err)
	}
	advanced, _ := registry.GetAgent("agent-1")
	if advanced.LastConfirmedAckSeq != 3 || !advanced.ConfirmedLeaseExpiresAt.Equal(clock.Now().Add(time.Minute)) {
		t.Fatalf("new echoed ack did not advance lease: %+v", advanced)
	}
}

func TestReserveAllIsAtomicWithInsufficientCapacity(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-1")
	template := testTemplate(t)
	_, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-1", StageExecutionID: "stage-two",
		Bindings: []BindingRequirement{
			{LogicalAgentName: "first", Namespace: "first", AgentTemplate: template},
			{LogicalAgentName: "second", Namespace: "second", AgentTemplate: template},
		},
	})
	if !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("two-slot reservation error = %v", err)
	}
	reservation, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-1", StageExecutionID: "stage-one",
		Bindings: []BindingRequirement{{LogicalAgentName: "only", Namespace: "only", AgentTemplate: template}},
	})
	if err != nil || len(reservation) != 1 || reservation[0].Grant.RuntimeInstanceID != "agent-1" {
		t.Fatalf("slot was mutated by failed reservation: (%+v, %v)", reservation, err)
	}
}

func TestReserveAllRejectsRunReservedAgentNamespace(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	_, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-1", StageExecutionID: "stage-1",
		Bindings: []BindingRequirement{{
			LogicalAgentName: "builder", Namespace: "inputs", AgentTemplate: testTemplate(t),
		}},
	})
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("reserved namespace error = %v", err)
	}
}

func TestConcurrentReserveAllNeverAssignsSlotTwice(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-1")
	registerReady(t, registry, "agent-2")
	template := testTemplate(t)
	bindings := []BindingRequirement{
		{LogicalAgentName: "first", Namespace: "first", AgentTemplate: template},
		{LogicalAgentName: "second", Namespace: "second", AgentTemplate: template},
	}
	type result struct {
		reservations []Reservation
		err          error
	}
	results := make(chan result, 2)
	var start sync.WaitGroup
	start.Add(1)
	for index := range 2 {
		go func(index int) {
			start.Wait()
			reservations, err := registry.ReserveAll(ReservationRequest{
				RunID: fmt.Sprintf("run-%d", index), StageExecutionID: fmt.Sprintf("stage-%d", index), Bindings: bindings,
			})
			results <- result{reservations, err}
		}(index)
	}
	start.Done()
	var success result
	failures := 0
	for range 2 {
		current := <-results
		if current.err == nil {
			success = current
		} else if errors.Is(current.err, ErrInsufficientCapacity) {
			failures++
		} else {
			t.Fatalf("unexpected reservation error: %v", current.err)
		}
	}
	if len(success.reservations) != 2 || failures != 1 {
		t.Fatalf("concurrent outcomes = success %+v, failures %d", success, failures)
	}
	if success.reservations[0].Grant.RuntimeInstanceID == success.reservations[1].Grant.RuntimeInstanceID {
		t.Fatal("one Runtime Agent slot was assigned twice")
	}
}

func TestReservationRetryReturnsSameAllocationsAndGrantLifecycleIsExact(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-1")
	registerReady(t, registry, "agent-2")
	request := ReservationRequest{
		RunID: "run-1", StageExecutionID: "stage-1",
		Bindings: []BindingRequirement{{LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: testTemplate(t)}},
	}
	first, err := registry.ReserveAll(request)
	if err != nil {
		t.Fatal(err)
	}
	retry, err := registry.ReserveAll(request)
	if err != nil || retry[0].Grant.AllocationID != first[0].Grant.AllocationID || retry[0].Grant.RuntimeInstanceID != first[0].Grant.RuntimeInstanceID {
		t.Fatalf("reservation retry = (%+v, %v), first %+v", retry, err, first)
	}
	allocationID := first[0].Grant.AllocationID
	grant, err := registry.GetGrant(allocationID)
	if err != nil || grant.ReadPolicy != ReadCurrentRun || grant.WritePolicy != WriteInputsAndIntermediates || grant.WriteFenced {
		t.Fatalf("active grant = (%+v, %v)", grant, err)
	}
	if err := registry.SetWriteFence("wrong-allocation"); !errors.Is(err, ErrAllocationNotFound) {
		t.Fatalf("wrong fence error = %v", err)
	}
	if err := registry.SetWriteFence(allocationID); err != nil {
		t.Fatal(err)
	}
	grant, _ = registry.GetGrant(allocationID)
	if !grant.WriteFenced {
		t.Fatal("write fence was not visible in active grant")
	}
	if err := registry.Release("wrong-allocation"); !errors.Is(err, ErrAllocationNotFound) {
		t.Fatalf("wrong release error = %v", err)
	}
	if err := registry.Release(allocationID); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.GetGrant(allocationID); !errors.Is(err, ErrAllocationNotFound) {
		t.Fatalf("released grant lookup error = %v", err)
	}
	if _, err := registry.ReserveAll(request); !errors.Is(err, ErrReservationReleased) {
		t.Fatalf("released reservation retry error = %v", err)
	}
}

func TestCapabilityMatchingRequiresExactRefsAndSelectedTools(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registration := testRegistration("agent-1")
	registration.SupportedToolsets[0].Tools = []string{"read_artifact"}
	registerReadyWith(t, registry, registration)
	_, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-1", StageExecutionID: "stage-1",
		Bindings: []BindingRequirement{{LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: testTemplate(t)}},
	})
	if !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("missing selected tool reservation error = %v", err)
	}
}

func TestInjectedOrderingIsDeterministic(t *testing.T) {
	clock := newTestClock()
	var counter atomic.Uint64
	registry, err := NewRegistry(RegistryOptions{
		Now: clock.Now, MonotonicNow: clock.MonotonicNow,
		NewID: func(prefix string) (string, error) {
			return fmt.Sprintf("%s%d", prefix, counter.Add(1)), nil
		},
		AgentOrderKey: func(registration contracts.AgentRegistration) string {
			if registration.InstanceID == "agent-2" {
				return "first"
			}
			return "second"
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	registerReady(t, registry, "agent-1")
	registerReady(t, registry, "agent-2")
	reservation, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-order", StageExecutionID: "stage-order",
		Bindings: []BindingRequirement{{LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: testTemplate(t)}},
	})
	if err != nil || reservation[0].Grant.RuntimeInstanceID != "agent-2" {
		t.Fatalf("ordered reservation = (%+v, %v)", reservation, err)
	}
}

func TestRegistrationNeverAdoptsObservedAllocation(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	registration := testRegistration("agent-orphan")
	orphan := "allocation-from-old-control-plane"
	registration.ObservedState = contracts.AgentFenced
	registration.AllocationID = &orphan
	snapshot, err := registry.Register(registration)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.AuthoritativeAllocationID != nil || !snapshot.ReconciliationRequired {
		t.Fatalf("orphan registration was adopted: %+v", snapshot)
	}
	heartbeatRequest := contracts.AgentHeartbeat{
		APIVersion: contracts.APIVersion, InstanceID: registration.InstanceID,
		HeartbeatSeq: 1, ObservedState: contracts.AgentFenced, AllocationID: &orphan,
	}
	response, err := registry.Heartbeat(heartbeatRequest)
	if err != nil || response.Action != contracts.ActionRelease || response.AllocationID == nil || *response.AllocationID != orphan {
		t.Fatalf("orphan reconciliation response = (%+v, %v)", response, err)
	}
}

func TestLeaseExpiryAfterResponsePartitionIsIrreversible(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-1")
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-lease", StageExecutionID: "stage-lease",
		Bindings: []BindingRequirement{{
			LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: testTemplate(t),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	allocationID := reservations[0].Grant.AllocationID
	// Ack 2 was received before the simulated response-only partition. Every
	// later request repeats it and therefore cannot renew the confirmed lease.
	allocated := func(sequence, echoed uint64) contracts.AgentHeartbeat {
		return contracts.AgentHeartbeat{
			APIVersion: contracts.APIVersion, InstanceID: "agent-1",
			HeartbeatSeq: sequence, EchoedAckSeq: echoed,
			ObservedState: contracts.AgentAllocated, AllocationID: &allocationID,
		}
	}
	if _, err := registry.Heartbeat(allocated(3, 2)); err != nil {
		t.Fatal(err)
	}
	for sequence := uint64(4); sequence <= 8; sequence++ {
		clock.Advance(10 * time.Second)
		if _, err := registry.Heartbeat(allocated(sequence, 2)); err != nil {
			t.Fatal(err)
		}
	}
	if losses := registry.PollAllocationLosses(); len(losses) != 0 {
		t.Fatalf("lease expired early: %+v", losses)
	}
	clock.Advance(10 * time.Second)
	losses := registry.PollAllocationLosses()
	if len(losses) != 1 || losses[0].AllocationID != allocationID ||
		losses[0].Reason != LossControlLeaseExpired {
		t.Fatalf("lease losses = %+v", losses)
	}
	grant, err := registry.GetGrant(allocationID)
	if err != nil || !grant.Lost || !grant.WriteFenced {
		t.Fatalf("lost grant = (%+v, %v)", grant, err)
	}
	// Ack 7 was issued while responses were supposedly lost. Even a delayed
	// echo cannot revive the already-lost allocation.
	if response, err := registry.Heartbeat(allocated(9, 7)); err != nil || response.Action != contracts.ActionDrain {
		t.Fatalf("late ack response = (%+v, %v)", response, err)
	}
	if losses := registry.PollAllocationLosses(); len(losses) != 0 {
		t.Fatalf("loss edge was emitted more than once: %+v", losses)
	}
	snapshot, _ := registry.GetAgent("agent-1")
	if !snapshot.LeaseExpired || !snapshot.ReconciliationRequired {
		t.Fatalf("expired agent snapshot = %+v", snapshot)
	}
}

func TestReservationAllowsIdleOnlyUntilAllocationIsObservedActive(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	registerReady(t, registry, "agent-1")
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-prepare", StageExecutionID: "stage-prepare",
		Bindings: []BindingRequirement{{
			LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: testTemplate(t),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	allocationID := reservations[0].Grant.AllocationID

	response, err := registry.Heartbeat(heartbeat("agent-1", 3, 2))
	if err != nil || response.Action != contracts.ActionContinue {
		t.Fatalf("idle preparation transition = (%+v, %v)", response, err)
	}
	if losses := registry.PollAllocationLosses(); len(losses) != 0 {
		t.Fatalf("reservation-to-prepare transition was lost: %+v", losses)
	}
	active := contracts.AgentHeartbeat{
		APIVersion: contracts.APIVersion, InstanceID: "agent-1",
		HeartbeatSeq: 4, EchoedAckSeq: 3,
		ObservedState: contracts.AgentAllocated, AllocationID: &allocationID,
	}
	if response, err = registry.Heartbeat(active); err != nil || response.Action != contracts.ActionContinue {
		t.Fatalf("allocation activation = (%+v, %v)", response, err)
	}
	response, err = registry.Heartbeat(heartbeat("agent-1", 5, 4))
	if err != nil || response.Action != contracts.ActionDrain {
		t.Fatalf("post-activation idle mismatch = (%+v, %v)", response, err)
	}
	losses := registry.PollAllocationLosses()
	if len(losses) != 1 || losses[0].Reason != LossRuntimeMismatch {
		t.Fatalf("post-activation losses = %+v", losses)
	}
}

func TestLeaseUsesMonotonicClockNotWallClock(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-1")

	clock.JumpWall(24 * time.Hour)
	if losses := registry.PollAllocationLosses(); len(losses) != 0 {
		t.Fatalf("wall-clock jump expired lease: %+v", losses)
	}
	clock.JumpWall(-48 * time.Hour)
	clock.Advance(time.Minute)
	registry.PollAllocationLosses()
	snapshot, _ := registry.GetAgent("agent-1")
	if !snapshot.LeaseExpired {
		t.Fatal("monotonic deadline did not expire after one minute")
	}
}

func TestReconcileRuntimeRestartWithholdsNewInstanceUntilOldAllocationReleased(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	oldRegistration := testRegistration("agent-old")
	registerReadyWith(t, registry, oldRegistration)
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-old", StageExecutionID: "stage-old",
		Bindings: []BindingRequirement{{
			LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: testTemplate(t),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}

	restarted := testRegistration("agent-new")
	restarted.ControlURL = oldRegistration.ControlURL
	restarted.A2AURL = oldRegistration.A2AURL
	registerReadyWith(t, registry, restarted)
	losses := registry.PollAllocationLosses()
	if len(losses) != 1 || losses[0].Reason != LossRuntimeRestarted {
		t.Fatalf("restart losses = %+v", losses)
	}
	_, err = registry.ReserveAll(ReservationRequest{
		RunID: "run-new", StageExecutionID: "stage-new",
		Bindings: []BindingRequirement{{
			LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: testTemplate(t),
		}},
	})
	if !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("new instance was offered before reconciliation: %v", err)
	}
	if err := registry.Release(reservations[0].Grant.AllocationID); err != nil {
		t.Fatal(err)
	}
	available, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-new", StageExecutionID: "stage-new-after-release",
		Bindings: []BindingRequirement{{
			LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: testTemplate(t),
		}},
	})
	if err != nil || available[0].Grant.RuntimeInstanceID != "agent-new" {
		t.Fatalf("new instance after reconciliation = (%+v, %v)", available, err)
	}
}

func TestReconcileLostReleaseResponseRepeatsReleaseWithoutSlotReuse(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReady(t, registry, "agent-1")
	reservations, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-release", StageExecutionID: "stage-release",
		Bindings: []BindingRequirement{{
			LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: testTemplate(t),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	allocationID := reservations[0].Grant.AllocationID
	if err := registry.SetWriteFence(allocationID); err != nil {
		t.Fatal(err)
	}
	if err := registry.Release(allocationID); err != nil {
		t.Fatal(err)
	}
	fenced := func(sequence, echoed uint64) contracts.AgentHeartbeat {
		return contracts.AgentHeartbeat{
			APIVersion: contracts.APIVersion, InstanceID: "agent-1",
			HeartbeatSeq: sequence, EchoedAckSeq: echoed,
			ObservedState: contracts.AgentFenced, AllocationID: &allocationID,
		}
	}
	for sequence := uint64(3); sequence <= 4; sequence++ {
		response, err := registry.Heartbeat(fenced(sequence, sequence-1))
		if err != nil || response.Action != contracts.ActionRelease ||
			response.AllocationID == nil || *response.AllocationID != allocationID {
			t.Fatalf("release reconciliation %d = (%+v, %v)", sequence, response, err)
		}
	}
	if _, err := registry.ReserveAll(ReservationRequest{
		RunID: "run-too-early", StageExecutionID: "stage-too-early",
		Bindings: []BindingRequirement{{
			LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: testTemplate(t),
		}},
	}); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("fenced slot was reused: %v", err)
	}
	response, err := registry.Heartbeat(heartbeat("agent-1", 5, 4))
	if err != nil || response.Action != contracts.ActionContinue {
		t.Fatalf("confirmed idle reconciliation = (%+v, %v)", response, err)
	}
}

func testTemplate(t *testing.T) contracts.ResolvedAgentTemplate {
	t.Helper()
	snapshot, err := config.Load("../config/testdata/valid", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	template, err := snapshot.AgentTemplate("artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	return template
}

func testRegistration(instanceID string) contracts.AgentRegistration {
	return contracts.AgentRegistration{
		APIVersion: contracts.APIVersion, InstanceID: instanceID,
		StartedAt:         time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC),
		ControlURL:        "https://" + instanceID + ".example:9443",
		A2AURL:            "https://" + instanceID + ".example:9444",
		SupportedRuntimes: []string{"adk@1"},
		SupportedToolsets: []contracts.ToolsetCapability{{
			Ref: "run-artifacts@1", Tools: []string{"list_artifacts", "read_artifact", "write_artifact"},
		}},
		SupportedSandboxProfiles: []string{"local-workdir@1"}, ObservedState: contracts.AgentIdle,
	}
}

func heartbeat(instanceID string, sequence, echoed uint64) contracts.AgentHeartbeat {
	return contracts.AgentHeartbeat{
		APIVersion: contracts.APIVersion, InstanceID: instanceID,
		HeartbeatSeq: sequence, EchoedAckSeq: echoed, ObservedState: contracts.AgentIdle,
	}
}

func registerReady(t *testing.T, registry *InMemoryRegistry, instanceID string) {
	t.Helper()
	registerReadyWith(t, registry, testRegistration(instanceID))
}

func registerReadyWith(t *testing.T, registry *InMemoryRegistry, registration contracts.AgentRegistration) {
	t.Helper()
	if _, err := registry.Register(registration); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Heartbeat(heartbeat(registration.InstanceID, 1, 0)); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Heartbeat(heartbeat(registration.InstanceID, 2, 1)); err != nil {
		t.Fatal(err)
	}
}

func newTestRegistry(t *testing.T, clock *testClock) *InMemoryRegistry {
	t.Helper()
	var counter atomic.Uint64
	registry, err := NewRegistry(RegistryOptions{
		Now: clock.Now, MonotonicNow: clock.MonotonicNow,
		NewID: func(prefix string) (string, error) {
			return fmt.Sprintf("%s%d", prefix, counter.Add(1)), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return registry
}

type testClock struct {
	mu        sync.Mutex
	now       time.Time
	monotonic time.Duration
}

func newTestClock() *testClock {
	return &testClock{now: time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC)}
}

func (c *testClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *testClock) Advance(duration time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(duration)
	c.monotonic += duration
}

func (c *testClock) JumpWall(duration time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(duration)
}

func (c *testClock) MonotonicNow() time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.monotonic
}
