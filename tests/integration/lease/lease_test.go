package lease_test

import (
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
)

const confirmedLease = 60 * time.Second

func TestAsymmetricPartitionsFenceBothAuthorities(t *testing.T) {
	t.Run("requests arrive but responses are lost", func(t *testing.T) {
		registry, runtime, clock, allocationID := readyAllocation(t, "request-only")

		// The first lost response's request confirms the preceding response.
		sendHeartbeat(t, registry, runtime, runtime.echoedAck, false)
		for range 5 {
			clock.Advance(10 * time.Second)
			sendHeartbeat(t, registry, runtime, runtime.echoedAck, false)
		}
		clock.Advance(10 * time.Second)
		runtime.expire(clock.MonotonicNow())
		losses := registry.PollAllocationLosses()

		assertSingleLoss(t, losses, allocationID)
		if runtime.workerLive || runtime.state != contracts.AgentFenced {
			t.Fatalf("Runtime did not self-fence: %+v", runtime)
		}
	})

	t.Run("responses arrive but new echoes do not", func(t *testing.T) {
		registry, runtime, clock, allocationID := readyAllocation(t, "response-only")
		staleEcho := runtime.echoedAck

		// New responses keep the Runtime's local lease alive, while a fault
		// proxy forces every request to carry the old echo at the Control Plane.
		sendHeartbeat(t, registry, runtime, staleEcho, true)
		for range 5 {
			clock.Advance(10 * time.Second)
			sendHeartbeat(t, registry, runtime, staleEcho, true)
		}
		clock.Advance(10 * time.Second)
		sendHeartbeat(t, registry, runtime, staleEcho, true)
		losses := registry.PollAllocationLosses()

		assertSingleLoss(t, losses, allocationID)
		if runtime.workerLive || runtime.state != contracts.AgentFenced {
			t.Fatalf("drain action did not fence Runtime: %+v", runtime)
		}
		// A delayed once-valid echo cannot renew an expired Control Plane lease.
		sendHeartbeat(t, registry, runtime, runtime.echoedAck, true)
		if repeated := registry.PollAllocationLosses(); len(repeated) != 0 {
			t.Fatalf("loss edge repeated after delayed echo: %+v", repeated)
		}
	})

	t.Run("full partition", func(t *testing.T) {
		registry, runtime, clock, allocationID := readyAllocation(t, "full")
		clock.Advance(confirmedLease)
		runtime.expire(clock.MonotonicNow())
		assertSingleLoss(t, registry.PollAllocationLosses(), allocationID)
		if runtime.workerLive || runtime.state != contracts.AgentFenced {
			t.Fatalf("Runtime did not self-fence: %+v", runtime)
		}
	})
}

func TestRuntimeRestartIsWithheldUntilOldAuthorityIsReleased(t *testing.T) {
	registry, oldRuntime, clock, allocationID := readyAllocation(t, "restart")
	restarted := newRuntime(
		"runtime-restarted", oldRuntime.controlURL, oldRuntime.a2aURL, clock,
	)
	registerRuntime(t, registry, restarted, clock)

	assertSingleLoss(t, registry.PollAllocationLosses(), allocationID)
	if _, err := registry.ReserveAll(reservationRequest(t, "run-new", "stage-new")); !errors.Is(err, controlplane.ErrInsufficientCapacity) {
		t.Fatalf("restarted slot was offered before old release: %v", err)
	}
	if err := registry.Release(allocationID); err != nil {
		t.Fatal(err)
	}
	// Complete the new identity's registration round trip after unblocking.
	sendHeartbeat(t, registry, restarted, 0, true)
	sendHeartbeat(t, registry, restarted, restarted.echoedAck, true)
	reservations, err := registry.ReserveAll(reservationRequest(t, "run-new", "stage-new-after-release"))
	if err != nil || reservations[0].Grant.RuntimeInstanceID != restarted.instanceID {
		t.Fatalf("restarted slot was not reusable after release: (%+v, %v)", reservations, err)
	}
}

type faultClock struct {
	mu        sync.Mutex
	wall      time.Time
	monotonic time.Duration
}

func (c *faultClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.wall
}

func (c *faultClock) MonotonicNow() time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.monotonic
}

func (c *faultClock) Advance(delta time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.wall = c.wall.Add(delta)
	c.monotonic += delta
}

type runtimeHarness struct {
	instanceID string
	controlURL string
	a2aURL     string
	sequence   uint64
	echoedAck  uint64
	deadline   time.Duration
	expired    bool
	state      contracts.AgentObservedState
	allocation *string
	workerLive bool
	clock      *faultClock
}

func newRuntime(
	instanceID, controlURL, a2aURL string, clock *faultClock,
) *runtimeHarness {
	return &runtimeHarness{
		instanceID: instanceID,
		controlURL: controlURL,
		a2aURL:     a2aURL,
		state:      contracts.AgentIdle,
		clock:      clock,
	}
}

func (r *runtimeHarness) heartbeat(echoed uint64) contracts.AgentHeartbeat {
	r.sequence++
	return contracts.AgentHeartbeat{
		APIVersion: contracts.APIVersion, InstanceID: r.instanceID,
		HeartbeatSeq: r.sequence, EchoedAckSeq: echoed,
		ObservedState: r.state, AllocationID: cloneString(r.allocation),
	}
}

func (r *runtimeHarness) receive(response contracts.HeartbeatResponse, now time.Duration) {
	if !r.expired && now < r.deadline && response.AckSeq > r.echoedAck {
		r.echoedAck = response.AckSeq
		r.deadline = now + confirmedLease
	}
	switch response.Action {
	case contracts.ActionDrain:
		if r.allocation != nil && response.AllocationID != nil && *r.allocation == *response.AllocationID {
			r.workerLive = false
			r.state = contracts.AgentFenced
		}
	case contracts.ActionRelease:
		if sameOptionalString(r.allocation, response.AllocationID) {
			r.workerLive = false
			r.state = contracts.AgentIdle
			r.allocation = nil
		}
	}
}

func (r *runtimeHarness) expire(now time.Duration) {
	if r.expired || now < r.deadline {
		return
	}
	r.expired = true
	r.workerLive = false
	r.state = contracts.AgentFenced
}

func readyAllocation(
	t *testing.T, suffix string,
) (*controlplane.InMemoryRegistry, *runtimeHarness, *faultClock, string) {
	t.Helper()
	clock := &faultClock{wall: time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC)}
	nextID := 0
	registry, err := controlplane.NewRegistry(controlplane.RegistryOptions{
		Now: clock.Now, MonotonicNow: clock.MonotonicNow,
		NewID: func(string) (string, error) {
			nextID++
			return fmt.Sprintf("allocation-%s-%d", suffix, nextID), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	runtime := newRuntime(
		"runtime-"+suffix,
		"https://runtime-"+suffix+".example:9443",
		"https://runtime-"+suffix+".example:9444",
		clock,
	)
	registerRuntime(t, registry, runtime, clock)
	sendHeartbeat(t, registry, runtime, 0, true)
	sendHeartbeat(t, registry, runtime, runtime.echoedAck, true)
	reservations, err := registry.ReserveAll(reservationRequest(t, "run-"+suffix, "stage-"+suffix))
	if err != nil {
		t.Fatal(err)
	}
	allocationID := reservations[0].Grant.AllocationID
	runtime.allocation = &allocationID
	runtime.state = contracts.AgentAllocated
	runtime.workerLive = true
	sendHeartbeat(t, registry, runtime, runtime.echoedAck, true)
	return registry, runtime, clock, allocationID
}

func registerRuntime(
	t *testing.T,
	registry *controlplane.InMemoryRegistry,
	runtime *runtimeHarness,
	clock *faultClock,
) {
	t.Helper()
	_, err := registry.Register(contracts.AgentRegistration{
		InitialLabels:            []string{},
		SupportedRuntimeAdapters: []contracts.RuntimeAdapterRef{},
		APIVersion:               contracts.APIVersion, InstanceID: runtime.instanceID, SoftwareVersion: "0.1.0",
		StartedAt: clock.Now(), ControlURL: runtime.controlURL, A2AURL: runtime.a2aURL,
		SupportedRuntimes: []string{"adk@1"},
		SupportedToolsets: []contracts.ToolsetCapability{{
			Ref: "run-artifacts@1", Tools: []string{"list_artifacts", "read_artifact", "write_artifact"},
		}},
		SupportedSandboxProfiles: []string{"local-workdir@1"},
		ObservedState:            runtime.state,
		AllocationID:             cloneString(runtime.allocation),
	})
	if err != nil {
		t.Fatal(err)
	}
	runtime.deadline = clock.MonotonicNow() + confirmedLease
}

func sendHeartbeat(
	t *testing.T,
	registry *controlplane.InMemoryRegistry,
	runtime *runtimeHarness,
	echoed uint64,
	deliverResponse bool,
) {
	t.Helper()
	response, err := registry.Heartbeat(runtime.heartbeat(echoed))
	if err != nil {
		t.Fatal(err)
	}
	if deliverResponse {
		runtime.receive(response, runtime.clock.MonotonicNow())
	}
}

func reservationRequest(t *testing.T, runID, stageID string) controlplane.ReservationRequest {
	t.Helper()
	snapshot, err := config.Load(
		filepath.Join("..", "..", "..", "internal", "config", "testdata", "valid"),
		config.MVPDescriptors(),
	)
	if err != nil {
		t.Fatal(err)
	}
	template, err := snapshot.AgentTemplate("artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	gateway, err := snapshot.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	return controlplane.ReservationRequest{
		RunID: runID, StageExecutionID: stageID,
		Bindings: []controlplane.BindingRequirement{{
			LogicalAgentName: "builder", Namespace: "builder", AgentTemplate: template,
			WorkerSessionMode: contracts.WorkerSessionIsolated,
			ExecutionConfig: controlplane.AllocationExecutionConfig{
				ModelPolicy: template.ModelPolicy.Ref, LLMGateway: gateway.Ref,
			},
		}},
	}
}

func assertSingleLoss(t *testing.T, losses []controlplane.AllocationLoss, allocationID string) {
	t.Helper()
	if len(losses) != 1 || losses[0].AllocationID != allocationID ||
		(losses[0].Reason != controlplane.LossControlLeaseExpired &&
			losses[0].Reason != controlplane.LossRuntimeRestarted) {
		t.Fatalf("allocation losses = %+v", losses)
	}
}

func cloneString(value *string) *string {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}

func sameOptionalString(left, right *string) bool {
	return left == nil && right == nil || left != nil && right != nil && *left == *right
}
