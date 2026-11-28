package controlplane

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/control"
)

// V498-003 connects the two release edges to real Registry and batch owners.
// The RPC peer injects either a failed cleanup or its lost success response.
func TestAllocationReleasePilotLostReplyAndReplacement(t *testing.T) {
	for _, failure := range []string{"cleanup-failed", "reply-lost"} {
		t.Run(failure, func(t *testing.T) {
			registry := newTestRegistry(t, newTestClock())
			registration := testRegistration("release-pilot")
			registration.AllocationSlots = 2
			registerReadyWith(t, registry, registration)
			placed, err := registry.ReserveAll(slotRequest(t, "stage-pilot", 2))
			if err != nil {
				t.Fatal(err)
			}
			failed, peer := placed[0].Grant.AllocationID, placed[1].Grant.AllocationID
			observations := []control.ObservedAllocation{}
			for _, reservation := range placed {
				id := reservation.Grant.AllocationID
				if err := registry.SetWriteFence(id); err != nil {
					t.Fatal(err)
				}
				observations = append(observations, control.ObservedAllocation{AllocationID: id, ObservedState: control.AgentFenced})
			}
			beat := heartbeat(registration.InstanceID, 3, 2)
			beat.ObservedState, beat.Allocations = control.AgentAllocated, observations
			if _, err := registry.Heartbeat(beat); err != nil {
				t.Fatal(err)
			}
			runtime := &releasePilotRuntime{recordingRuntime: &recordingRuntime{}, failed: failed, failure: failure, cleaned: map[string]bool{}}
			batch, err := NewRuntimeBatchController(runtime, registry, RuntimeBatchOptions{PrepareTimeout: time.Minute, CleanupTimeout: time.Minute})
			if err != nil {
				t.Fatal(err)
			}
			if err := batch.ReleaseAll(t.Context(), placed); err == nil {
				t.Fatal("injected release failure disappeared")
			}
			grant, err := registry.GetGrant(failed)
			if err != nil || !grant.WriteFenced {
				t.Fatalf("uncertain release lost its grant/fence: %+v %v", grant, err)
			}
			if runtime.cleaned[failed] != (failure == "reply-lost") {
				t.Fatal("failure injection did not reach the intended edge")
			}
			if _, err := registry.GetGrant(peer); !errors.Is(err, ErrAllocationNotFound) {
				t.Fatalf("healthy release grant: %v", err)
			}
			if got := registry.SnapshotOperations().RuntimeAgents[0].AvailableSlots; got != 0 {
				t.Fatalf("slot reused before heartbeat: %d", got)
			}
			beat.HeartbeatSeq, beat.EchoedAckSeq = 4, 3
			releaseResponse, err := registry.Heartbeat(beat)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, action := range releaseResponse.AllocationActions {
				if action.AllocationID == peer && action.Action == control.ActionRelease {
					found = true
				}
			}
			if !found {
				t.Fatal("missing addressed release action")
			}
			staleBeat := beat
			beat.HeartbeatSeq, beat.EchoedAckSeq = 5, 4
			beat.Allocations = []control.ObservedAllocation{{AllocationID: failed, ObservedState: control.AgentFenced}}
			if _, err := registry.Heartbeat(beat); err != nil {
				t.Fatal(err)
			}
			replacement, err := registry.ReserveAll(slotRequest(t, "stage-replacement", 1))
			if err != nil {
				t.Fatalf("failed cleanup blocked healthy neighboring capacity: %v", err)
			}
			newID := replacement[0].Grant.AllocationID
			if newID == failed || newID == peer {
				t.Fatal("replacement reused an Allocation ID")
			}
			before, err := registry.GetGrant(newID)
			if err != nil {
				t.Fatal(err)
			}
			// Cached old heartbeat replies and delayed RPC successes address the old ID.
			replayed, err := registry.Heartbeat(staleBeat)
			if err != nil || !reflect.DeepEqual(replayed, releaseResponse) {
				t.Fatalf("old heartbeat replay: %+v %v", replayed, err)
			}
			if err := batch.ReleaseAll(t.Context(), placed[1:]); !errors.Is(err, ErrAllocationNotFound) {
				t.Fatalf("delayed old release response: %v", err)
			}
			after, err := registry.GetGrant(newID)
			if err != nil || !reflect.DeepEqual(before, after) {
				t.Fatalf("old release changed replacement: %+v %v", after, err)
			}
			if err := batch.ReleaseAll(t.Context(), placed[:1]); err != nil {
				t.Fatalf("idempotent retry: %v", err)
			}
			if _, err := registry.GetGrant(failed); !errors.Is(err, ErrAllocationNotFound) {
				t.Fatalf("retry retained old grant: %v", err)
			}
			if !runtime.cleaned[failed] {
				t.Fatal("retry skipped cleanup")
			}
		})
	}
}

type releasePilotRuntime struct {
	*recordingRuntime
	mu              sync.Mutex
	failed, failure string
	injected        bool
	cleaned         map[string]bool
}

func (r *releasePilotRuntime) Release(_ context.Context, reservation Reservation) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	id := reservation.Grant.AllocationID
	if id == r.failed && !r.injected {
		r.injected = true
		if r.failure == "reply-lost" {
			r.cleaned[id] = true
		}
		return errors.New("injected " + r.failure)
	}
	r.cleaned[id] = true
	return nil
}
