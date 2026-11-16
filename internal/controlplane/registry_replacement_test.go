package controlplane

import (
	"testing"
	"time"
)

func TestReservationRemovalUnblocksReplacement(t *testing.T) {
	for _, operation := range []string{"discard_candidate", "release", "release_lost"} {
		t.Run(operation, func(t *testing.T) {
			clock := newTestClock()
			registry := newTestRegistry(t, clock)
			principal := testPrincipal("d")
			old := testRegistration("old-owner")
			registerReadyAs(t, registry, principal, old)
			clock.Advance(59 * time.Second)
			request := ReservationRequest{RunID: "run-replacement", StageExecutionID: "stage-replacement", Bindings: []BindingRequirement{testBinding(t, "builder", "builder", testTemplate(t))}}
			var reservations []Reservation
			var err error
			if operation == "discard_candidate" {
				reservations, err = registry.ReserveCandidateEdges(request, []CandidateEdge{{LogicalAgentName: "builder", RuntimeAgentID: principal.RuntimeAgentID, RuntimeAgentInstanceID: old.InstanceID, RuntimeAgentLabelRevision: principal.LabelRevision}})
			} else {
				reservations, err = registry.ReserveAll(request)
			}
			if err != nil {
				t.Fatal(err)
			}
			clock.Advance(2 * time.Second)
			replacement := testRegistration("new-owner")
			replacement.ControlURL, replacement.A2AURL = old.ControlURL, old.A2AURL
			registerReadyAs(t, registry, principal, replacement)
			if registry.agents[replacement.InstanceID].blockedByInstanceID == nil || len(registry.PlacementCandidates()) != 0 {
				t.Fatal("replacement must wait for the old allocation")
			}
			switch operation {
			case "discard_candidate":
				err = registry.DiscardCandidateReservations(request.StageExecutionID)
			case "release":
				err = registry.Release(reservations[0].Grant.AllocationID)
			case "release_lost":
				var released bool
				released, err = registry.ReleaseLost(reservations[0].Grant.AllocationID)
				if err == nil && !released {
					t.Fatal("lost allocation was not released")
				}
			}
			if err != nil {
				t.Fatal(err)
			}
			registry.PollAllocationLosses()
			if _, err := registry.HeartbeatAuthenticated(principal.RuntimeAgentID, heartbeat(replacement.InstanceID, 3, 2)); err != nil {
				t.Fatal(err)
			}
			candidates := registry.PlacementCandidates()
			if len(candidates) != 1 || candidates[0].Registration.InstanceID != replacement.InstanceID {
				t.Fatalf("replacement is excluded after %s: candidates=%d", operation, len(candidates))
			}
			if len(registry.allocations) != 0 {
				t.Fatal("allocation survived removal")
			}
			if _, retained := registry.agents[old.InstanceID]; retained {
				t.Fatal("inactive owner was not retired")
			}
			stage, retained := registry.stageReservations[request.StageExecutionID]
			if operation == "discard_candidate" && retained || operation != "discard_candidate" && (!retained || !stage.released) {
				t.Fatal("removal changed stage reservation replay semantics")
			}
		})
	}
}
