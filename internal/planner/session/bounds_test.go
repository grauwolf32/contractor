package session

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// A session write over a session bound is refused as runstore.ErrInvalid,
// which the Planner reports as the non-retryable planner_session_rejected
// (PLN-R106): the 2 MiB JSON bound the store also enforces on the stored
// value, and the recorded fact limit. Neither changes the session.
func TestSessionBoundsRefuseWritesAsInvalid(t *testing.T) {
	if _, err := encodeBounded(map[string]string{"a": strings.Repeat("x", maxSessionJSONBytes)}); !errors.Is(err, runstore.ErrInvalid) {
		t.Fatalf("oversized Planner JSON = %v, want runstore.ErrInvalid", err)
	}

	for _, limit := range []contracts.PlannerSubtaskLimit{0, 1, 100, contracts.MaxPlannerSubtasks} {
		t.Run(fmt.Sprint(limit), func(t *testing.T) {
			store, service, identity := runningTestService(t)
			state, err := decodeState(store.session.State)
			if err != nil {
				t.Fatal(err)
			}
			state.MaxSubtasks = limit
			state.FactDigests = make(map[string]string, state.factLimit())
			for index := range state.factLimit() {
				state.FactDigests[fmt.Sprintf("current:%d", index)] = "sha256:" + strings.Repeat("e", 64)
			}
			full, err := encodeState(state)
			if err != nil {
				t.Fatal(err)
			}
			store.session.State = full
			before := store.snapshotEvents()
			err = service.RecordFact(t.Context(), identity, planner.PlannerFact{
				Kind: planner.PlannerEventFinishRequested, Key: "finish:0:failed", Outcome: "failed",
			})
			if !errors.Is(err, runstore.ErrInvalid) || !planner.SessionWriteRejected(err) {
				t.Fatalf("fact past the limit = %v, want runstore.ErrInvalid", err)
			}
			if after := store.snapshotEvents(); len(after) != len(before) || string(store.session.State) != string(full) {
				t.Fatalf("refused fact changed the session: %d events, want %d", len(after), len(before))
			}
			state.FactDigests["overflow"] = "sha256:" + strings.Repeat("e", 64)
			invalid, err := json.Marshal(state)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := decodeState(invalid); err == nil {
				t.Fatal("durable read accepted too many fact receipts")
			}
		})
	}
}

// The session fake refuses a state as the planner_session_state_size CHECK
// does, on its stored jsonb text: compact JSON within 2 MiB whose separators
// jsonb widens is refused like the real store refuses it.
func TestSessionFakeRefusesStatesOverTheStoredBound(t *testing.T) {
	store, _, identity := runningTestService(t)
	expanding := json.RawMessage(`{"a":[` + strings.TrimSuffix(strings.Repeat("0,", 1_000_000), ",") + `]}`)
	if len(expanding) > maxSessionJSONBytes {
		t.Fatalf("compact state of %d bytes exceeds the pre-check", len(expanding))
	}
	err := store.AppendPlannerEvent(t.Context(), runstore.AppendPlannerEventParams{
		SessionID: identity.SessionID, StageExecutionID: identity.StageExecutionID,
		InvocationID: identity.InvocationID, SequenceNumber: store.session.NextEventSequence,
		NewStateSchemaVersion: contracts.APIVersion, NewState: expanding,
	})
	if !errors.Is(err, runstore.ErrInvalid) {
		t.Fatalf("fake append of a widened state = %v, want runstore.ErrInvalid", err)
	}
	err = store.StartPlanner(t.Context(), runstore.StartPlannerParams{
		StageExecutionID: identity.StageExecutionID, SessionID: "session-other",
		InvocationID: "invocation-other", StateSchemaVersion: contracts.APIVersion, InitialState: expanding,
	})
	if !errors.Is(err, runstore.ErrInvalid) {
		t.Fatalf("fake start with a widened state = %v, want runstore.ErrInvalid", err)
	}
}
