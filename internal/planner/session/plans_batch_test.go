package session

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type batchPlanTestStore struct {
	Store
	sessions map[string]runstore.PlannerSession
	calls    int
}

func (s *batchPlanTestStore) GetPlannerSessions(context.Context, []string) (map[string]runstore.PlannerSession, error) {
	s.calls++
	return s.sessions, nil
}
func (*batchPlanTestStore) GetPlannerSession(context.Context, string) (runstore.PlannerSession, error) {
	panic("per-session read in batch path")
}

func TestLoadPlansChecksEverySessionIdentityAndOptionalPlan(t *testing.T) {
	identity := planner.SessionIdentity{SessionID: "session", StageExecutionID: "stage", InvocationID: "invocation"}
	valid := runstore.PlannerSession{SessionID: "session", StageExecutionID: "stage", InvocationID: "invocation", StateSchemaVersion: contracts.APIVersion, State: json.RawMessage(`{"status":"running","nextSequence":2,"requestRecorded":false}`)}
	for _, test := range []struct {
		name    string
		change  func(*runstore.PlannerSession)
		missing bool
	}{
		{name: "absent-plan"},
		{name: "missing-session", missing: true},
		{name: "wrong-stage", change: func(s *runstore.PlannerSession) { s.StageExecutionID = "other" }},
		{name: "wrong-invocation", change: func(s *runstore.PlannerSession) { s.InvocationID = "other" }},
		{name: "wrong-schema", change: func(s *runstore.PlannerSession) { s.StateSchemaVersion = "unknown" }},
		{name: "invalid-state", change: func(s *runstore.PlannerSession) { s.State = json.RawMessage(`{"secret":"must not be accepted"}`) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			row := valid
			if test.change != nil {
				test.change(&row)
			}
			store := &batchPlanTestStore{sessions: map[string]runstore.PlannerSession{identity.SessionID: row}}
			if test.missing {
				delete(store.sessions, identity.SessionID)
			}
			service, _ := New(store, Options{})
			plans, err := service.LoadPlans(t.Context(), []planner.SessionIdentity{identity})
			if test.name == "absent-plan" {
				if err != nil || len(plans) != 0 {
					t.Fatalf("optional plan: %v %v", plans, err)
				}
			} else if err == nil {
				t.Fatal("invalid/missing identity accepted")
			}
			if test.missing && !errors.Is(err, runstore.ErrNotFound) {
				t.Fatalf("missing identity cause: %v", err)
			}
			if store.calls != 1 {
				t.Fatalf("batch calls=%d", store.calls)
			}
		})
	}
	store := &batchPlanTestStore{}
	service, _ := New(store, Options{})
	if plans, err := service.LoadPlans(t.Context(), nil); err != nil || len(plans) != 0 || store.calls != 0 {
		t.Fatalf("empty batch: %v %v calls=%d", plans, err, store.calls)
	}
	if _, err := service.LoadPlans(t.Context(), []planner.SessionIdentity{{SessionID: "session"}}); err == nil || store.calls != 0 {
		t.Fatal("incomplete identity queried store")
	}
}
