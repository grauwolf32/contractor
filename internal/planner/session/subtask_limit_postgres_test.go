package session

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresWorkflowSubtaskLimitSurvivesPublicationAndRecovery(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	pool := isolatedPlannerPool(t, ctx)
	store := runstore.NewPostgresStore(pool)
	const limit = contracts.PlannerSubtaskLimit(101)
	createPlannerStageWithLimit(t, ctx, store, limit)
	service, err := New(store, Options{})
	if err != nil {
		t.Fatal(err)
	}
	started, err := service.Begin(ctx, "stage-planner")
	if err != nil {
		t.Fatal(err)
	}
	if err := service.RecordRequest(ctx, started.Identity, planner.RequestFacts{
		Bindings: []string{"builder"}, ObjectiveDigest: "sha256:" + strings.Repeat("a", 64),
		InstructionsDigest: "sha256:" + strings.Repeat("b", 64), ParameterNames: []string{}, Artifacts: map[string]contracts.ArtifactRef{},
	}); err != nil {
		t.Fatal(err)
	}
	// An intentionally more permissive producer proves the durable boundary
	// rejects overflow independently of the in-memory controller.
	controller, err := planner.NewPlannerPlanController("Inspect input", limit+1)
	if err != nil {
		t.Fatal(err)
	}
	var last planner.PlannerPlanTransition
	for range limit.Effective() {
		before := controller.Snapshot()
		after, err := controller.AddSubtask("Inspect", "Read source")
		if err != nil {
			t.Fatal(err)
		}
		last = planner.PlannerPlanTransition{Kind: planner.PlannerEventPlanChanged, ExpectedRevision: before.Revision, Plan: after.Projection()}
		if err := service.RecordPlan(ctx, started.Identity, last); err != nil {
			t.Fatal(err)
		}
	}
	before, err := store.GetPlannerSession(ctx, started.Identity.SessionID)
	if err != nil {
		t.Fatal(err)
	}
	cursor, err := store.GetRunEventCursor(ctx, "run-planner")
	if err != nil {
		t.Fatal(err)
	}
	if err := service.RecordPlan(ctx, started.Identity, last); err != nil {
		t.Fatalf("replay: %v", err)
	}
	previous := controller.Snapshot()
	overflow, planErr := controller.AddSubtask("Overflow", "Read source")
	if planErr != nil {
		t.Fatal(planErr)
	}
	if err := service.RecordPlan(ctx, started.Identity, planner.PlannerPlanTransition{
		Kind: planner.PlannerEventPlanChanged, ExpectedRevision: previous.Revision, Plan: overflow.Projection(),
	}); err == nil {
		t.Fatal("plan above the pinned limit was published")
	}
	after, err := store.GetPlannerSession(ctx, started.Identity.SessionID)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("rejected append or replay changed session: %v", err)
	}
	current, err := store.GetRunEventCursor(ctx, "run-planner")
	if err != nil || !reflect.DeepEqual(cursor, current) {
		t.Fatalf("rejected append or replay changed Run events: %v", err)
	}
	restarted, err := New(store, Options{})
	if err != nil {
		t.Fatal(err)
	}
	plans, err := restarted.LoadPlans(ctx, []planner.SessionIdentity{started.Identity})
	if err != nil || len(plans["stage-planner"].Subtasks) != limit.Effective() {
		t.Fatalf("reloaded plans: %v", err)
	}
	if _, err := restarted.Begin(ctx, "stage-planner"); !errors.Is(err, planner.ErrInvocationInProgress) {
		t.Fatalf("running recovery: %v", err)
	}
	if err := restarted.Complete(ctx, started.Identity, planner.Completion{Failure: &planner.Failure{Code: "plan_budget", Message: "Plan budget reached"}}); err != nil {
		t.Fatal(err)
	}
	recovered, err := restarted.Begin(ctx, "stage-planner")
	if err != nil || recovered.Invoke || recovered.Completion == nil {
		t.Fatalf("completion recovery: %+v %v", recovered, err)
	}
	saved, err := store.GetPlannerSession(ctx, started.Identity.SessionID)
	if err != nil {
		t.Fatal(err)
	}
	state, err := decodeState(saved.State)
	if err != nil || state.MaxSubtasks != limit || len(state.Plan.Subtasks) != limit.Effective() {
		t.Fatalf("pinned state: %v", err)
	}
	// Simulate an invalid durable row, bypassing the service's write validation.
	state.MaxSubtasks = 1
	invalid, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE planner_sessions SET state = $2 WHERE session_id = $1`, started.Identity.SessionID, invalid); err != nil {
		t.Fatal(err)
	}
	if _, err := restarted.LoadPlans(ctx, []planner.SessionIdentity{started.Identity}); err == nil {
		t.Fatal("batch projection accepted a plan above its pinned limit")
	}
	if _, err := restarted.Begin(ctx, "stage-planner"); err == nil {
		t.Fatal("recovery accepted a plan above its pinned limit")
	}
}
