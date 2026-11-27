package session

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresTypedPlanEventsAreOrderedAndCompareAppendIsAtomic(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPlannerPool(t, ctx)
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	createPlannerStage(t, ctx, store)
	service, err := New(store, Options{})
	if err != nil {
		t.Fatal(err)
	}
	started, err := service.Begin(ctx, "stage-planner")
	if err != nil {
		t.Fatal(err)
	}
	if err := service.RecordRequest(ctx, started.Identity, planner.RequestFacts{
		Bindings:           []string{"builder", "reviewer"},
		ObjectiveDigest:    "sha256:" + strings.Repeat("a", 64),
		InstructionsDigest: "sha256:" + strings.Repeat("b", 64),
		ParameterNames:     []string{}, Artifacts: map[string]contracts.ArtifactRef{},
	}); err != nil {
		t.Fatal(err)
	}

	controller, err := planner.NewPlannerPlanController("global-objective-never-copy", 0)
	if err != nil {
		t.Fatal(err)
	}
	before := controller.Snapshot()
	first, planErr := controller.AddSubtask("Inspect", "Read the source")
	if planErr != nil {
		t.Fatal(planErr)
	}
	recordPlan(t, service, started.Identity, before, first, planner.PlannerEventPlanChanged)
	recordFact(t, service, started.Identity, planner.PlannerFact{
		Kind: planner.PlannerEventCurrentChanged, Key: "current:1",
		PlanRevision: first.Revision, SubtaskID: "0",
	})
	before = controller.Snapshot()
	second, planErr := controller.AddSubtask("Review", "Check completeness")
	if planErr != nil {
		t.Fatal(planErr)
	}
	recordPlan(t, service, started.Identity, before, second, planner.PlannerEventPlanChanged)
	before = controller.Snapshot()
	claim, planErr := controller.ClaimCurrentSubtask("0", "reviewer")
	if planErr != nil {
		t.Fatal(planErr)
	}
	recordFact(t, service, started.Identity, planner.PlannerFact{
		Kind: planner.PlannerEventDispatchSelected, Key: "selected:" + claim.CallID,
		PlanRevision: before.Revision, SubtaskID: "0", CallID: claim.CallID, WorkerName: "reviewer",
	})
	dispatched := controller.Snapshot()
	recordPlan(t, service, started.Identity, before, dispatched, planner.PlannerEventDispatchStarted)
	before = controller.Snapshot()
	completed, planErr := controller.CompleteDispatch(claim.CallID, stageresult.StageSucceeded)
	if planErr != nil {
		t.Fatal(planErr)
	}
	recordPlan(t, service, started.Identity, before, completed, planner.PlannerEventDispatchCompleted)
	recordFact(t, service, started.Identity, planner.PlannerFact{
		Kind: planner.PlannerEventCurrentChanged, Key: "current:4",
		PlanRevision: completed.Revision, SubtaskID: "1",
	})

	base, ok, err := loadPlan(ctx, service, started.Identity)
	if err != nil || !ok || base.Revision != 4 || base.CurrentSubtaskID != "1" {
		t.Fatalf("base plan = (%+v, %t, %v)", base, ok, err)
	}
	candidates := make([]planner.PlannerPlanProjection, 2)
	for index, objective := range []string{"Race left", "Race right"} {
		candidates[index] = base
		candidates[index].Subtasks = append([]planner.PlannerSubtask(nil), base.Subtasks...)
		candidates[index].Revision++
		candidates[index].Subtasks = append(candidates[index].Subtasks, planner.PlannerSubtask{
			ID: "2", Objective: objective, Instructions: "Only one append may commit",
			Status: planner.PlannerSubtaskPending,
		})
	}
	startRace := make(chan struct{})
	results := make(chan error, len(candidates))
	var group sync.WaitGroup
	for _, candidate := range candidates {
		candidate := candidate
		group.Add(1)
		go func() {
			defer group.Done()
			<-startRace
			results <- service.RecordPlan(ctx, started.Identity, planner.PlannerPlanTransition{
				Kind: planner.PlannerEventPlanChanged, ExpectedRevision: base.Revision, Plan: candidate,
			})
		}()
	}
	close(startRace)
	group.Wait()
	close(results)
	var succeeded, conflicted int
	for result := range results {
		switch {
		case result == nil:
			succeeded++
		case errors.Is(result, runstore.ErrConflict):
			conflicted++
		default:
			t.Fatalf("unexpected compare-append result: %v", result)
		}
	}
	if succeeded != 1 || conflicted != 1 {
		t.Fatalf("compare append succeeded=%d conflicted=%d", succeeded, conflicted)
	}
	latest, ok, err := loadPlan(ctx, service, started.Identity)
	if err != nil || !ok || latest.Revision != 5 || len(latest.Subtasks) != 3 {
		t.Fatalf("latest plan = (%+v, %t, %v)", latest, ok, err)
	}
	if err := service.RecordPlan(ctx, started.Identity, planner.PlannerPlanTransition{
		Kind: planner.PlannerEventPlanChanged, ExpectedRevision: base.Revision, Plan: latest,
	}); err != nil {
		t.Fatalf("exact committed retry: %v", err)
	}

	// Run events 1 to 5 record the Run's initializing, pending and running
	// states and its Stage's preparing and running states.
	const lifecycleEvents = 5
	runEventPage, err := store.ListRunEvents(ctx, "run-planner", 0, 100)
	if err != nil || len(runEventPage.Events) != lifecycleEvents+10 || runEventPage.More {
		t.Fatalf("Run events = (%d, more=%t, %v)", len(runEventPage.Events), runEventPage.More, err)
	}
	runEvents := runEventPage.Events
	wantKinds := []runstore.RunEventKind{
		runstore.RunEventPlannerStarted,
		runstore.RunEventPlannerRequestRecorded,
		runstore.RunEventPlannerPlanChanged,
		runstore.RunEventPlannerCurrentChanged,
		runstore.RunEventPlannerPlanChanged,
		runstore.RunEventPlannerDispatchSelected,
		runstore.RunEventPlannerDispatchStarted,
		runstore.RunEventPlannerDispatchCompleted,
		runstore.RunEventPlannerCurrentChanged,
		runstore.RunEventPlannerPlanChanged,
	}
	for index := range wantKinds {
		event := runEvents[lifecycleEvents+index]
		if event.SequenceNumber != int64(lifecycleEvents+index+1) || event.Kind != wantKinds[index] {
			t.Fatalf("Planner Run event %d = %+v", index, event)
		}
	}
	cursor, err := store.GetRunEventCursor(ctx, "run-planner")
	if err != nil || cursor.Sequence != lifecycleEvents+10 || cursor.Generation == "" {
		t.Fatalf("Run cursor = (%+v, %v)", cursor, err)
	}
	// The page stops at its event count; the committed last event follows.
	page, err := store.ListRunEvents(ctx, "run-planner", lifecycleEvents+7, 2)
	if err != nil || len(page.Events) != 2 || page.Events[0].SequenceNumber != lifecycleEvents+8 ||
		page.Events[1].SequenceNumber != lifecycleEvents+9 || !page.More {
		t.Fatalf("resumed Run event page = (%+v, %v)", page, err)
	}
	session, err := store.GetPlannerSession(ctx, started.Identity.SessionID)
	if err != nil || session.NextEventSequence != 11 ||
		strings.Contains(string(session.State), "global-objective-never-copy") {
		t.Fatalf("Planner session = (%+v, %v)", session, err)
	}
}

func createPlannerStage(
	t *testing.T, ctx context.Context, store *runstore.PostgresStore,
) {
	t.Helper()
	createPlannerStageWithLimit(t, ctx, store, 0)
}

func createPlannerStageWithLimit(
	t *testing.T, ctx context.Context, store *runstore.PostgresStore, limit contracts.PlannerSubtaskLimit,
) {
	t.Helper()
	stage := map[string]any{"objective": "build", "session": "isolated"}
	if limit != 0 {
		stage["maxSubtasks"] = limit
	}
	stageSnapshot, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	_, err = runstoretest.CreateRun(ctx, store, runstore.CreateRunParams{
		LaneClass: resolved.LaneClassModel,
		RunID:     "run-planner", OwnerID: "user", WorkflowName: "workflow", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot:      json.RawMessage(`{"name":"workflow"}`),
		Parameters:            map[string]string{"mode": "strict"},
		RuntimeConfig:         runtimeconfigtest.BuiltInRunSnapshot(),
	})

	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(
		ctx, "run-planner", runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"},
	); err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(
		ctx, "run-planner", runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"},
	); err != nil {
		t.Fatal(err)
	}
	_, err = store.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		LaneClass:        resolved.LaneClassModel,
		StageExecutionID: "stage-planner", RunID: "run-planner", StageName: "build", Attempt: 1,
		StageSpecSchemaVersion:    contracts.APIVersion,
		StageSpecSnapshot:         stageSnapshot,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext:              runstore.StageContextSnapshot{},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func isolatedPlannerPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{})
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}
