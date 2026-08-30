package session

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/adk/model"
	adksession "google.golang.org/adk/session"
	"google.golang.org/genai"
)

func TestPostgresSessionPersistsCompletionForRecovery(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPlannerPool(t, ctx)
	store := runstore.NewPostgresStore(pool)
	createPlannerStage(t, ctx, store)
	ids := []string{
		"session-1", "invocation-1", "event-started", "event-request", "event-adk", "event-complete",
	}
	service, err := New(store, Options{NewID: func(string) (string, error) {
		result := ids[0]
		ids = ids[1:]
		return result, nil
	}})
	if err != nil {
		t.Fatal(err)
	}

	started, err := service.Begin(ctx, "stage-planner")
	if err != nil || !started.Invoke {
		t.Fatalf("Begin = (%+v, %v)", started, err)
	}
	if err := service.RecordRequest(ctx, started.Identity, planner.RequestFacts{
		Bindings: []string{"builder"}, ObjectiveDigest: "sha256:" + strings.Repeat("a", 64),
		InstructionsDigest: "sha256:" + strings.Repeat("b", 64),
		ParameterNames:     []string{"mode"}, Artifacts: map[string]contracts.ArtifactRef{},
	}); err != nil {
		t.Fatal(err)
	}
	const providerSecret = "sk-provider-postgres-secret"
	adk, err := service.NewADKSession(ctx, started.Identity, ADKOptions{
		AppName: "contractor_streamline", UserID: "stage-planner", AllowedTools: []string{"finish"},
	})
	if err != nil {
		t.Fatal(err)
	}
	created, err := adk.Create(ctx, &adksession.CreateRequest{
		AppName: "contractor_streamline", UserID: "stage-planner", SessionID: started.Identity.SessionID,
	})
	if err != nil {
		t.Fatal(err)
	}
	event := adksession.NewEvent("adk-invocation")
	event.Author = "streamline_planner"
	event.LLMResponse = model.LLMResponse{
		Content: genai.NewContentFromFunctionCall(
			"finish", map[string]any{"summary": providerSecret}, genai.RoleModel,
		),
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 13, CandidatesTokenCount: 5,
		},
	}
	if err := adk.AppendEvent(ctx, created.Session, event); err != nil {
		t.Fatal(err)
	}
	revision := "result-r1"
	result := contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded, Summary: "done",
		Artifacts: map[string]contracts.ArtifactRef{
			"report": {Namespace: "builder", Name: "report", Revision: &revision},
		},
	}
	if err := service.Complete(ctx, started.Identity, planner.Completion{Result: &result}); err != nil {
		t.Fatal(err)
	}

	recovered, err := service.Begin(ctx, "stage-planner")
	if err != nil || recovered.Invoke || recovered.Completion == nil ||
		!reflect.DeepEqual(*recovered.Completion.Result, result) {
		t.Fatalf("recovery = (%+v, %v)", recovered, err)
	}
	events, err := store.ListPlannerEvents(ctx, started.Identity.SessionID, 0)
	if err != nil || len(events) != 4 || events[0].SequenceNumber != 1 ||
		events[1].SequenceNumber != 2 || events[2].SequenceNumber != 3 ||
		events[3].SequenceNumber != 4 {
		t.Fatalf("events = (%+v, %v)", events, err)
	}
	if strings.Contains(string(events[2].Event), providerSecret) ||
		!strings.Contains(string(events[2].Event), `"finish"`) {
		t.Fatalf("unsafe ADK event = %s", events[2].Event)
	}
	runEvents, err := store.ListRunEvents(ctx, "run-planner", 0, 20)
	if err != nil || len(runEvents) != 8 {
		t.Fatalf("Run events = (%+v, %v)", runEvents, err)
	}
	wantKinds := []runstore.RunEventKind{
		runstore.RunEventPlannerStarted, runstore.RunEventPlannerRequestRecorded,
		runstore.RunEventPlannerActivity, runstore.RunEventPlannerCompleted,
	}
	for index := range events {
		if events[index].RunID == nil || *events[index].RunID != "run-planner" ||
			events[index].RunEventSequence == nil || *events[index].RunEventSequence != int64(index+5) ||
			runEvents[index+4].SequenceNumber != int64(index+5) || runEvents[index+4].Kind != wantKinds[index] ||
			strings.Contains(string(runEvents[index+4].Data), providerSecret) {
			t.Fatalf("event linkage/redaction %d: planner=%+v run=%+v", index, events[index], runEvents[index+4])
		}
	}
	session, err := store.GetPlannerSession(ctx, started.Identity.SessionID)
	if err != nil || session.NextEventSequence != 5 || strings.Contains(string(session.State), providerSecret) {
		t.Fatalf("Planner session = (%+v, %v)", session, err)
	}
	cursor, err := store.GetRunEventCursor(ctx, "run-planner")
	if err != nil || cursor.Sequence != 8 || cursor.Generation == "" {
		t.Fatalf("Run cursor = (%+v, %v)", cursor, err)
	}
	execution, err := store.GetStageExecution(ctx, "stage-planner")
	if err != nil || execution.State != runstore.StageRunning || execution.CandidateResult != nil {
		t.Fatalf("Planner changed result ownership: (%+v, %v)", execution, err)
	}
}

func TestPostgresTypedPlanEventsAreOrderedAndCompareAppendIsAtomic(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPlannerPool(t, ctx)
	store := runstore.NewPostgresStore(pool)
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

	controller, err := planner.NewPlannerPlanController("global-objective-never-copy")
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
	completed, planErr := controller.CompleteDispatch(claim.CallID, contracts.StageSucceeded)
	if planErr != nil {
		t.Fatal(planErr)
	}
	recordPlan(t, service, started.Identity, before, completed, planner.PlannerEventDispatchCompleted)
	recordFact(t, service, started.Identity, planner.PlannerFact{
		Kind: planner.PlannerEventCurrentChanged, Key: "current:4",
		PlanRevision: completed.Revision, SubtaskID: "1",
	})

	base, ok, err := service.LoadPlan(ctx, started.Identity)
	if err != nil || !ok || base.Revision != 4 || base.CurrentSubtaskID != "1" {
		t.Fatalf("base plan = (%+v, %t, %v)", base, ok, err)
	}
	candidates := make([]planner.PlannerPlanProjection, 2)
	for index, objective := range []string{"Race left", "Race right"} {
		candidates[index] = clonePlanProjection(base)
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
	latest, ok, err := service.LoadPlan(ctx, started.Identity)
	if err != nil || !ok || latest.Revision != 5 || len(latest.Subtasks) != 3 {
		t.Fatalf("latest plan = (%+v, %t, %v)", latest, ok, err)
	}
	if err := service.RecordPlan(ctx, started.Identity, planner.PlannerPlanTransition{
		Kind: planner.PlannerEventPlanChanged, ExpectedRevision: base.Revision, Plan: latest,
	}); err != nil {
		t.Fatalf("exact committed retry: %v", err)
	}

	plannerEvents, err := store.ListPlannerEvents(ctx, started.Identity.SessionID, 0)
	if err != nil || len(plannerEvents) != 10 {
		t.Fatalf("Planner events = (%d, %v)", len(plannerEvents), err)
	}
	runEvents, err := store.ListRunEvents(ctx, "run-planner", 0, 100)
	if err != nil || len(runEvents) != 14 {
		t.Fatalf("Run events = (%d, %v)", len(runEvents), err)
	}
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
		plannerSequence := int64(index + 1)
		runSequence := int64(index + 5)
		if plannerEvents[index].SequenceNumber != plannerSequence ||
			plannerEvents[index].RunEventSequence == nil || *plannerEvents[index].RunEventSequence != runSequence ||
			runEvents[index+4].SequenceNumber != runSequence || runEvents[index+4].Kind != wantKinds[index] {
			t.Fatalf("event %d linkage: planner=%+v run=%+v", index, plannerEvents[index], runEvents[index+4])
		}
	}
	cursor, err := store.GetRunEventCursor(ctx, "run-planner")
	if err != nil || cursor.Sequence != 14 || cursor.Generation == "" {
		t.Fatalf("Run cursor = (%+v, %v)", cursor, err)
	}
	page, err := store.ListRunEvents(ctx, "run-planner", 11, 2)
	if err != nil || len(page) != 2 || page[0].SequenceNumber != 12 || page[1].SequenceNumber != 13 {
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
	_, err := store.CreateRun(ctx, runstore.CreateRunParams{
		RunID: "run-planner", OwnerID: "user", WorkflowName: "workflow", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot:      json.RawMessage(`{"name":"workflow"}`),
		Parameters:            map[string]string{"mode": "strict"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(
		ctx, "run-planner", runstore.RunInitializing, runstore.RunRunning,
		runstore.Reason{Code: "initialized"},
	); err != nil {
		t.Fatal(err)
	}
	_, err = store.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: "stage-planner", RunID: "run-planner", StageName: "build", Attempt: 1,
		StageSpecSchemaVersion:    contracts.APIVersion,
		StageSpecSnapshot:         json.RawMessage(`{"objective":"build"}`),
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext:              runstore.StageContextSnapshot{},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func isolatedPlannerPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := cryptorand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_planner_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop Planner test schema: %v", err)
		}
		admin.Close()
	})
	return pool
}
