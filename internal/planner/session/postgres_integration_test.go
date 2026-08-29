package session

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"os"
	"reflect"
	"strings"
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
	ids := []string{"session-1", "invocation-1", "event-1", "event-adk", "event-2"}
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
	if err != nil || len(events) != 3 || events[0].SequenceNumber != 1 ||
		events[1].SequenceNumber != 2 || events[2].SequenceNumber != 3 {
		t.Fatalf("events = (%+v, %v)", events, err)
	}
	if strings.Contains(string(events[1].Event), providerSecret) ||
		!strings.Contains(string(events[1].Event), `"finish"`) {
		t.Fatalf("unsafe ADK event = %s", events[1].Event)
	}
	execution, err := store.GetStageExecution(ctx, "stage-planner")
	if err != nil || execution.State != runstore.StageRunning || execution.CandidateResult != nil {
		t.Fatalf("Planner changed result ownership: (%+v, %v)", execution, err)
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
