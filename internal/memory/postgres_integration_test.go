package memory

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/scheduler"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresPlannerMemoryRequiresRunningStage(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedMemoryPool(t, ctx)
	store, err := NewPostgresStore(pool)
	if err != nil {
		t.Fatal(err)
	}

	for _, terminal := range []StageExecutionStateForTest{stageFinalizingForTest, stageAbortingForTest} {
		t.Run(string(terminal), func(t *testing.T) {
			runID := "run-memory-" + string(terminal)
			stageID := "stage-memory-" + string(terminal)
			createRunningMemoryStage(t, ctx, pool, runID, stageID)
			namespace, err := NewNamespace(store, Binding{
				RunID: runID, StageExecutionID: stageID, Namespace: "builder",
			})
			if err != nil {
				t.Fatal(err)
			}
			written, err := namespace.WriteMemory(ctx, "shared_note", "before transition", "", nil)
			if err != nil || written.Content != "before transition" {
				t.Fatalf("initial write = (%+v, %v)", written, err)
			}
			transitionMemoryStage(t, ctx, pool, stageID, terminal)
			if _, err := namespace.AppendMemory(ctx, "shared_note", "late write"); memoryErrorCode(err) != CodeForbidden {
				t.Fatalf("post-transition append error = %v, want %s", err, CodeForbidden)
			}
			if _, err := namespace.ReadMemory(ctx, "shared_note"); memoryErrorCode(err) != CodeForbidden {
				t.Fatalf("post-transition read error = %v, want %s", err, CodeForbidden)
			}
			runArtifacts, _ := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Run(runID)
			stored, err := runArtifacts.Read(ctx, artifacts.ArtifactRef{
				Namespace: "builder", Name: "memory.shared_note",
			})
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := Decode("memory.shared_note", stored.Payload.Data)
			if err != nil || decoded.Content != "before transition" {
				t.Fatalf("stored note after rejected write = (%+v, %v)", decoded, err)
			}
		})
	}
}

func TestPostgresPlannerMemoryWriteLinearizesWithTerminalTransition(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedMemoryPool(t, ctx)
	store, err := NewPostgresStore(pool)
	if err != nil {
		t.Fatal(err)
	}
	for _, target := range []StageExecutionStateForTest{stageFinalizingForTest, stageAbortingForTest} {
		t.Run(string(target), func(t *testing.T) {
			for index := 0; index < 8; index++ {
				runID := fmt.Sprintf("run-memory-race-%s-%d", target, index)
				stageID := fmt.Sprintf("stage-memory-race-%s-%d", target, index)
				createRunningMemoryStage(t, ctx, pool, runID, stageID)
				namespace, err := NewNamespace(store, Binding{
					RunID: runID, StageExecutionID: stageID, Namespace: "builder",
				})
				if err != nil {
					t.Fatal(err)
				}
				start := make(chan struct{})
				var wait sync.WaitGroup
				wait.Add(2)
				var writeErr error
				var transitionErr error
				go func() {
					defer wait.Done()
					<-start
					_, writeErr = namespace.WriteMemory(ctx, "race_note", "complete payload", "", nil)
				}()
				go func() {
					defer wait.Done()
					<-start
					transitionErr = transitionProductionMemoryStage(ctx, pool, runID, stageID, target)
				}()
				close(start)
				wait.Wait()
				if transitionErr != nil {
					t.Fatalf("race %d %s transition: %v", index, target, transitionErr)
				}
				if writeErr != nil && memoryErrorCode(writeErr) != CodeForbidden {
					t.Fatalf("race %d write error = %v", index, writeErr)
				}
				var state runstore.StageExecutionState
				var transitionedAt time.Time
				if err := pool.QueryRow(ctx, `
SELECT state, updated_at FROM stage_executions WHERE stage_execution_id = $1`, stageID).
					Scan(&state, &transitionedAt); err != nil || string(state) != string(target) {
					t.Fatalf("race %d Stage = (%s, %s, %v), want %s", index, state, transitionedAt, err, target)
				}
				var revisionCount int
				var revisionAt *time.Time
				if err := pool.QueryRow(ctx, `
SELECT count(*), max(created_at)
FROM artifact_binding_revisions
WHERE scope_kind = 'run' AND scope_id = $1
  AND namespace = 'builder' AND name = 'memory.race_note'`, runID).
					Scan(&revisionCount, &revisionAt); err != nil {
					t.Fatal(err)
				}
				if writeErr == nil {
					if revisionCount != 1 || revisionAt == nil || revisionAt.After(transitionedAt) {
						t.Fatalf(
							"race %d successful write did not precede transition: revisions=%d revisionAt=%v transitionAt=%s",
							index, revisionCount, revisionAt, transitionedAt,
						)
					}
				} else if revisionCount != 0 || revisionAt != nil {
					t.Fatalf("race %d forbidden write created %d revisions at %v", index, revisionCount, revisionAt)
				}
			}
		})
	}
}

type StageExecutionStateForTest string

const (
	stageFinalizingForTest StageExecutionStateForTest = "finalizing"
	stageAbortingForTest   StageExecutionStateForTest = "aborting"
)

func transitionMemoryStage(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	stageID string,
	target StageExecutionStateForTest,
) {
	t.Helper()
	err := transitionProductionMemoryStage(ctx, pool, "run-memory-"+string(target), stageID, target)
	if err != nil {
		t.Fatalf("enter %s: %v", target, err)
	}
}

func transitionProductionMemoryStage(
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
	stageID string,
	target StageExecutionStateForTest,
) error {
	store, err := scheduler.NewPostgresPersistence(pool)
	if err != nil {
		return err
	}
	switch target {
	case stageFinalizingForTest:
		return store.EnterFinalizingWithResult(ctx, finalizingParams(stageID))
	case stageAbortingForTest:
		now := time.Now().UTC()
		return store.EnterAbortingWithTermination(ctx, runID, runstore.EnterAbortingParams{
			StageExecutionID: stageID, ExpectedState: runstore.StageRunning,
			TerminationSchemaVersion: contracts.APIVersion,
			Termination: runstore.StageTermination{
				Outcome: runstore.TerminationInterrupted, Code: "memory_test_abort",
				Message: "test abort", Retryable: false, Phase: runstore.TerminationRunning,
				OccurredAt: now,
			},
			AbortID: "abort-" + stageID, Deadline: now.Add(time.Minute),
			Reason: runstore.Reason{Code: "memory_test_abort"},
		})
	default:
		return fmt.Errorf("unknown terminal transition %q", target)
	}
}

func finalizingParams(stageID string) runstore.EnterFinalizingParams {
	return runstore.EnterFinalizingParams{
		StageExecutionID: stageID, ResultSchemaVersion: contracts.APIVersion,
		Candidate: contracts.StageContentResult{
			APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded,
			Summary: "memory race complete", Artifacts: map[string]contracts.ArtifactRef{},
		},
		FinalizationID: "finalization-" + stageID, Deadline: time.Now().UTC().Add(time.Minute),
		Reason: runstore.Reason{Code: "memory_test_finalizing"},
	}
}

func createRunningMemoryStage(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
	stageID string,
) {
	t.Helper()
	store := runstore.NewPostgresStore(pool)
	_, err := store.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: "memory-test-user", WorkflowName: "memory-test", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot:      json.RawMessage(`{"name":"memory-test"}`),
		Parameters:            map[string]string{},
		RuntimeConfig:         runtimeconfig.BuiltInRunSnapshot(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(
		ctx, runID, runstore.RunInitializing, runstore.RunRunning,
		runstore.Reason{Code: "initialized"},
	); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: stageID, RunID: runID, StageName: "memory", Attempt: 1,
		StageSpecSchemaVersion:    contracts.APIVersion,
		StageSpecSnapshot:         json.RawMessage(`{"objective":"memory test"}`),
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext:              runstore.StageContextSnapshot{},
	}); err != nil {
		t.Fatal(err)
	}
	sessions, err := plannersession.New(store, plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sessions.Begin(ctx, stageID); err != nil {
		t.Fatal(err)
	}
}

func memoryErrorCode(err error) string {
	var bounded *ToolError
	if errors.As(err, &bounded) {
		return bounded.Code
	}
	return ""
}

func isolatedMemoryPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
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
	schema := "contractor_memory_" + hex.EncodeToString(random)
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
			t.Logf("drop Memory test schema: %v", err)
		}
		admin.Close()
	})
	return pool
}

func TestPostgresPlannerMemoryAddsNoPersistenceObject(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedMemoryPool(t, ctx)
	rows, err := pool.Query(ctx, `
SELECT table_name
FROM information_schema.tables
WHERE table_schema = current_schema() AND table_name LIKE '%memory%'
ORDER BY table_name`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var names []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			t.Fatal(err)
		}
		names = append(names, name)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(names) != 0 {
		t.Fatalf("Memory-specific persistence objects exist: %s", strings.Join(names, ", "))
	}
}
