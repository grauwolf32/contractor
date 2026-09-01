package memory

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
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

func TestPostgresPlannerMemoryCommitAmbiguityReplaysExactMutation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedMemoryPool(t, ctx)
	for _, mutation := range []string{"create", "replace", "append"} {
		for _, loss := range []string{"before_commit", "after_commit"} {
			t.Run(mutation+"/"+loss, func(t *testing.T) {
				runID := "run-memory-ambiguity-" + mutation + "-" + loss
				stageID := "stage-memory-ambiguity-" + mutation + "-" + loss
				createRunningMemoryStage(t, ctx, pool, runID, stageID)
				store, err := NewPostgresStore(pool)
				if err != nil {
					t.Fatal(err)
				}
				recording := &recordingMemoryStore{Store: store}
				namespace, err := NewNamespace(recording, Binding{
					RunID: runID, StageExecutionID: stageID, Namespace: "builder",
				})
				if err != nil {
					t.Fatal(err)
				}
				baselineRevisions := 0
				if mutation != "create" {
					if _, err := namespace.WriteMemory(ctx, "shared", "base", "baseline", []string{"base"}); err != nil {
						t.Fatal(err)
					}
					baselineRevisions = 1
					recording.writes = nil
				}

				commitCalls := 0
				store.writeBoundary = &postgresWriteBoundary{commit: func(commitContext context.Context, tx pgx.Tx) error {
					commitCalls++
					if commitCalls == 1 {
						if loss == "after_commit" {
							if err := tx.Commit(commitContext); err != nil {
								return err
							}
						}
						return errors.New("synthetic Planner Memory commit response loss")
					}
					return tx.Commit(commitContext)
				}}

				var note Note
				switch mutation {
				case "create":
					note, err = namespace.WriteMemory(ctx, "shared", "created", "", nil)
				case "replace":
					note, err = namespace.WriteMemory(ctx, "shared", "replacement", "", nil)
				case "append":
					note, err = namespace.AppendMemory(ctx, "shared", "appended")
				}
				if err != nil {
					t.Fatalf("%s with %s = %v", mutation, loss, err)
				}
				wantContent := map[string]string{
					"create": "created", "replace": "replacement", "append": "base\nappended",
				}[mutation]
				if note.Content != wantContent {
					t.Fatalf("%s content = %q, want %q", mutation, note.Content, wantContent)
				}
				if len(recording.writes) != 2 || !reflect.DeepEqual(recording.writes[0], recording.writes[1]) {
					t.Fatalf("%s replay changed bytes or precondition: %+v", mutation, recording.writes)
				}
				if mutation == "create" && recording.writes[0].expectedRevision != nil ||
					mutation != "create" && recording.writes[0].expectedRevision == nil {
					t.Fatalf("%s replay precondition = %v", mutation, recording.writes[0].expectedRevision)
				}
				if got := countMemoryRevisions(t, ctx, pool, runID, "shared"); got != baselineRevisions+1 {
					t.Fatalf("%s semantic revision count = %d, want %d", mutation, got, baselineRevisions+1)
				}
			})
		}
	}
}

func TestPostgresPlannerMemorySameBindingInterferenceNeverOverwrites(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedMemoryPool(t, ctx)
	for _, boundary := range []string{"before_write", "after_ambiguous_commit"} {
		t.Run(boundary, func(t *testing.T) {
			runID := "run-memory-interference-" + boundary
			stageID := "stage-memory-interference-" + boundary
			createRunningMemoryStage(t, ctx, pool, runID, stageID)
			store, err := NewPostgresStore(pool)
			if err != nil {
				t.Fatal(err)
			}
			recording := &recordingMemoryStore{Store: store}
			namespace, err := NewNamespace(recording, Binding{
				RunID: runID, StageExecutionID: stageID, Namespace: "builder",
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := namespace.WriteMemory(ctx, "shared", "base", "", nil); err != nil {
				t.Fatal(err)
			}
			recording.writes = nil
			externalPayload := encodedTestNote(t, "shared", "external winner", 0)
			runArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Run(runID)
			if err != nil {
				t.Fatal(err)
			}

			if boundary == "before_write" {
				recording.beforeWrite = func(
					_ Binding, target artifacts.ArtifactRef, _ artifacts.Payload, expected *string,
				) error {
					_, err := runArtifacts.Write(ctx, target, artifacts.Payload{
						MediaType: MediaType, Data: externalPayload,
					}, expected)
					return err
				}
			} else {
				commitCalls := 0
				store.writeBoundary = &postgresWriteBoundary{commit: func(commitContext context.Context, tx pgx.Tx) error {
					commitCalls++
					if commitCalls != 1 {
						return tx.Commit(commitContext)
					}
					if err := tx.Commit(commitContext); err != nil {
						return err
					}
					current, err := runArtifacts.Read(commitContext, artifacts.ArtifactRef{
						Namespace: "builder", Name: "memory.shared",
					})
					if err != nil {
						return err
					}
					_, err = runArtifacts.Write(commitContext, artifacts.ArtifactRef{
						Namespace: "builder", Name: "memory.shared",
					}, artifacts.Payload{MediaType: MediaType, Data: externalPayload}, current.Ref.Revision)
					if err != nil {
						return err
					}
					return errors.New("synthetic response loss after commit and external advance")
				}}
			}

			_, err = namespace.AppendMemory(ctx, "shared", "stale append")
			assertToolError(t, err, CodeChanged, true)
			if got := readPlannerMemoryContent(t, ctx, runArtifacts, "shared"); got != "external winner" {
				t.Fatalf("current content = %q, want external winner", got)
			}
			wantAttempts := 1
			wantRevisions := 2
			if boundary == "after_ambiguous_commit" {
				wantAttempts = 2
				wantRevisions = 3
				if !reflect.DeepEqual(recording.writes[0], recording.writes[1]) {
					t.Fatalf("ambiguous retry changed bytes or precondition: %+v", recording.writes)
				}
			}
			if len(recording.writes) != wantAttempts {
				t.Fatalf("write attempts = %d, want %d", len(recording.writes), wantAttempts)
			}
			if got := countMemoryRevisions(t, ctx, pool, runID, "shared"); got != wantRevisions {
				t.Fatalf("revision count = %d, want %d", got, wantRevisions)
			}
		})
	}
}

func TestPostgresPlannerMemoryWriteLinearizesWithTerminalTransition(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedMemoryPool(t, ctx)
	for _, target := range []StageExecutionStateForTest{stageFinalizingForTest, stageAbortingForTest} {
		for _, winner := range []string{"write", "transition"} {
			t.Run(string(target)+"/"+winner+"_wins", func(t *testing.T) {
				runID := "run-memory-race-" + string(target) + "-" + winner
				stageID := "stage-memory-race-" + string(target) + "-" + winner
				createRunningMemoryStage(t, ctx, pool, runID, stageID)
				store, err := NewPostgresStore(pool)
				if err != nil {
					t.Fatal(err)
				}
				namespace, err := NewNamespace(store, Binding{
					RunID: runID, StageExecutionID: stageID, Namespace: "builder",
				})
				if err != nil {
					t.Fatal(err)
				}
				paused := make(chan struct{})
				release := make(chan struct{})
				pause := func(boundaryContext context.Context) {
					close(paused)
					select {
					case <-release:
					case <-boundaryContext.Done():
					}
				}
				store.writeBoundary = &postgresWriteBoundary{}
				if winner == "write" {
					store.writeBoundary.beforeCommit = pause
				} else {
					store.writeBoundary.beforeLock = pause
				}
				writeDone := make(chan error, 1)
				go func() {
					_, writeErr := namespace.WriteMemory(ctx, "race_note", "complete payload", "", nil)
					writeDone <- writeErr
				}()
				waitMemorySignal(t, paused, "Planner write boundary")

				var writeErr, transitionErr error
				if winner == "write" {
					transitionDone := make(chan error, 1)
					transitionStarted := make(chan struct{})
					go func() {
						close(transitionStarted)
						transitionDone <- transitionProductionMemoryStage(ctx, pool, runID, stageID, target)
					}()
					waitMemorySignal(t, transitionStarted, "terminal transition start")
					select {
					case early := <-transitionDone:
						t.Fatalf("terminal transition overtook Planner write: %v", early)
					case <-time.After(25 * time.Millisecond):
					}
					close(release)
					writeErr = <-writeDone
					transitionErr = <-transitionDone
				} else {
					transitionErr = transitionProductionMemoryStage(ctx, pool, runID, stageID, target)
					close(release)
					writeErr = <-writeDone
				}
				if transitionErr != nil {
					t.Fatalf("%s transition: %v", target, transitionErr)
				}
				if winner == "write" && writeErr != nil {
					t.Fatalf("write-winner mutation: %v", writeErr)
				}
				if winner == "transition" && memoryErrorCode(writeErr) != CodeForbidden {
					t.Fatalf("transition-winner write error = %v, want %s", writeErr, CodeForbidden)
				}
				assertPlannerMemoryRaceState(t, ctx, pool, runID, stageID, target, winner)
			})
		}
	}
}

type plannerMemoryWriteAttempt struct {
	binding          Binding
	target           artifacts.ArtifactRef
	payload          artifacts.Payload
	expectedRevision *string
}

type recordingMemoryStore struct {
	Store
	writes      []plannerMemoryWriteAttempt
	beforeWrite func(Binding, artifacts.ArtifactRef, artifacts.Payload, *string) error
}

func (s *recordingMemoryStore) Write(
	ctx context.Context,
	binding Binding,
	target artifacts.ArtifactRef,
	payload artifacts.Payload,
	expectedRevision *string,
) (artifacts.WriteResult, error) {
	attempt := plannerMemoryWriteAttempt{
		binding: binding, target: target,
		payload: artifacts.Payload{
			MediaType: payload.MediaType, Data: append([]byte(nil), payload.Data...),
		},
		expectedRevision: cloneString(expectedRevision),
	}
	s.writes = append(s.writes, attempt)
	if s.beforeWrite != nil {
		hook := s.beforeWrite
		s.beforeWrite = nil
		if err := hook(binding, target, payload, expectedRevision); err != nil {
			return artifacts.WriteResult{}, err
		}
	}
	return s.Store.Write(ctx, binding, target, payload, expectedRevision)
}

func countMemoryRevisions(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
	name string,
) int {
	t.Helper()
	var count int
	if err := pool.QueryRow(ctx, `
SELECT count(*)
FROM artifact_binding_revisions
WHERE scope_kind = 'run' AND scope_id = $1
  AND namespace = 'builder' AND name = $2`, runID, "memory."+name).Scan(&count); err != nil {
		t.Fatal(err)
	}
	return count
}

func readPlannerMemoryContent(
	t *testing.T,
	ctx context.Context,
	store artifacts.ScopedStore,
	name string,
) string {
	t.Helper()
	value, err := store.Read(ctx, artifacts.ArtifactRef{
		Namespace: "builder", Name: "memory." + name,
	})
	if err != nil {
		t.Fatal(err)
	}
	note, err := Decode("memory."+name, value.Payload.Data)
	if err != nil {
		t.Fatal(err)
	}
	return note.Content
}

func waitMemorySignal(t *testing.T, signal <-chan struct{}, description string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", description)
	}
}

func assertPlannerMemoryRaceState(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
	stageID string,
	target StageExecutionStateForTest,
	winner string,
) {
	t.Helper()
	var state runstore.StageExecutionState
	var transitionedAt time.Time
	if err := pool.QueryRow(ctx, `
SELECT state, updated_at FROM stage_executions WHERE stage_execution_id = $1`, stageID).
		Scan(&state, &transitionedAt); err != nil || string(state) != string(target) {
		t.Fatalf("Stage = (%s, %s, %v), want %s", state, transitionedAt, err, target)
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
	if winner == "write" {
		if revisionCount != 1 || revisionAt == nil || revisionAt.After(transitionedAt) {
			t.Fatalf(
				"successful write did not precede transition: revisions=%d revisionAt=%v transitionAt=%s",
				revisionCount, revisionAt, transitionedAt,
			)
		}
	} else if revisionCount != 0 || revisionAt != nil {
		t.Fatalf("forbidden write created %d revisions at %v", revisionCount, revisionAt)
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
