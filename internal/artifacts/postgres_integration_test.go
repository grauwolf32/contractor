package artifacts

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresIntegrationInputForkAndHistoricalReads(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	createArtifactRun(t, ctx, pool, "run-fork", false)
	service := NewService(NewPostgresRepository(pool))
	user, _ := service.User("user-1")
	run, _ := service.Run("run-fork")

	userWrite, err := user.Write(
		ctx,
		ArtifactRef{Namespace: "projects", Name: "source"},
		Payload{MediaType: "text/plain", Data: []byte("user original")},
		nil,
	)
	if err != nil {
		t.Fatalf("write UserScope source: %v", err)
	}
	fork, err := service.ForkInput(
		ctx, "user-1", ArtifactRef{Namespace: "projects", Name: "source"}, "run-fork", "source",
	)
	if err != nil {
		t.Fatalf("fork input: %v", err)
	}
	if fork.SourceRef.Revision == nil || *fork.SourceRef.Revision != *userWrite.Ref.Revision ||
		fork.TargetRef.Namespace != "inputs" || fork.TargetRef.Revision == nil {
		t.Fatalf("fork result = %+v", fork)
	}

	runUpdate, err := run.Write(
		ctx,
		ArtifactRef{Namespace: "inputs", Name: "source"},
		Payload{MediaType: "text/plain", Data: []byte("run changed")},
		fork.TargetRef.Revision,
	)
	if err != nil {
		t.Fatalf("update RunScope input: %v", err)
	}
	userCurrent, err := user.Read(ctx, ArtifactRef{Namespace: "projects", Name: "source"})
	if err != nil || !bytes.Equal(userCurrent.Payload.Data, []byte("user original")) ||
		*userCurrent.Ref.Revision != *userWrite.Ref.Revision {
		t.Fatalf("UserScope current after Run mutation = (%+v, %v)", userCurrent, err)
	}
	runCurrent, err := run.Read(ctx, ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil || !bytes.Equal(runCurrent.Payload.Data, []byte("run changed")) ||
		*runCurrent.Ref.Revision != *runUpdate.Ref.Revision {
		t.Fatalf("RunScope current = (%+v, %v)", runCurrent, err)
	}
	runHistorical, err := run.Read(ctx, fork.TargetRef)
	if err != nil || !bytes.Equal(runHistorical.Payload.Data, []byte("user original")) {
		t.Fatalf("historical RunScope read = (%+v, %v)", runHistorical, err)
	}

	stale := *fork.TargetRef.Revision
	_, err = run.Write(
		ctx,
		ArtifactRef{Namespace: "inputs", Name: "source"},
		Payload{MediaType: "text/plain", Data: []byte("stale")},
		&stale,
	)
	if !errors.Is(err, ErrArtifactConflict) {
		t.Fatalf("stale RunScope update error = %v", err)
	}
	_, err = run.Write(
		ctx,
		ArtifactRef{Namespace: "analysis", Name: "report"},
		Payload{MediaType: "text/plain", Data: []byte("report")},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	listed, err := run.List(ctx, nil)
	if err != nil || len(listed) != 2 || listed[0].Namespace != "analysis" || listed[1].Namespace != "inputs" ||
		listed[0].Revision != nil || listed[1].Revision != nil {
		t.Fatalf("RunScope list = (%+v, %v)", listed, err)
	}

	var distinctVersions int
	err = pool.QueryRow(ctx, `
SELECT count(DISTINCT revision.version_id)
FROM artifact_binding_revisions AS revision
WHERE (scope_kind, scope_id, namespace, name, revision) IN (
    ('user', 'user-1', 'projects', 'source', $1),
    ('run', 'run-fork', 'inputs', 'source', $2)
)`, *fork.SourceRef.Revision, *fork.TargetRef.Revision).Scan(&distinctVersions)
	if err != nil || distinctVersions != 1 {
		t.Fatalf("fork version reuse = (%d, %v), want one internal version", distinctVersions, err)
	}
	if err := service.PinExact(ctx, mustRunScope(t, "run-fork"), runUpdate.Ref, PinStageContext, "stage-context-1"); err != nil {
		t.Fatalf("pin exact StageContext artifact: %v", err)
	}
	if err := service.PinExact(ctx, mustRunScope(t, "run-fork"), runUpdate.Ref, PinStageContext, "stage-context-1"); err != nil {
		t.Fatalf("repeat idempotent pin: %v", err)
	}

	_, err = pool.Exec(ctx, `
INSERT INTO artifact_bindings (
    scope_kind, scope_id, namespace, name, current_revision
) VALUES ('run', 'run-fork', 'illegal', 'cross-binding', $1)`, *fork.TargetRef.Revision)
	if persistencepostgres.SQLState(err) != "23503" {
		t.Fatalf("cross-binding revision SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
	_, err = pool.Exec(ctx, `
INSERT INTO artifact_binding_revisions (
    scope_kind, scope_id, namespace, name, revision, version_id
)
SELECT scope_kind, scope_id, namespace, name, revision, version_id
FROM artifact_binding_revisions
WHERE scope_kind = 'run' AND scope_id = 'run-fork'
  AND namespace = 'inputs' AND name = 'source' AND revision = $1`, *fork.TargetRef.Revision)
	if persistencepostgres.SQLState(err) != "23505" {
		t.Fatalf("revision reuse SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
	_, err = pool.Exec(ctx, `
INSERT INTO artifact_blobs (sha256, payload, size_bytes)
VALUES (decode(repeat('00', 32), 'hex'), 'bad digest'::bytea, 10)`)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("invalid blob digest SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
	_, err = pool.Exec(ctx, `UPDATE artifact_blobs SET payload = 'changed'::bytea`)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("blob mutation SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
	_, err = pool.Exec(ctx, `DELETE FROM artifact_blobs`)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("blob deletion SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
}

func TestPostgresIntegrationArtifactCASRace(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	service := NewService(NewPostgresRepository(pool))
	user, _ := service.User("user-race")
	initial, err := user.Write(
		ctx,
		ArtifactRef{Namespace: "docs", Name: "report"},
		Payload{MediaType: "text/plain", Data: []byte("initial")},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	expected := *initial.Ref.Revision
	var successes int32
	var conflicts int32
	var wait sync.WaitGroup
	for _, body := range []string{"writer-a", "writer-b"} {
		wait.Add(1)
		go func(body string) {
			defer wait.Done()
			_, err := user.Write(
				ctx,
				ArtifactRef{Namespace: "docs", Name: "report"},
				Payload{MediaType: "text/plain", Data: []byte(body)},
				&expected,
			)
			switch {
			case err == nil:
				atomic.AddInt32(&successes, 1)
			case errors.Is(err, ErrArtifactConflict):
				atomic.AddInt32(&conflicts, 1)
			default:
				t.Errorf("concurrent Write: %v", err)
			}
		}(body)
	}
	wait.Wait()
	if successes != 1 || conflicts != 1 {
		t.Fatalf("CAS race successes=%d conflicts=%d", successes, conflicts)
	}
	historical, err := user.Read(ctx, initial.Ref)
	if err != nil || !bytes.Equal(historical.Payload.Data, []byte("initial")) {
		t.Fatalf("historical read after race = (%+v, %v)", historical, err)
	}
	current, err := user.Read(ctx, ArtifactRef{Namespace: "docs", Name: "report"})
	if err != nil || bytes.Equal(current.Payload.Data, []byte("initial")) {
		t.Fatalf("current read after race = (%+v, %v)", current, err)
	}
}

func TestPostgresIntegrationConcurrentFirstScopeAndBlobWrites(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)

	t.Run("scope created by concurrent transaction", func(t *testing.T) {
		createArtifactRun(t, ctx, pool, "run-scope-race", true)
		tx, err := pool.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		first, _ := NewService(NewPostgresRepository(tx)).Run("run-scope-race")
		if _, err := first.Write(
			ctx, ArtifactRef{Namespace: "builder", Name: "first"},
			Payload{MediaType: "text/plain", Data: []byte("first payload")}, nil,
		); err != nil {
			_ = tx.Rollback(ctx)
			t.Fatal(err)
		}
		conn, err := pool.Acquire(ctx)
		if err != nil {
			_ = tx.Rollback(ctx)
			t.Fatal(err)
		}
		defer conn.Release()
		pid := postgresBackendPID(t, ctx, conn)
		second, _ := NewService(NewPostgresRepository(conn)).Run("run-scope-race")
		done := make(chan error, 1)
		go func() {
			_, writeErr := second.Write(
				ctx, ArtifactRef{Namespace: "builder", Name: "second"},
				Payload{MediaType: "text/plain", Data: []byte("second payload")}, nil,
			)
			done <- writeErr
		}()
		commitBlockedWriter(t, ctx, pool, tx, pid, done)
	})

	t.Run("blob committed by concurrent transaction", func(t *testing.T) {
		createArtifactRun(t, ctx, pool, "run-blob-race", true)
		tx, err := pool.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		first, _ := NewService(NewPostgresRepository(tx)).Run("run-blob-race")
		payload := Payload{MediaType: "text/plain", Data: []byte("shared payload")}
		if _, err := first.Write(
			ctx, ArtifactRef{Namespace: "builder", Name: "first"}, payload, nil,
		); err != nil {
			_ = tx.Rollback(ctx)
			t.Fatal(err)
		}
		conn, err := pool.Acquire(ctx)
		if err != nil {
			_ = tx.Rollback(ctx)
			t.Fatal(err)
		}
		defer conn.Release()
		pid := postgresBackendPID(t, ctx, conn)
		second, _ := NewService(NewPostgresRepository(conn)).Run("run-blob-race")
		done := make(chan error, 1)
		go func() {
			_, writeErr := second.Write(
				ctx, ArtifactRef{Namespace: "builder", Name: "second"}, payload, nil,
			)
			done <- writeErr
		}()
		commitBlockedWriter(t, ctx, pool, tx, pid, done)
	})
}

func TestPostgresIntegrationFreezeEmptyRunOutputs(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	createArtifactRun(t, ctx, pool, "run-empty-freeze", true)
	service := NewService(NewPostgresRepository(pool))
	if err := service.FreezeRunOutputs(ctx, "run-empty-freeze"); err != nil {
		t.Fatalf("freeze empty Run outputs: %v", err)
	}
	var scopes int
	if err := pool.QueryRow(ctx, `
SELECT count(*) FROM artifact_scopes
WHERE scope_kind = 'run' AND scope_id = 'run-empty-freeze'`).Scan(&scopes); err != nil {
		t.Fatal(err)
	}
	if scopes != 0 {
		t.Fatalf("empty freeze created an unnecessary Artifact scope: %d", scopes)
	}
}

func postgresBackendPID(t *testing.T, ctx context.Context, db persistencepostgres.DBTX) int32 {
	t.Helper()
	var pid int32
	if err := db.QueryRow(ctx, `SELECT pg_backend_pid()`).Scan(&pid); err != nil {
		t.Fatal(err)
	}
	return pid
}

func commitBlockedWriter(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	tx pgx.Tx,
	blockedPID int32,
	done <-chan error,
) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		select {
		case err := <-done:
			_ = tx.Rollback(ctx)
			t.Fatalf("concurrent write completed before blocker committed: %v", err)
		default:
		}
		var blocked bool
		if err := pool.QueryRow(ctx,
			`SELECT cardinality(pg_blocking_pids($1)) > 0`, blockedPID,
		).Scan(&blocked); err != nil {
			_ = tx.Rollback(ctx)
			t.Fatal(err)
		}
		if blocked {
			break
		}
		if time.Now().After(deadline) {
			_ = tx.Rollback(ctx)
			t.Fatal("concurrent artifact write did not reach the expected PostgreSQL conflict wait")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("concurrent artifact write after winner commit: %v", err)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
}

func TestPostgresIntegrationOutputBindingSharesCallerTransaction(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	createArtifactRun(t, ctx, pool, "run-output", true)
	service := NewService(NewPostgresRepository(pool))
	run, _ := service.Run("run-output")

	selected, err := run.Write(
		ctx,
		ArtifactRef{Namespace: "builder", Name: "result"},
		Payload{MediaType: "application/json", Data: []byte(`{"version":1}`)},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	_, err = run.Write(
		ctx,
		ArtifactRef{Namespace: "builder", Name: "result"},
		Payload{MediaType: "application/json", Data: []byte(`{"version":2}`)},
		selected.Ref.Revision,
	)
	if err != nil {
		t.Fatal(err)
	}

	rollback := errors.New("rollback output")
	err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		txService := NewService(NewPostgresRepository(tx))
		if _, err := txService.BindOutputExact(ctx, "run-output", "result", selected.Ref, nil); err != nil {
			return err
		}
		outside, _ := service.Run("run-output")
		if _, err := outside.Read(ctx, ArtifactRef{Namespace: "outputs", Name: "result"}); !errors.Is(err, ErrArtifactNotFound) {
			return errors.New("uncommitted output became externally visible")
		}
		return rollback
	})
	if !errors.Is(err, rollback) {
		t.Fatalf("rollback transaction = %v", err)
	}
	if _, err := run.Read(ctx, ArtifactRef{Namespace: "outputs", Name: "result"}); !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("rolled-back output read error = %v", err)
	}

	err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		txService := NewService(NewPostgresRepository(tx))
		if _, err := txService.BindOutputExact(ctx, "run-output", "result", selected.Ref, nil); err != nil {
			return err
		}
		if err := txService.FreezeRunOutputs(ctx, "run-output"); err != nil {
			return err
		}
		_, err := runstore.NewPostgresStore(tx).TransitionRun(
			ctx, "run-output", runstore.RunRunning, runstore.RunSucceeded,
			runstore.Reason{Code: "outputs_frozen"},
		)
		return err
	})
	if err != nil {
		t.Fatalf("commit output and Run success: %v", err)
	}
	output, err := run.Read(ctx, ArtifactRef{Namespace: "outputs", Name: "result"})
	if err != nil || !bytes.Equal(output.Payload.Data, []byte(`{"version":1}`)) {
		t.Fatalf("frozen output = (%+v, %v)", output, err)
	}
	currentIntermediate, err := run.Read(ctx, ArtifactRef{Namespace: "builder", Name: "result"})
	if err != nil || !bytes.Equal(currentIntermediate.Payload.Data, []byte(`{"version":2}`)) {
		t.Fatalf("current intermediate = (%+v, %v)", currentIntermediate, err)
	}
	_, err = service.BindOutputExact(
		ctx, "run-output", "result", currentIntermediate.Ref, output.Ref.Revision,
	)
	if !errors.Is(err, ErrArtifactFrozen) {
		t.Fatalf("update frozen output error = %v", err)
	}
}

func createArtifactRun(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
	start bool,
) {
	t.Helper()
	store := runstore.NewPostgresStore(pool)
	_, err := store.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1",
		WorkflowSchemaVersion: "contractor/v1alpha1",
		WorkflowSnapshot:      json.RawMessage(`{"ref":{"name":"artifact-copy","version":"1"}}`),
	})
	if err != nil {
		t.Fatalf("create WorkflowRun: %v", err)
	}
	if start {
		if _, err := store.TransitionRun(
			ctx, runID, runstore.RunInitializing, runstore.RunRunning,
			runstore.Reason{Code: "initialized"},
		); err != nil {
			t.Fatalf("start WorkflowRun: %v", err)
		}
	}
}

func isolatedArtifactPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatalf("parse test database URL: %v", err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatalf("open test admin pool: %v", err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatalf("ping test database: %v", err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_test_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatalf("create isolated schema: %v", err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatalf("open isolated pool: %v", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		t.Fatalf("ping isolated pool: %v", err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		t.Fatalf("apply migrations: %v", err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanupCtx, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop isolated schema: %v", err)
		}
		admin.Close()
	})
	return pool
}

func mustRunScope(t *testing.T, runID string) Scope {
	t.Helper()
	scope, err := RunScope(runID)
	if err != nil {
		t.Fatal(err)
	}
	return scope
}
