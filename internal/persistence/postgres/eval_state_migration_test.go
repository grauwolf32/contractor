package postgres

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestPostgresEvalInterruptedStateMigration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 74)

	_, err := pool.Exec(ctx, `
INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES ('eval-project','owner','evaluation','Eval project','create-project','sha256:'||repeat('a',64));
INSERT INTO eval_experiments(
    experiment_id,owner_id,project_id,portable_id,control_mode,name,state,max_in_flight,wall_ms)
VALUES ('experiment','owner','eval-project','portable','server','Experiment','paused',1,1000);
`)
	if err != nil {
		t.Fatal(err)
	}

	result, err := ApplyMigrations(ctx, pool)
	if err != nil || len(result.AppliedVersions) == 0 || result.AppliedVersions[0] != 75 {
		t.Fatalf("upgrade Eval state constraint = %+v, %v", result, err)
	}
	var state string
	if err := pool.QueryRow(ctx, `SELECT state FROM eval_experiments WHERE experiment_id='experiment'`).Scan(&state); err != nil || state != "paused" {
		t.Fatalf("upgrade changed retained experiment state = %q, %v", state, err)
	}
	_, err = pool.Exec(ctx, `
INSERT INTO eval_experiments(
    experiment_id,owner_id,project_id,portable_id,control_mode,name,state,max_in_flight,wall_ms)
VALUES ('experiment-interrupted','owner','eval-project','portable-interrupted','server','Invalid experiment','interrupted',1,1000)
`)
	var pgErr *pgconn.PgError
	if !errors.As(err, &pgErr) || pgErr.Code != "23514" || pgErr.ConstraintName != "eval_experiments_state_check" {
		t.Fatalf("interrupted experiment state passed tightened constraint: %v", err)
	}
	if replay, err := ApplyMigrations(ctx, pool); err != nil || len(replay.AppliedVersions) != 0 {
		t.Fatalf("migration replay = %+v, %v", replay, err)
	}
}
