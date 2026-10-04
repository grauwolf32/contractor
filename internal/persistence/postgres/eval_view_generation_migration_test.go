package postgres

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestPostgresEvalViewGenerationPruningMigration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 77)

	_, err := pool.Exec(ctx, `
INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES ('eval-project','owner','evaluation','Eval project','create-project','sha256:'||repeat('a',64));
INSERT INTO eval_experiments(
    experiment_id,owner_id,project_id,portable_id,control_mode,name,state,max_in_flight,wall_ms)
VALUES ('experiment','owner','eval-project','portable','external','Experiment','running',1,1000);
INSERT INTO eval_view_generations(experiment_id,generation,snapshot_id,summary)
SELECT 'experiment', generation, 'view-'||generation, convert_to('{}','UTF8') FROM generate_series(1,3) generation;
INSERT INTO eval_projection_queue(experiment_id,revision,published_revision,snapshot_id)
VALUES ('experiment',4,3,'view-3');
`)
	if err != nil {
		t.Fatal(err)
	}

	result, err := ApplyMigrations(ctx, pool)
	if err != nil || !slices.Contains(result.AppliedVersions, 78) {
		t.Fatalf("upgrade view generations = %+v, %v", result, err)
	}
	// Later migrations fold the surviving generation into the queue row.
	var generations []int64
	if err := pool.QueryRow(ctx, `
SELECT array_agg(generation ORDER BY generation) FROM eval_projection_queue WHERE generation IS NOT NULL
`).Scan(&generations); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(generations, []int64{3}) {
		t.Fatalf("retained generations=%v, want only the current generation 3", generations)
	}
	var pgErr *pgconn.PgError
	_, err = pool.Exec(ctx, `
UPDATE eval_projection_queue SET content_sha256 = 'not-a-digest' WHERE experiment_id = 'experiment'`)
	if !errors.As(err, &pgErr) || pgErr.Code != "23514" {
		t.Fatalf("malformed content digest accepted: %v", err)
	}
	if replay, err := ApplyMigrations(ctx, pool); err != nil || len(replay.AppliedVersions) != 0 {
		t.Fatalf("migration replay = %+v, %v", replay, err)
	}
}
