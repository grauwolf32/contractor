package postgres

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestPostgresEvalViewMembersFoldIntoPairsMigration(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 84)

	_, err := pool.Exec(ctx, `
INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES ('eval-project','owner','evaluation','Eval project','create-project','sha256:'||repeat('a',64));
INSERT INTO eval_experiments(
    experiment_id,owner_id,project_id,portable_id,control_mode,name,state,max_in_flight,wall_ms)
VALUES ('experiment','owner','eval-project','portable','external','Experiment','running',1,1000);
INSERT INTO eval_view_generations(experiment_id,generation,snapshot_id,summary)
VALUES ('experiment',1,'view-1',convert_to('{}','UTF8'));
INSERT INTO eval_view_members(experiment_id,generation,ordinal,member_id,pair_id,document,collection_complete)
VALUES ('experiment',1,0,repeat('a',64),repeat('c',64),convert_to('{}','UTF8'),true),
       ('experiment',1,1,repeat('b',64),repeat('c',64),convert_to('{}','UTF8'),false);
INSERT INTO eval_view_pairs(experiment_id,generation,ordinal,pair_id,suite_id,document,regression,unresolved)
VALUES ('experiment',1,0,repeat('c',64),'suite',
    convert_to(jsonb_build_object(
        'a', jsonb_build_object('member', jsonb_build_object('memberId', repeat('a',64))),
        'b', jsonb_build_object('member', jsonb_build_object('memberId', repeat('b',64)))
    )::text,'UTF8'),false,true);
INSERT INTO eval_projection_queue(experiment_id,revision,published_revision,snapshot_id)
VALUES ('experiment',1,1,'view-1');
`)
	if err != nil {
		t.Fatal(err)
	}

	result, err := ApplyMigrations(ctx, pool)
	if err != nil || !slices.Contains(result.AppliedVersions, 85) {
		t.Fatalf("fold member views = %+v, %v", result, err)
	}
	var completeA, completeB, membersGone bool
	if err := pool.QueryRow(ctx, `
SELECT complete_a, complete_b, to_regclass('eval_view_members') IS NULL
FROM eval_view_pairs WHERE experiment_id = 'experiment'`).Scan(&completeA, &completeB, &membersGone); err != nil {
		t.Fatal(err)
	}
	if !completeA || completeB || !membersGone {
		t.Fatalf("pair completeness = (%t, %t), member table dropped %t", completeA, completeB, membersGone)
	}
	var pgErr *pgconn.PgError
	_, err = pool.Exec(ctx, `UPDATE eval_view_pairs SET complete_b = true`)
	if !errors.As(err, &pgErr) || pgErr.Code != "23514" {
		t.Fatalf("current pair changed after backfill: %v", err)
	}
}
