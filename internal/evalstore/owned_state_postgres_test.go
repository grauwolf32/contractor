package evalstore

import (
	"testing"

	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestSharedPostgresEvalObservationsDoNotRehashFrozenPlan(t *testing.T) {
	pool := testPool(t)
	f := newNativeFreeze(t, pool)
	// Replace only the generated expression with a counted equivalent in this
	// private test schema. Reset the counter after the DDL backfills existing
	// rows, then prove that freeze evaluates it once and observations never do.
	if _, err := pool.Exec(t.Context(), `
CREATE SEQUENCE plan_digest_calls;
CREATE FUNCTION counted_plan_digest(document bytea) RETURNS text LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
    PERFORM nextval('plan_digest_calls');
    RETURN CASE WHEN document IS NOT NULL THEN 'sha256:' || encode(sha256(document),'hex') END;
END; $$;
ALTER TABLE eval_experiments ALTER COLUMN plan_document_sha256 SET EXPRESSION AS (counted_plan_digest(plan_document));
ALTER SEQUENCE plan_digest_calls RESTART WITH 1`); err != nil {
		t.Fatal(err)
	}
	mustTx(t, pool, func(s *Store) error { return f.freeze(t.Context(), s) })
	if _, err := pool.Exec(t.Context(), `UPDATE eval_experiments SET observed_tokens=observed_tokens+1,updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond') WHERE experiment_id=$1`, f.e.ID); err != nil {
		t.Fatal(err)
	}
	var calls int
	if err := pool.QueryRow(t.Context(), `SELECT last_value FROM plan_digest_calls`).Scan(&calls); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("freeze plus progress hashed the plan %d times, want once", calls)
	}
	var correct bool
	if err := pool.QueryRow(t.Context(), `SELECT plan_document_sha256='sha256:'||encode(sha256(plan_document),'hex') FROM eval_experiments WHERE experiment_id=$1`, f.e.ID).Scan(&correct); err != nil || !correct {
		t.Fatalf("database digest differs from frozen bytes: %v", err)
	}
}

func TestSharedPostgresEvalSelectionClearResumesWithinPurgeBudget(t *testing.T) {
	pool := testPool(t)
	scope := setupProject(t, pool, "owner", "eval")
	const id = "selected-cleanup"
	const n = 1200
	seedSelectedView(t, pool, scope, id, n, 100, nil)
	// This is a storage fixture: all expected members have a retained result
	// and select it. No provider or Runtime execution is needed for FK/purge.
	for _, statement := range []string{`
INSERT INTO eval_records(experiment_id,member_id,kind,record_sha256,document,actor_id)
SELECT experiment_id,member_id,'result','sha256:'||encode(sha256('{}'::bytea),'hex'),'{}','owner'
FROM eval_members WHERE experiment_id=$1`, `
UPDATE eval_members SET selected_result_sha256='sha256:'||encode(sha256('{}'::bytea),'hex'),selection_revision=1,selected_by='owner'
WHERE experiment_id=$1`} {
		if _, err := pool.Exec(t.Context(), statement, id); err != nil {
			t.Fatal(err)
		}
	}
	var previousRevision int64
	if err := pool.QueryRow(t.Context(), `SELECT revision FROM eval_projection_queue WHERE experiment_id=$1`, id).Scan(&previousRevision); err != nil {
		t.Fatal(err)
	}
	for pass := 0; pass < 2; pass++ {
		mustTx(t, pool, func(s *Store) error {
			if _, err := s.locked(t.Context(), scope, id); err != nil {
				return err
			}
			if _, err := s.db.Exec(t.Context(), `SELECT experiment_id FROM eval_projection_queue WHERE experiment_id=$1 FOR UPDATE`, id); err != nil {
				return err
			}
			budget := newPurgeBudget()
			done, err := s.clearPurgedSelections(t.Context(), id, &budget)
			if err != nil {
				return err
			}
			want := int64(pg.CleanupBatchRows)
			if pass == 1 {
				want = n - pg.CleanupBatchRows
			}
			if budget.Rows != want || budget.Bytes <= 0 || budget.Bytes > purgeBatchBytes || done != (pass == 1) {
				t.Fatalf("pass %d: done=%v budget=%+v", pass, done, budget)
			}
			return nil
		})
	}
	var revision int64
	if err := pool.QueryRow(t.Context(), `SELECT revision FROM eval_projection_queue WHERE experiment_id=$1`, id).Scan(&revision); err != nil {
		t.Fatal(err)
	}
	if revision != previousRevision {
		t.Fatal("selection cleanup dirtied the projection")
	}
	if _, err := pool.Exec(t.Context(), `SELECT set_config('contractor.eval_purge','on',false); DELETE FROM eval_records WHERE experiment_id='selected-cleanup'`); err != nil {
		t.Fatal("cleared selections still block record deletion:", err)
	}
	if got := rows(t, pool, `SELECT count(*) FROM eval_members WHERE experiment_id=$1`, id); got != n {
		t.Fatalf("cleanup removed members: %d", got)
	}
}

func TestSharedPostgresEvalPlanStillRequiresCompleteMatrix(t *testing.T) {
	pool := testPool(t)
	f := newNativeFreeze(t, pool)
	if _, err := pool.Exec(t.Context(), `INSERT INTO eval_plan_resources(experiment_id,resource_path,document_kind,document) VALUES($1,'case.json','playground.case/v2','{}')`, f.e.ID); pg.SQLState(err) != pg.SQLStateCheckViolation {
		t.Fatalf("resource without frozen plan: %v", err)
	}
	err := transaction(pool, func(s *Store) error {
		_, err := s.db.Exec(t.Context(), `
UPDATE eval_experiments SET expected_count=1,plan_kind='ExternalRegistration',plan_document='{}',
 plan_sha256='sha256:'||repeat('a',64),plan_setup='{}',plan_created_at=clock_timestamp(),
 updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond') WHERE experiment_id=$1`, f.e.ID)
		return err
	})
	if pg.SQLState(err) != pg.SQLStateCheckViolation {
		t.Fatalf("incomplete matrix committed: %v", err)
	}
	assertNothingFrozen(t, pool, f.e.ID)
	// The complete native transaction still succeeds after the rejected freeze.
	mustTx(t, pool, func(s *Store) error { return f.freeze(t.Context(), s) })
}
