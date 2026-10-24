package evalstore

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// consolidationExperimentsSQL loads 900 experiments for each of three owners
// over four evaluation Projects. Three consecutive experiments share
// created_at; one in a hundred is paused and one in a hundred external; every
// second names one of 20 datasets of its Project.
const consolidationExperimentsSQL = `
INSERT INTO eval_experiments (experiment_id, owner_id, project_id, portable_id, control_mode, name, state,
    dataset_id, dataset_revision, max_in_flight, wall_ms, created_at, updated_at)
SELECT 'exp-' || o || '-' || lpad(i::text, 4, '0'), 'owner-' || o, 'project-' || o || '-' || (i % 4),
       'portable-' || i, CASE WHEN i % 100 = 0 THEN 'external' ELSE 'server' END, 'Experiment ' || i,
       CASE WHEN i % 100 = 7 THEN 'paused' WHEN i % 10 = 1 THEN 'running' WHEN i % 10 = 2 THEN 'draft'
            WHEN i % 10 = 3 THEN 'cancelled' ELSE 'finished' END,
       CASE WHEN i % 2 = 0 THEN 'dataset-' || ((i / 2) % 20) END, CASE WHEN i % 2 = 0 THEN '1' END,
       1, 1000,
       timestamptz '2026-09-01 00:00:00+00' + (i / 3) * interval '1 second',
       timestamptz '2026-09-01 00:00:00+00' + (i / 3) * interval '1 second'
FROM generate_series(0, 2) AS o, generate_series(1, 900) AS i`

// consolidationRecordsSQL gives one external experiment 50 members, each with
// one native collector record and 100 newer operator records of both kinds.
const consolidationRecordsSQL = `
INSERT INTO eval_experiments (experiment_id, owner_id, project_id, portable_id, control_mode, name, state,
    expected_count, max_in_flight, wall_ms)
VALUES ('exp-records', 'owner-0', 'project-0-0', 'portable-records', 'external', 'Records', 'ready', 50, 1, 1000);
INSERT INTO eval_frozen_plans (experiment_id, document_kind, document, setup, plan_sha256)
VALUES ('exp-records', 'ExternalRegistration', convert_to('{}', 'UTF8'), convert_to('{}', 'UTF8'), 'sha256:' || repeat('c', 64));
INSERT INTO eval_members (experiment_id, member_id, pair_id, ordinal, suite_id, case_id, sample, variant_id,
    case_sha256, binding_sha256, eligibility, execution_kind, recipe, submission_key)
SELECT 'exp-records', encode(sha256(convert_to('member-' || n, 'UTF8')), 'hex'),
       encode(sha256(convert_to('pair-' || n, 'UTF8')), 'hex'), n, 'suite', 'case-' || n, 1, 'a',
       'sha256:' || repeat('a', 64), 'sha256:' || repeat('b', 64), 'eligible', 'run', '{}', 'key-' || n
  FROM generate_series(0, 49) AS n;
INSERT INTO eval_records (experiment_id, member_id, kind, record_sha256, document, actor_id, created_at)
SELECT 'exp-records', m.member_id, kind, 'sha256:' || encode(sha256(record.document), 'hex'), record.document,
       CASE WHEN r = 0 THEN 'system:eval-collector' ELSE 'user:operator' END,
       timestamptz '2026-09-01 00:00:00+00' + r * interval '1 second'
  FROM eval_members AS m CROSS JOIN generate_series(0, 100) AS r CROSS JOIN (VALUES ('result'), ('assessment')) AS kinds(kind)
 CROSS JOIN LATERAL (SELECT convert_to('{"m":"' || m.member_id || '","r":' || r || ',"k":"' || kind || '"}', 'UTF8') AS document) AS record
 WHERE m.experiment_id = 'exp-records'`

// TestPostgresEvalReadersWithoutOwnerPageIndex reads the statements that
// eval_experiments_owner_page_idx (owner_id, experiment_id) served before
// V409-007 dropped it, in the custom and the generic plan. Following every
// summary page, unfiltered and filtered by Project, state, dataset and
// control mode, returns the owner's matching experiments once each in
// (created_at, experiment_id) descending order across created_at ties. A
// statement that selects one experiment by owner and ID reads it through
// the primary key: the native-collector record read reads one experiment and
// one record.
func TestPostgresEvalReadersWithoutOwnerPageIndex(t *testing.T) {
	pool := testPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	for _, statement := range []string{
		`INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest)
SELECT 'project-' || o || '-' || p, 'owner-' || o, 'evaluation', 'Eval', 'create-' || p, 'sha256:' || repeat('a', 64)
  FROM generate_series(0, 2) AS o, generate_series(0, 3) AS p`,
		`INSERT INTO eval_dataset_revisions (owner_id, project_id, dataset_id, revision, document, metadata)
SELECT 'owner-' || o, 'project-' || o || '-' || p, 'dataset-' || d, '1', convert_to('{}', 'UTF8'), '{}'::jsonb
  FROM generate_series(0, 2) AS o, generate_series(0, 3) AS p, generate_series(0, 19) AS d`,
		consolidationExperimentsSQL,
	} {
		if _, err := pool.Exec(ctx, statement); err != nil {
			t.Fatal(err)
		}
	}
	if err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, consolidationRecordsSQL)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `VACUUM ANALYZE eval_experiments, eval_records, projects`); err != nil {
		t.Fatal(err)
	}

	const pageSize = 50
	for _, mode := range explaintest.Modes {
		config := pool.Config()
		config.ConnConfig.RuntimeParams["plan_cache_mode"] = string(mode)
		reader, err := pgxpool.NewWithConfig(ctx, config)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(reader.Close)
		store := NewPostgresStore(reader)
		for _, filter := range []ListParams{
			{OwnerID: "owner-1"},
			{OwnerID: "owner-1", ProjectID: "project-1-2"},
			{OwnerID: "owner-1", State: "paused"},
			{OwnerID: "owner-1", State: "finished"},
			{OwnerID: "owner-1", DatasetID: "dataset-3"},
			{OwnerID: "owner-1", ControlMode: "external"},
		} {
			var want []string
			rows, err := pool.Query(ctx, `SELECT experiment_id FROM eval_experiments
 WHERE owner_id = $1 AND ($2 = '' OR project_id = $2) AND ($3 = '' OR state = $3)
   AND ($4 = '' OR dataset_id = $4) AND ($5 = '' OR control_mode = $5)
 ORDER BY created_at DESC, experiment_id DESC`, filter.OwnerID, filter.ProjectID, filter.State, filter.DatasetID, filter.ControlMode)
			if err == nil {
				want, err = pgx.CollectRows(rows, pgx.RowTo[string])
			}
			if err != nil || len(want) == 0 {
				t.Fatalf("%+v: independent read = %d, %v", filter, len(want), err)
			}
			var got []string
			params := SummaryPageParams{ListParams: filter}
			params.Limit = pageSize
			for pages := 0; ; pages++ {
				if pages > len(want)/pageSize+1 {
					t.Fatalf("%s %+v: paging does not end", mode, filter)
				}
				page, err := store.SummaryPage(ctx, params)
				if err != nil {
					t.Fatalf("%s %+v: %v", mode, filter, err)
				}
				for _, item := range page.Items {
					got = append(got, item.ID)
				}
				if !page.HasMore {
					break
				}
				last := page.LastCreatedAt
				params.AfterCreatedAt, params.AfterID = &last, page.Items[len(page.Items)-1].ID
			}
			if !slices.Equal(got, want) {
				t.Fatalf("%s %+v: pages returned %d experiments, want %d in order", mode, filter, len(got), len(want))
			}
			// V414-026: each page reads experiments in proportion to the
			// page, in generic plans too, at the first page and at a deep
			// cursor. The statement with optional predicates read every
			// experiment of the owner in a generic plan.
			const budgetLimit = 25
			var deep time.Time
			if err := pool.QueryRow(ctx, `SELECT created_at FROM eval_experiments WHERE experiment_id = $1`, want[len(want)/2]).Scan(&deep); err != nil {
				t.Fatal(err)
			}
			for _, after := range []*time.Time{nil, &deep} {
				page := SummaryPageParams{ListParams: filter}
				page.Limit = budgetLimit
				if after != nil {
					page.AfterCreatedAt, page.AfterID = after, want[len(want)/2]
				}
				query, args := summaryPageQuery(page)
				analysis, err := explaintest.ExplainPrepared(ctx, pool, mode, query, args...)
				if err != nil {
					t.Fatal(err)
				}
				if reads := analysis.Reads.Of("eval_experiments"); reads > 2*(budgetLimit+1) {
					t.Errorf("%s %+v after %v: the summary page read %d experiments for a page of %d:\n%s", mode, filter, after != nil, reads, budgetLimit+1, analysis.Plan)
				}
			}
		}

		var member string
		if err := pool.QueryRow(ctx, `SELECT member_id FROM eval_members WHERE experiment_id = 'exp-records' AND ordinal = 7`).Scan(&member); err != nil {
			t.Fatal(err)
		}
		analysis, err := explaintest.ExplainPrepared(ctx, pool, mode, latestNativeRecordSQL, "owner-0", "exp-records", member, "result")
		if err != nil {
			t.Fatal(err)
		}
		experiments, records := analysis.Reads.Of("eval_experiments"), analysis.Reads.Of("eval_records")
		if experiments > 1 || records > 1 {
			t.Fatalf("%s: the native record read read %d experiments and %d records:\n%s", mode, experiments, records, analysis.Plan)
		}
		t.Logf("%s: the native record read read %d experiment and %d record", mode, experiments, records)
	}
}
