package evalstore

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// consolidationExperimentsSQL loads 3,000 experiments for each of three owners
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
FROM generate_series(0, 2) AS o, generate_series(1, 3000) AS i`

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

// loadConsolidationFixture loads four evaluation Projects for each of three
// owners, 20 dataset revisions in each Project and consolidationExperimentsSQL,
// runs each statement in more, and analyzes the tables. The loads run without
// the test pool's statement timeout.
func loadConsolidationFixture(ctx context.Context, t *testing.T, pool *pgxpool.Pool, more ...func(pgx.Tx) error) {
	t.Helper()
	statements := []string{
		`INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest)
SELECT 'project-' || o || '-' || p, 'owner-' || o, 'evaluation', 'Eval', 'create-' || p, 'sha256:' || repeat('a', 64)
  FROM generate_series(0, 2) AS o, generate_series(0, 3) AS p`,
		`INSERT INTO eval_dataset_revisions (owner_id, project_id, dataset_id, revision, document, metadata)
SELECT 'owner-' || o, 'project-' || o || '-' || p, 'dataset-' || d, '1', convert_to('{}', 'UTF8'), '{}'::jsonb
  FROM generate_series(0, 2) AS o, generate_series(0, 3) AS p, generate_series(0, 19) AS d`,
		consolidationExperimentsSQL,
	}
	seedWithoutStatementTimeout(ctx, t, pool, func(tx pgx.Tx) error {
		for _, statement := range statements {
			if _, err := tx.Exec(ctx, statement); err != nil {
				return err
			}
		}
		return nil
	})
	for _, load := range more {
		seedWithoutStatementTimeout(ctx, t, pool, load)
	}
	if _, err := pool.Exec(ctx, `VACUUM ANALYZE eval_experiments, eval_records, eval_collections, projects`); err != nil {
		t.Fatal(err)
	}
}

// planPool is a pool of pool's database whose sessions plan prepared
// statements in mode, as a pooled statement cache may after five executions.
func planPool(ctx context.Context, t *testing.T, pool *pgxpool.Pool, mode explaintest.Mode) *pgxpool.Pool {
	t.Helper()
	config := pool.Config()
	config.ConnConfig.RuntimeParams["plan_cache_mode"] = string(mode)
	reader, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(reader.Close)
	return reader
}

// TestPostgresEvalReadersWithoutOwnerPageIndex reads a statement that
// eval_experiments_owner_page_idx (owner_id, experiment_id) served before
// V409-007 dropped it, in the custom and the generic plan: the
// native-collector record read selects one experiment by owner and ID through
// the primary key and reads one experiment and one record.
func TestPostgresEvalReadersWithoutOwnerPageIndex(t *testing.T) {
	pool := testPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	loadConsolidationFixture(ctx, t, pool, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, consolidationRecordsSQL)
		return err
	})
	var member string
	if err := pool.QueryRow(ctx, `SELECT member_id FROM eval_members WHERE experiment_id = 'exp-records' AND ordinal = 7`).Scan(&member); err != nil {
		t.Fatal(err)
	}
	for _, mode := range explaintest.Modes {
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

// summaryPageSelection is how summaryPageSQL ends its page selection: the
// predicates marker and LIMIT $3.
const summaryPageSelection = ` /* predicates */
    ORDER BY created_at DESC, experiment_id DESC LIMIT $3`

// optionalPageSelection is the page selection of the summary page statement
// before V414-026: optional Project, state, dataset, control-mode and cursor
// predicates in one statement, limited to $8.
const optionalPageSelection = `
        AND ($2 = '' OR project_id = $2)
        AND ($3 = '' OR state = $3)
        AND ($4 = '' OR dataset_id = $4)
        AND ($5 = '' OR control_mode = $5)
        AND ($6::timestamptz IS NULL OR (created_at, experiment_id) < ($6, $7))
    ORDER BY created_at DESC, experiment_id DESC LIMIT $8`

// optionalFilterSummaryPage returns the summary page statement before
// V414-026 and its arguments for p, with every removed filter unset.
func optionalFilterSummaryPage(t *testing.T, p SummaryPageParams) (string, []any) {
	t.Helper()
	if strings.Count(summaryPageSQL, summaryPageSelection) != 1 {
		t.Fatal("summaryPageSQL no longer ends its page selection with the predicates marker and LIMIT $3")
	}
	return strings.Replace(summaryPageSQL, summaryPageSelection, optionalPageSelection, 1),
		[]any{p.OwnerID, p.ProjectID, "", "", "", p.AfterCreatedAt, p.AfterID, p.Limit + 1}
}

// TestPostgresEvalSummaryPagesReadPageSizedRows reads the experiment list in
// its four statements (every Project of the owner or one, first page or after
// a deep cursor) in the custom and the generic plan (EVM-R247). Each page
// reads at most twice its rows of eval_experiments, not the owner's 3,000
// experiments, while the statement with optional predicates that it replaced
// reads hundreds at the deep cursor in the generic plan. Following every page
// returns the owner's experiments once each in (created_at, experiment_id)
// descending order across created_at ties, each page with the same revision
// and the same items as the replaced statement, and a stale revision still
// answers eval_view_changed.
func TestPostgresEvalSummaryPagesReadPageSizedRows(t *testing.T) {
	pool := testPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	loadConsolidationFixture(ctx, t, pool)

	const limit = 25
	const budget = 2 * (limit + 1)
	for _, mode := range explaintest.Modes {
		reader := planPool(ctx, t, pool, mode)
		store := NewPostgresStore(reader)
		for _, projectID := range []string{"", "project-1-2"} {
			rows, err := pool.Query(ctx, `SELECT experiment_id, created_at FROM eval_experiments
 WHERE owner_id = 'owner-1' AND ($1 = '' OR project_id = $1)
 ORDER BY created_at DESC, experiment_id DESC`, projectID)
			if err != nil {
				t.Fatal(err)
			}
			type position struct {
				ID        string
				CreatedAt time.Time
			}
			want, err := pgx.CollectRows(rows, pgx.RowToStructByPos[position])
			if err != nil || len(want) < 700 {
				t.Fatalf("%q: independent read = %d experiments, %v", projectID, len(want), err)
			}

			params := SummaryPageParams{OwnerID: "owner-1", ProjectID: projectID, Limit: limit}
			var got []position
			var revision int64
			for pages := 0; ; pages++ {
				if pages > len(want)/limit+1 {
					t.Fatalf("%s %q: paging does not end", mode, projectID)
				}
				query, args := summaryPageQuery(params)
				old, oldArgs := optionalFilterSummaryPage(t, params)
				var oldRevision int64
				var raw, oldRaw []byte
				if err := reader.QueryRow(ctx, query, args...).Scan(&revision, &raw); err != nil {
					t.Fatal(err)
				}
				if err := reader.QueryRow(ctx, old, oldArgs...).Scan(&oldRevision, &oldRaw); err != nil {
					t.Fatal(err)
				}
				if revision != oldRevision || !bytes.Equal(raw, oldRaw) {
					t.Fatalf("%s %q page %d: revision %d and %d bytes, the replaced statement %d and %d bytes", mode, projectID, pages, revision, len(raw), oldRevision, len(oldRaw))
				}
				page, err := store.SummaryPage(ctx, params)
				if err != nil {
					t.Fatalf("%s %q: %v", mode, projectID, err)
				}
				if page.Revision != revision {
					t.Fatalf("%s %q: page revision %d, statement revision %d", mode, projectID, page.Revision, revision)
				}
				for _, item := range page.Items {
					got = append(got, position{ID: item.ID})
				}
				if len(page.Items) > 0 {
					got[len(got)-1].CreatedAt = page.LastCreatedAt
				}
				if !page.HasMore {
					break
				}
				last := page.LastCreatedAt
				params.AfterCreatedAt, params.AfterID, params.Revision = &last, page.Items[len(page.Items)-1].ID, &page.Revision
			}
			if len(got) != len(want) {
				t.Fatalf("%s %q: pages returned %d experiments, want %d", mode, projectID, len(got), len(want))
			}
			for i := range want {
				if got[i].ID != want[i].ID || (!got[i].CreatedAt.IsZero() && !got[i].CreatedAt.Equal(want[i].CreatedAt)) {
					t.Fatalf("%s %q: position %d is %+v, want %+v", mode, projectID, i, got[i], want[i])
				}
			}
			stale := revision + 1
			if _, err := store.SummaryPage(ctx, SummaryPageParams{OwnerID: "owner-1", ProjectID: projectID, Limit: limit, Revision: &stale}); !code(err, "eval_view_changed") {
				t.Fatalf("%s %q: a stale revision answered %v", mode, projectID, err)
			}

			deep := want[len(want)/2]
			for _, after := range []*position{nil, &deep} {
				page := SummaryPageParams{OwnerID: "owner-1", ProjectID: projectID, Limit: limit}
				if after != nil {
					page.AfterCreatedAt, page.AfterID = &after.CreatedAt, after.ID
				}
				query, args := summaryPageQuery(page)
				analysis, err := explaintest.ExplainPrepared(ctx, pool, mode, query, args...)
				if err != nil {
					t.Fatal(err)
				}
				reads := analysis.Reads.Of("eval_experiments")
				if reads > budget {
					t.Errorf("%s %q cursor=%v: the page read %d experiments for %d rows:\n%s", mode, projectID, after != nil, reads, limit+1, analysis.Plan)
				}
				t.Logf("%s %q cursor=%v: the page read %d experiments", mode, projectID, after != nil, reads)
				if mode != explaintest.Generic || after == nil {
					continue
				}
				old, oldArgs := optionalFilterSummaryPage(t, page)
				analysis, err = explaintest.ExplainPrepared(ctx, pool, mode, old, oldArgs...)
				if err != nil {
					t.Fatal(err)
				}
				if reads = analysis.Reads.Of("eval_experiments"); reads <= budget {
					t.Errorf("%q: the replaced statement read only %d experiments at a deep cursor in the generic plan, so the budget proves nothing:\n%s", projectID, reads, analysis.Plan)
				}
				t.Logf("%s %q cursor=true: the replaced statement read %d experiments", mode, projectID, reads)
			}
		}
	}
}
