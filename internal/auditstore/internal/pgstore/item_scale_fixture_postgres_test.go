package pgstore

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	// itemScaleRounds and itemScaleRoundItems fill one Audit to its limits:
	// LIM-AUD-25 (100,000 items) in LIM-AUD-26 rounds of at most 10,000.
	itemScaleRounds     = 32
	itemScaleRoundItems = 3_125
	itemScaleItems      = itemScaleRounds * itemScaleRoundItems
	// itemScaleAwaitingEvery spaces the current round's items that await
	// review, the inbox's sparse state.
	itemScaleAwaitingEvery = 500
	// planningEntries bounds the index entries the planner reads to find a
	// column's current extremes, in custom and generic plans alike, when a
	// comparison value, such as the other side's extreme in a merge join's
	// estimate, falls in a histogram end bin (see explaintest). ANALYZE
	// samples the large tables, so whether it does varies between runs (the
	// reconcile window's planning read 0 or 2 items); in a scale fixture's
	// own vacuumed database each probe reads one live entry. Row budgets add
	// it.
	planningEntries = 8
)

// itemScaleFixture is an Audit at the item limits: Round 1, the current
// round, materialized through the store with ready items and a sparse set
// awaiting review; Rounds 2 to 32 copied from it in SQL as closed rounds of
// settled items. Every round repeats the same subject keys, so a subject
// matches one item per round. A second, small Audit of the same owner and an
// Audit of another owner keep the reads honest about their Audit and owner.
// The fixture has a database of its own (budgetAuditPool) and is vacuumed,
// so its reads visit no superseded version of what the store wrote.
type itemScaleFixture struct {
	ctx      context.Context
	pool     *pgxpool.Pool
	store    *testStore
	owner    string
	auditID  string
	smallID  string
	foreign  string
	roundIDs []string
}

func newItemScaleFixture(t *testing.T) itemScaleFixture {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	t.Cleanup(cancel)
	pool := budgetAuditPool(t, ctx, databaseURL)
	f := itemScaleFixture{ctx: ctx, pool: pool, store: newTestStore(pool), owner: "owner-item-scale"}
	projects := projectstore.NewPostgresStore(pool)
	for _, owner := range []string{f.owner, "owner-item-scale-foreign"} {
		if _, _, err := projects.Create(ctx, projectstore.CreateParams{
			ProjectID: "project-" + owner, OwnerID: owner, Kind: projectstore.KindProject,
			Name: "Item scale", IdempotencyKey: "project-" + owner, RequestDigest: testDigest("1"),
		}); err != nil {
			t.Fatal(err)
		}
	}
	f.auditID = f.materialize(t, f.owner, "audit-item-scale", itemScaleRoundItems, itemScaleAwaitingEvery)
	f.roundIDs = []string{"round-audit-item-scale-01"}
	f.cloneSettledRounds(t, "audit-item-scale", itemScaleRounds)
	f.smallID = f.materialize(t, f.owner, "audit-item-small", 7, 3)
	f.cloneSettledRounds(t, "audit-item-small", 3)
	f.foreign = f.materialize(t, "owner-item-scale-foreign", "audit-item-foreign", 5, 2)
	if _, err := pool.Exec(ctx, `VACUUM (INDEX_CLEANUP ON, ANALYZE) audit_items, audit_rounds, audits`); err != nil {
		t.Fatal(err)
	}
	var total int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM audit_items WHERE audit_id = $1`, f.auditID).Scan(&total); err != nil {
		t.Fatal(err)
	}
	if total != itemScaleItems {
		t.Fatalf("scale Audit holds %d items, want %d", total, itemScaleItems)
	}
	return f
}

// materialize creates an Audit of owner with a first round of count ready
// items, every awaitingEvery-th of which awaits review instead.
func (f *itemScaleFixture) materialize(t *testing.T, owner, auditID string, count, awaitingEvery int) string {
	t.Helper()
	audit, _, err := f.store.CreateDraft(f.ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: owner, ProjectID: "project-" + owner,
		Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: json.RawMessage(`{"name":"profile"}`),
		InputSelection:  json.RawMessage(`{"inputs":{}}`),
		Limits: auditstore.Limits{
			MaxRounds: itemScaleRounds, BatchSize: 1, MaxItemsPerRound: 10_000, MaxItemsTotal: 100_000,
			MaxSubmittedRuns: 100_000, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
		},
		IdempotencyKey: auditID, RequestDigest: testDigest("3"),
	})
	if err != nil {
		t.Fatal(err)
	}
	items := make([]auditstore.MaterializedItem, count)
	for ordinal := range items {
		key := fmt.Sprintf("check-%05d", ordinal)
		items[ordinal] = auditstore.MaterializedItem{
			ItemID: fmt.Sprintf("item-%s-%05d", auditID, ordinal), ItemKey: key, Ordinal: ordinal,
			Kind: "checklist", SubjectKey: fmt.Sprintf("subject-%05d", ordinal),
			Task: testExact("audits", "task-"+key, "r1"), Origin: testOrigin(key), WorkflowRole: "check",
			InitialState: auditstore.ItemReady, Coverage: emptyCoverage(),
		}
		if ordinal%awaitingEvery == awaitingEvery-1 {
			items[ordinal].InitialState = auditstore.ItemAwaitingReview
			items[ordinal].ApprovalKind = auditstore.ItemApprovalActiveCheck
			items[ordinal].ApprovalDigest = testDigest("5")
		}
	}
	if _, _, err := f.store.MaterializeRound(f.ctx, auditstore.MaterializeRoundParams{
		OwnerID: owner, AuditID: auditID, ExpectedRevision: audit.Revision,
		RoundID: "round-" + auditID + "-01", RoundOrdinal: 1, Manifest: testExact("audits", "round-"+auditID, "r1"),
		BaselineSnapshot: json.RawMessage(`{"inputs":{},"skills":[]}`),
		DeadlineAt:       time.Now().Add(time.Hour), Items: items,
		IdempotencyKey: "start-" + auditID, RequestDigest: testDigest("4"),
	}); err != nil {
		t.Fatal(err)
	}
	return auditID
}

// cloneSettledRounds copies the first round of auditID into rounds 2 to
// rounds as closed rounds whose items are all settled.
func (f *itemScaleFixture) cloneSettledRounds(t *testing.T, auditID string, rounds int) {
	t.Helper()
	first := "round-" + auditID + "-01"
	roundID := `'round-` + auditID + `-' || lpad(n::text, 2, '0')`
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(f.ctx)) }()
	cloneWhere(t, f.ctx, tx, "audit_rounds", "round_id = $1", first, rounds-1, map[string]string{
		"round_id": roundID, "ordinal": "n", "state": "'closed'",
	})
	cloneWhere(t, f.ctx, tx, "audit_items", "round_id = $1", first, rounds-1, map[string]string{
		"item_id": `src.item_id || '-r' || lpad(n::text, 2, '0')`, "round_id": roundID,
		"state": "'settled'", "final_disposition": "'excluded'",
		"approval_kind": "'none'", "approval_subject_digest": "NULL",
	})
	if err := tx.Commit(f.ctx); err != nil {
		t.Fatal(err)
	}
	if auditID == f.auditID {
		for n := 2; n <= rounds; n++ {
			f.roundIDs = append(f.roundIDs, fmt.Sprintf("round-%s-%02d", auditID, n))
		}
	}
}

// cloneWhere inserts, for n from 2 to copies+1, a copy of every row of table
// that matches where (with $1 bound to key), overriding the columns of
// overrides with their expressions over src and n.
func cloneWhere(
	t *testing.T, ctx context.Context, tx pgx.Tx,
	table, where, key string, copies int, overrides map[string]string,
) {
	t.Helper()
	rows, err := tx.Query(ctx, `
SELECT attname::text FROM pg_attribute
 WHERE attrelid = $1::regclass AND attnum > 0 AND NOT attisdropped AND attgenerated = ''
 ORDER BY attnum`, table)
	if err != nil {
		t.Fatal(err)
	}
	columns, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, len(columns))
	values := make([]string, len(columns))
	used := 0
	for index, column := range columns {
		names[index] = pgx.Identifier{column}.Sanitize()
		values[index] = "src." + names[index]
		if expression, overridden := overrides[column]; overridden {
			values[index] = expression
			used++
		}
	}
	if used != len(overrides) {
		t.Fatalf("%s: overrides %v name a missing column", table, overrides)
	}
	identifier := pgx.Identifier{table}.Sanitize()
	if _, err := tx.Exec(ctx, `INSERT INTO `+identifier+` (`+strings.Join(names, ", ")+`)
SELECT `+strings.Join(values, ", ")+`
  FROM `+identifier+` AS src, generate_series(2, $2::integer + 1) AS n
 WHERE src.`+where, key, copies); err != nil {
		t.Fatalf("clone %s: %v", table, err)
	}
}

// explainPrepared analyzes sql with args prepared and planned in mode on a
// connection of pool, and fails t when the analysis fails.
func explainPrepared(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, mode explaintest.Mode, sql string, args ...any,
) explaintest.Analysis {
	t.Helper()
	analysis, err := explaintest.ExplainPrepared(ctx, pool, mode, sql, args...)
	if err != nil {
		t.Fatalf("%s: explain: %v\n%s", mode, err, sql)
	}
	return analysis
}
