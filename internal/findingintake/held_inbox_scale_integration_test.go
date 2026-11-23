//go:build integration

package findingintake

import (
	"context"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/jackc/pgx/v5/pgxpool"
)

// holdOrderedHeldInboxKeysSQL is auditHeldInboxKeysSQL as V414-019 found it:
// the hold index carried the hold's time, so the branch probed the receipt of
// every hold of the Audit and sorted them before its LIMIT. It is the
// recorded case the page budget must reject.
var holdOrderedHeldInboxKeysSQL = `
    SELECT receipt.receipt_id, receipt.created_at
      FROM finding_proposal_audit_holds AS hold
      JOIN audits AS audit
        ON audit.audit_id = hold.audit_id AND audit.project_id = hold.project_id
     CROSS JOIN LATERAL (
           SELECT candidate.receipt_id, candidate.created_at
             FROM finding_proposal_receipts AS candidate
            WHERE candidate.receipt_id = hold.receipt_id AND candidate.owner_id = $1
           OFFSET 0
       ) AS receipt
     WHERE hold.audit_id = $2 AND audit.owner_id = $1
       AND ` + auditInboxAfterCursorSQL("receipt.created_at", "receipt.receipt_id") + `
     ORDER BY receipt.created_at, receipt.receipt_id
     LIMIT $5`

// holdOrderedAuditInboxSQL and holdOrderedAuditHeldInboxSQL are
// listAuditInboxSQL and listAuditHeldInboxSQL over that held branch.
var (
	holdOrderedAuditInboxSQL = `
WITH page AS (
    (` + auditNativeInboxKeysSQL + `)
     UNION
    (` + holdOrderedHeldInboxKeysSQL + `)
     ORDER BY created_at, receipt_id
     LIMIT $5
)
SELECT ` + receiptProjection + `
  FROM page
  JOIN finding_proposal_receipts AS receipt ON receipt.receipt_id = page.receipt_id
 ORDER BY receipt.created_at, receipt.receipt_id`
	holdOrderedAuditHeldInboxSQL = `
WITH page AS (` + holdOrderedHeldInboxKeysSQL + `
)
SELECT ` + receiptProjection + `
  FROM page
  JOIN finding_proposal_receipts AS receipt ON receipt.receipt_id = page.receipt_id
 ORDER BY receipt.created_at, receipt.receipt_id`
)

const (
	// heldScaleImported old receipts of an ordinary Run are imported into the
	// measured Audit in the reverse of their receipt order, so hold time and
	// receipt time disagree.
	heldScaleImported = 20_000
	// heldScaleNative native receipts of the measured Audit's live child Run,
	// of which heldScaleOverlap are held too, and heldScaleGone held native
	// receipts of a deleted child Run.
	heldScaleNative, heldScaleOverlap, heldScaleGone = 300, 100, 50
	// heldScaleHolds are the measured Audit's holds.
	heldScaleHolds = heldScaleImported + heldScaleOverlap + heldScaleGone
	// heldScaleSpread other Audits of both owners hold heldScaleSpreadHolds
	// receipts each, and the sibling Audit heldScaleSibling.
	heldScaleSpread, heldScaleSpreadHolds, heldScaleSibling = 20, 300, 5_000
	// nextRoundScan is the next-Round builder's scan budget (LIM-AUX-9,
	// maxNextRoundInboxScan in internal/auditservice), which it takes in
	// pages of pagelimit.MaxRows and one look-ahead row.
	nextRoundScan = 10_000
)

// seedHeldScaleInbox seeds the measured Audit inboxAudit of inboxOwner with
// heldScaleHolds holds and heldScaleNative native receipts, other Audits'
// holds, and analyzes the tables. Holding a receipt rewrites its retention
// state, so the tables are vacuumed as well; a concurrent test's snapshot can
// still keep the old versions, which the budgets allow for.
func seedHeldScaleInbox(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	createInboxProject(t, ctx, pool, inboxOwner, inboxProject)
	createInboxProject(t, ctx, pool, inboxForeignOwner, inboxForeignProject)
	createInboxAudit(t, ctx, pool, inboxOwner, inboxProject, inboxAudit)
	createInboxAudit(t, ctx, pool, inboxOwner, inboxProject, inboxSiblingAudit)
	span := 5 * time.Hour
	for _, group := range []syntheticReceipts{
		// Created from the end of the span back to its start, held in that
		// order: the first hold is the latest receipt.
		{prefix: "imported", owner: inboxOwner, project: inboxProject, run: "plain", count: heldScaleImported,
			offset: span, step: -span / heldScaleImported},
		{prefix: "native-held", owner: inboxOwner, project: inboxProject, run: "x-live", audit: inboxAudit,
			count: heldScaleOverlap, offset: 7 * time.Second, step: span / heldScaleOverlap},
		{prefix: "native", owner: inboxOwner, project: inboxProject, run: "x-live", audit: inboxAudit,
			count: heldScaleNative - heldScaleOverlap, offset: 11 * time.Second, step: span / (heldScaleNative - heldScaleOverlap)},
		{prefix: "native-gone-held", owner: inboxOwner, project: inboxProject, run: "x-gone", audit: inboxAudit,
			count: heldScaleGone, offset: 13 * time.Second, step: span / heldScaleGone, sourceDeleted: true, auditHeld: true},
		{prefix: "sibling", owner: inboxOwner, project: inboxProject, run: "y-run", audit: inboxSiblingAudit,
			count: heldScaleSibling, offset: 17 * time.Second, step: span / heldScaleSibling},
	} {
		seedSyntheticReceipts(t, ctx, pool, group)
	}
	holdSyntheticReceiptsInBulk(t, ctx, pool, "imported", inboxAudit, inboxProject, heldScaleImported)
	holdSyntheticReceipts(t, ctx, pool, "native-held", inboxAudit, inboxProject, heldScaleOverlap)
	holdSyntheticReceipts(t, ctx, pool, "native-gone-held", inboxAudit, inboxProject, heldScaleGone)
	holdSyntheticReceiptsInBulk(t, ctx, pool, "sibling", inboxSiblingAudit, inboxProject, heldScaleSibling)
	for index := range heldScaleSpread {
		owner, project := inboxOwner, inboxProject
		if index%2 == 1 {
			owner, project = inboxForeignOwner, inboxForeignProject
		}
		auditID := fmt.Sprintf("spread-audit-%02d", index)
		createInboxAudit(t, ctx, pool, owner, project, auditID)
		seedSyntheticReceipts(t, ctx, pool, syntheticReceipts{
			prefix: auditID, owner: owner, project: project, run: auditID + "-run", audit: auditID,
			count: heldScaleSpreadHolds, offset: time.Duration(index) * time.Second, step: span / heldScaleSpreadHolds,
		})
		holdSyntheticReceipts(t, ctx, pool, auditID, auditID, project, heldScaleSpreadHolds)
	}
	if _, err := pool.Exec(ctx,
		`VACUUM (INDEX_CLEANUP ON, ANALYZE) finding_proposal_receipts, finding_proposal_audit_holds, audits`); err != nil {
		t.Fatal(err)
	}
	var holds int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM finding_proposal_audit_holds WHERE audit_id = $1`, inboxAudit).Scan(&holds); err != nil || holds != heldScaleHolds {
		t.Fatalf("measured Audit holds = (%d, %v), want %d", holds, err, heldScaleHolds)
	}
}

// holdSyntheticReceiptsInBulk is holdSyntheticReceipts for thousands of
// holds. Every hold a store writer inserts admits a finding and advances its
// Audit's revision and events, which a single statement of thousands of holds
// of one Audit makes quadratic, so the bulk copies leave the hold triggers
// off and write the receipt time the copy trigger would. The copies are
// read-model fixtures: the inbox reads neither findings nor retention states.
func holdSyntheticReceiptsInBulk(t *testing.T, ctx context.Context, pool *pgxpool.Pool, prefix, auditID, projectID string, count int) {
	t.Helper()
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	if _, err := tx.Exec(ctx, `ALTER TABLE finding_proposal_audit_holds DISABLE TRIGGER USER`); err != nil {
		t.Fatal(err)
	}
	tag, err := tx.Exec(ctx, `
INSERT INTO finding_proposal_audit_holds (receipt_id, audit_id, project_id, proposal_ref, evidence, receipt_created_at)
SELECT receipt.receipt_id, $2, $3,
       jsonb_build_object('namespace', 'audit-copy', 'name', receipt.receipt_id, 'revision', 'r1'), '[]'::jsonb,
       receipt.created_at
  FROM generate_series(1, $4::integer) AS n
  JOIN finding_proposal_receipts AS receipt ON receipt.receipt_id = $1 || '-' || n
 ORDER BY n`, prefix, auditID, projectID, count)
	if err != nil || tag.RowsAffected() != int64(count) {
		t.Fatalf("hold %s receipts in %s in bulk = (%v, %v), want %d rows", prefix, auditID, tag, err, count)
	}
	if _, err := tx.Exec(ctx, `ALTER TABLE finding_proposal_audit_holds ENABLE TRIGGER USER`); err != nil {
		t.Fatal(err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
}

// heldInboxPageBudget is what one page of limit rows may read: each branch
// reads its first limit entries after the cursor and the page reads its rows
// whole, every receipt read possibly twice while a held receipt's superseded
// retention version is not vacuumed yet, plus the planner's index endpoint
// reads. It depends on the page and nothing else.
func heldInboxPageBudget(limit int, combined bool) inboxReads {
	branches := int64(1)
	if combined {
		branches = 2
	}
	return inboxReads{
		receipts: 2*(branches+1)*int64(limit) + inboxPlanningRows,
		holds:    int64(limit) + inboxPlanningRows,
	}
}

// TestPostgresAuditHeldInboxPageReadsOnlyThePage reads the held inbox and the
// combined inbox of an Audit that holds 20,150 receipts, beside other Audits'
// holds, at the first, a middle and the last cursor, in pages of 21 and 200
// (V414-019). Under custom, generic and the service's cached-statement plans
// each page reads at most the budget of its page size, returns the pages of
// the original statements, and the statements that ordered holds by their
// own time read every hold of the Audit for each page.
func TestPostgresAuditHeldInboxPageReadsOnlyThePage(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 4*time.Minute)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	seedHeldScaleInbox(t, ctx, pool)
	statements := []struct {
		name, sql, original string
		combined, recorded  bool
	}{
		{name: "held inbox", sql: listAuditHeldInboxSQL, original: originalAuditHeldInboxSQL},
		{name: "inbox", sql: listAuditInboxSQL, original: originalAuditInboxSQL, combined: true},
		{name: "hold-ordered held inbox", sql: holdOrderedAuditHeldInboxSQL, original: originalAuditHeldInboxSQL, recorded: true},
		{name: "hold-ordered inbox", sql: holdOrderedAuditInboxSQL, original: originalAuditInboxSQL, combined: true, recorded: true},
	}
	for _, statement := range statements {
		all := inboxIDs(t, ctx, pool, statement.original, inboxOwner, inboxAudit, nil, "", 100_000)
		want := heldScaleHolds
		if statement.combined {
			want = heldScaleImported + heldScaleNative + heldScaleGone
		}
		if len(all) != want {
			t.Fatalf("%s fixture lists %d receipts, want %d", statement.name, len(all), want)
		}
		for _, limit := range []int{inboxFetch, pagelimit.MaxRows} {
			budget := heldInboxPageBudget(limit, statement.combined)
			for _, cursor := range []struct {
				name  string
				after int // index into all of the cursor receipt, -1 for none
			}{{"first", -1}, {"middle", len(all) / 2}, {"last", len(all) - limit/2}} {
				args := []any{inboxOwner, inboxAudit, nil, "", limit}
				if cursor.after >= 0 {
					var at time.Time
					if err := pool.QueryRow(ctx, `SELECT created_at FROM finding_proposal_receipts WHERE receipt_id = $1`,
						all[cursor.after]).Scan(&at); err != nil {
						t.Fatal(err)
					}
					args = []any{inboxOwner, inboxAudit, at, all[cursor.after], limit}
				}
				name := fmt.Sprintf("%s, page %d, %s", statement.name, limit, cursor.name)
				readings := map[string]inboxReads{}
				for _, mode := range explaintest.Modes {
					reads, plan := explainInbox(t, ctx, pool, statement.sql, mode, args)
					readings[string(mode)] = reads
					if statement.recorded || reads.receipts > budget.receipts || reads.holds > budget.holds {
						t.Logf("%s, %s: %d receipt rows, %d hold rows:\n%s", name, mode, reads.receipts, reads.holds, plan)
					}
				}
				service, got := serviceInbox(t, ctx, pool, statement.sql, args, 8)
				readings["service"] = service
				end := len(all)
				if cursor.after+1+limit < end {
					end = cursor.after + 1 + limit
				}
				if wantPage := all[cursor.after+1 : end]; !slices.Equal(got, wantPage) {
					t.Fatalf("%s page = %v, want %v", name, got, wantPage)
				}
				for mode, reads := range readings {
					t.Logf("%s, %s: %d receipt rows, %d hold rows (budget %d and %d)",
						name, mode, reads.receipts, reads.holds, budget.receipts, budget.holds)
					within := reads.receipts <= budget.receipts && reads.holds <= budget.holds
					switch {
					case statement.recorded && within:
						t.Errorf("%s, %s read %d receipt and %d hold rows, within the page budget; the recorded case no longer fails",
							name, mode, reads.receipts, reads.holds)
					case !statement.recorded && !within:
						t.Errorf("%s, %s read %d receipt and %d hold rows, budget %d and %d",
							name, mode, reads.receipts, reads.holds, budget.receipts, budget.holds)
					}
				}
			}
		}
	}
}

// TestPostgresNextRoundHeldInboxPassReadsGrowWithPages takes the next-Round
// builder's pass over the held inbox of an Audit that holds 20,150 receipts:
// pages of pagelimit.MaxRows through the service's cached statement until
// 10,000 receipts are scanned, then one look-ahead row, as
// selectNextProposalChecksFromInbox in internal/auditservice pages
// Service.ListAuditHeldInbox (V414-019). Every call reads at most the budget
// of its page, so the pass reads rows in proportion to the receipts it scans,
// and it lists the first 10,001 held receipts in the original order. The
// statement that ordered holds by their own time read every hold once per
// call. The builder hydrates and selects from the listed receipts without
// further inbox reads, so the pass is measured at the statement it calls.
func TestPostgresNextRoundHeldInboxPassReadsGrowWithPages(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 4*time.Minute)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	seedHeldScaleInbox(t, ctx, pool)
	want := inboxIDs(t, ctx, pool, originalAuditHeldInboxSQL, inboxOwner, inboxAudit, nil, "", nextRoundScan+1)
	calls := int64(nextRoundScan/pagelimit.MaxRows + 1)
	budget := inboxReads{holds: nextRoundScan + 1 + calls*inboxPlanningRows}
	for _, page := range []int{pagelimit.MaxRows, 1} {
		pageBudget := heldInboxPageBudget(page, false)
		if page == 1 {
			budget.receipts += pageBudget.receipts
		} else {
			budget.receipts += (calls - 1) * pageBudget.receipts
		}
	}
	for _, statement := range []struct {
		name, sql string
		recorded  bool
	}{
		{name: "held inbox", sql: listAuditHeldInboxSQL},
		{name: "hold-ordered held inbox", sql: holdOrderedAuditHeldInboxSQL, recorded: true},
	} {
		listed, made, total := nextRoundHeldPass(t, ctx, pool, statement.sql, !statement.recorded)
		if !slices.Equal(listed, want) {
			t.Fatalf("%s pass listed %d receipts that differ from the original's first %d", statement.name, len(listed), len(want))
		}
		t.Logf("%s pass of %d calls read %d receipt and %d hold rows (budget %d and %d); the Audit holds %d",
			statement.name, made, total.receipts, total.holds, budget.receipts, budget.holds, heldScaleHolds)
		within := total.receipts <= budget.receipts && total.holds <= budget.holds
		switch {
		case made != calls:
			t.Errorf("%s pass made %d calls, want %d", statement.name, made, calls)
		case statement.recorded && (within || total.holds < calls*heldScaleHolds):
			t.Errorf("%s pass read %d hold rows, want every hold of the Audit for each of its %d calls; the recorded case no longer fails",
				statement.name, total.holds, made)
		case !statement.recorded && !within:
			t.Errorf("%s pass read %d receipt and %d hold rows, budget %d and %d",
				statement.name, total.receipts, total.holds, budget.receipts, budget.holds)
		}
	}
}

// nextRoundHeldPass pages statement as the next-Round builder pages the held
// inbox, in one transaction through pgx's statement cache, and returns the
// receipts it listed, its calls and the rows they read. With checkPages, each
// call must stay within the budget of its page.
func nextRoundHeldPass(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, statement string, checkPages bool,
) ([]string, int64, inboxReads) {
	t.Helper()
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	var listed []string
	var total inboxReads
	var afterAt *time.Time
	afterID := ""
	var calls int64
	page := func(limit int) []Receipt {
		t.Helper()
		calls++
		var receipts []Receipt
		counted, err := explaintest.CountReads(ctx, tx, func() error {
			rows, err := tx.Query(ctx, statement, inboxOwner, inboxAudit, afterAt, afterID, limit)
			if err != nil {
				return err
			}
			receipts, err = scanReceiptRows(rows)
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
		reads, budget := inboxReadsOf(counted), heldInboxPageBudget(limit, false)
		if checkPages && (reads.receipts > budget.receipts || reads.holds > budget.holds) {
			t.Errorf("call %d of %d rows read %d receipt and %d hold rows, budget %d and %d",
				calls, limit, reads.receipts, reads.holds, budget.receipts, budget.holds)
		}
		total.receipts += reads.receipts
		total.holds += reads.holds
		return receipts
	}
	for len(listed) < nextRoundScan {
		limit := min(pagelimit.MaxRows, nextRoundScan-len(listed))
		receipts := page(limit)
		listed = append(listed, receiptIDs(receipts)...)
		if len(receipts) < limit {
			return listed, calls, total
		}
		last := receipts[len(receipts)-1]
		afterAt, afterID = &last.CreatedAt, last.ReceiptID
	}
	listed = append(listed, receiptIDs(page(1))...)
	return listed, calls, total
}
