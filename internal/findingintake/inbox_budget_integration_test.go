//go:build integration

package findingintake

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	inboxOwner, inboxForeignOwner     = "inbox-owner", "inbox-foreign-owner"
	inboxProject, inboxForeignProject = "inbox-project", "inbox-foreign-project"
	inboxAudit, inboxSiblingAudit     = "inbox-audit", "inbox-sibling-audit"
	// inboxFetch is the fetch of a default public page: 20 rows and the
	// look-ahead row.
	inboxFetch = 21
	// inboxPlanningRows allows for the few index endpoints a custom plan's
	// estimates may fetch per relation (get_actual_variable_range).
	inboxPlanningRows = 8
)

var inboxEpoch = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

// originalAuditInboxSQL is listAuditInboxSQL as V25-010 wrote it: one scan of
// the owner's receipts with native and held membership under OR. It is the
// recorded case the row budget must reject and the oracle of inbox order.
const originalAuditInboxSQL = `
SELECT ` + receiptProjection + `
  FROM finding_proposal_receipts AS receipt
 WHERE receipt.owner_id = $1
   AND ((receipt.audit_id = $2 AND receipt.source_run_deleted_at IS NULL) OR EXISTS (
       SELECT 1 FROM finding_proposal_audit_holds AS hold
        JOIN audits AS audit
          ON audit.audit_id = hold.audit_id AND audit.project_id = hold.project_id
       WHERE hold.receipt_id = receipt.receipt_id
         AND hold.audit_id = $2 AND audit.owner_id = $1
   ))
   AND ($3::timestamptz IS NULL OR (receipt.created_at, receipt.receipt_id) > ($3, $4))
 ORDER BY receipt.created_at, receipt.receipt_id
 LIMIT $5`

// originalAuditHeldInboxSQL is listAuditHeldInboxSQL before V409-002, the
// oracle of held-only order.
const originalAuditHeldInboxSQL = `
SELECT ` + receiptProjection + `
  FROM finding_proposal_receipts AS receipt
  JOIN finding_proposal_audit_holds AS hold
    ON hold.receipt_id = receipt.receipt_id AND hold.audit_id = $2
  JOIN audits AS audit
    ON audit.audit_id = hold.audit_id AND audit.project_id = hold.project_id
 WHERE receipt.owner_id = $1 AND audit.owner_id = $1
   AND ($3::timestamptz IS NULL OR (receipt.created_at, receipt.receipt_id) > ($3, $4))
 ORDER BY receipt.created_at, receipt.receipt_id
 LIMIT $5`

// syntheticReceipts is one group of receipts seeded by SQL. They have no
// proposal bytes, so only statements that do not hydrate may read them.
type syntheticReceipts struct {
	prefix, owner, project, run string
	audit                       string
	count                       int
	// The n-th receipt (1-based) is created at inboxEpoch + offset + n*step.
	offset, step  time.Duration
	sourceDeleted bool
	auditHeld     bool
}

func seedSyntheticReceipts(t *testing.T, ctx context.Context, pool *pgxpool.Pool, group syntheticReceipts) {
	t.Helper()
	var audit *string
	if group.audit != "" {
		audit = &group.audit
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id,
    runtime_instance_id, stage_execution_id, logical_agent_name,
    invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id, audit_execution_id, audit_id, audit_role,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type,
    proposal_size_bytes, evidence, report_projection,
    created_at, retention_state, source_run_deleted_at, discarded_at, checks_projection
)
SELECT $1 || '-' || n, $1 || '-proposal-' || n, $1 || '-allocation', 'runtime',
       'instance', 'stage', 'worker', 'invocation', $1 || '-' || n,
       'key-' || n, 'sha256:' || repeat('a', 64), $2, $3, $4,
       CASE WHEN $5::text IS NOT NULL THEN $5 || '-execution' END, $5::text,
       CASE WHEN $5::text IS NOT NULL THEN 'discovery' END,
       'synthetic', '1', 'contractor/v1alpha1', '{"name":"synthetic","version":"1"}'::jsonb,
       'sha256:' || repeat('b', 64),
       jsonb_build_object('namespace', 'finding-proposals', 'name', $1 || '-proposal-' || n, 'revision', 'r1'),
       'sha256:' || repeat('c', 64), 'application/json', 1, '[]'::jsonb,
       CASE WHEN $8 AND NOT $10 THEN '{}' ELSE '{"description":"Test finding.","limitations":[],"preconditions":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"Test finding."}' END,
       $6::timestamptz + n * $7::interval,
       CASE WHEN $10 THEN 'audit-held' WHEN $8 THEN 'discarded' ELSE 'source-held' END,
       CASE WHEN $8 THEN $6::timestamptz END, CASE WHEN $8 AND NOT $10 THEN $6::timestamptz END,
       CASE WHEN $8 AND NOT $10 THEN contractor_empty_finding_checks('sha256:' || repeat('c',64),1) END
  FROM generate_series(1, $9::integer) AS n`,
		group.prefix, group.run, group.owner, group.project, audit,
		inboxEpoch.Add(group.offset), group.step, group.sourceDeleted, group.count, group.auditHeld,
	); err != nil {
		t.Fatalf("seed %s receipts: %v", group.prefix, err)
	}
}

// holdSyntheticReceipts makes Audit auditID hold the first count receipts of
// group prefix in Project projectID, as collection or an owner import does.
func holdSyntheticReceipts(t *testing.T, ctx context.Context, pool *pgxpool.Pool, prefix, auditID, projectID string, count int) {
	t.Helper()
	if _, err := pool.Exec(ctx, `
INSERT INTO finding_proposal_audit_holds (receipt_id, audit_id, project_id, proposal_ref, evidence)
SELECT $1 || '-' || n, $2, $3,
       jsonb_build_object('namespace', 'audit-copy', 'name', $1 || '-' || n, 'revision', 'r1'), '[]'::jsonb
  FROM generate_series(1, $4::integer) AS n`, prefix, auditID, projectID, count); err != nil {
		t.Fatalf("hold %s receipts in %s: %v", prefix, auditID, err)
	}
}

func createInboxProject(t *testing.T, ctx context.Context, pool *pgxpool.Pool, owner, projectID string) {
	t.Helper()
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: owner, Kind: projectstore.KindProject, Name: projectID,
		IdempotencyKey: projectID, RequestDigest: auditdomain.DigestBytes([]byte(projectID)),
	}); err != nil {
		t.Fatal(err)
	}
}

func createInboxAudit(t *testing.T, ctx context.Context, pool *pgxpool.Pool, owner, projectID, auditID string) {
	t.Helper()
	if _, _, err := auditstoretest.NewStore(pool).WithPurgeDependencies(DeleteAuditHoldsBatch).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: owner, ProjectID: projectID,
		Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: auditdomain.DigestBytes([]byte("profile"))},
		ProfileSnapshot: json.RawMessage(`{"interaction":{}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 100, MaxItemsTotal: 100,
			MaxSubmittedRuns: 100, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20},
		IdempotencyKey: auditID, RequestDigest: auditdomain.DigestBytes([]byte(auditID)),
	}); err != nil {
		t.Fatal(err)
	}
}

// seedInboxCandidates seeds the fixed inbox of inboxAudit: native receipts of
// a live and of a deleted source Run, some of them held, old receipts of an
// ordinary Run imported later, a sibling Audit's receipts imported into it,
// and the invisible kinds: holds in another Project, another owner's receipts
// naming the Audit and native receipts of a deleted Run that were never held.
func seedInboxCandidates(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	for _, group := range []syntheticReceipts{
		{prefix: "native", owner: inboxOwner, project: inboxProject, run: "x-live", audit: inboxAudit, count: 150, step: 61 * time.Second},
		{prefix: "native-held", owner: inboxOwner, project: inboxProject, run: "x-live", audit: inboxAudit, count: 60, offset: 7 * time.Second, step: 149 * time.Second},
		{prefix: "native-gone", owner: inboxOwner, project: inboxProject, run: "x-gone", audit: inboxAudit, count: 20, offset: 11 * time.Second, step: 401 * time.Second, sourceDeleted: true},
		{prefix: "native-gone-held", owner: inboxOwner, project: inboxProject, run: "x-gone", audit: inboxAudit, count: 20, offset: 13 * time.Second, step: 433 * time.Second, sourceDeleted: true, auditHeld: true},
		{prefix: "imported", owner: inboxOwner, project: inboxProject, run: "plain", count: 40, offset: -time.Hour, step: 223 * time.Second},
		{prefix: "sibling-imported", owner: inboxOwner, project: inboxProject, run: "y-run", audit: inboxSiblingAudit, count: 10, offset: 17 * time.Second, step: 811 * time.Second},
		{prefix: "foreign-project-held", owner: inboxOwner, project: inboxProject, run: "plain", count: 5, offset: 19 * time.Second, step: 997 * time.Second},
		{prefix: "intruder", owner: inboxForeignOwner, project: inboxForeignProject, run: "b-run", audit: inboxAudit, count: 5, offset: 23 * time.Second, step: 1009 * time.Second},
	} {
		seedSyntheticReceipts(t, ctx, pool, group)
	}
	holdSyntheticReceipts(t, ctx, pool, "native-held", inboxAudit, inboxProject, 60)
	holdSyntheticReceipts(t, ctx, pool, "native-gone-held", inboxAudit, inboxProject, 20)
	holdSyntheticReceipts(t, ctx, pool, "imported", inboxAudit, inboxProject, 40)
	holdSyntheticReceipts(t, ctx, pool, "sibling-imported", inboxAudit, inboxProject, 10)
	holdSyntheticReceipts(t, ctx, pool, "foreign-project-held", inboxAudit, inboxForeignProject, 5)
}

// inboxStatement is one inbox statement under measurement, with the original
// statement whose pages it must reproduce. A recorded statement is an
// original itself.
type inboxStatement struct {
	name, sql, original string
	held, recorded      bool
}

// inboxReads is the receipt and hold rows one execution read, as explaintest
// counts the rows a statement reads.
type inboxReads struct{ receipts, holds int64 }

func inboxReadsOf(reads explaintest.Reads) inboxReads {
	return inboxReads{receipts: reads.Of("finding_proposal_receipts"), holds: reads.Of("finding_proposal_audit_holds")}
}

// explainInbox runs statement once with EXPLAIN ANALYZE as a prepared
// statement planned in mode and returns the rows it read and its plan.
func explainInbox(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, statement string, mode explaintest.Mode, args []any,
) (inboxReads, explaintest.Node) {
	t.Helper()
	analysis, err := explaintest.ExplainPrepared(ctx, pool, mode, statement, args...)
	if err != nil {
		t.Fatalf("%s: explain: %v", mode, err)
	}
	return inboxReadsOf(analysis.Reads), analysis.Plan
}

// serviceInbox runs statement executions times through pgx's default
// statement cache on one connection, as the service does, so PostgreSQL's
// plan cache may switch to a generic plan after five custom ones. It returns
// the most rows any execution read and the receipt IDs of the last.
func serviceInbox(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, statement string, args []any, executions int,
) (inboxReads, []string) {
	t.Helper()
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(context.Background())
	var most inboxReads
	var receipts []Receipt
	for range executions {
		counted, err := explaintest.CountReads(ctx, tx, func() error {
			rows, err := tx.Query(ctx, statement, args...)
			if err != nil {
				return err
			}
			receipts, err = scanReceiptRows(rows)
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
		reads := inboxReadsOf(counted)
		most.receipts = max(most.receipts, reads.receipts)
		most.holds = max(most.holds, reads.holds)
	}
	return most, receiptIDs(receipts)
}

func inboxIDs(t *testing.T, ctx context.Context, db querier, statement string, args ...any) []string {
	t.Helper()
	rows, err := db.Query(ctx, statement, args...)
	if err != nil {
		t.Fatal(err)
	}
	receipts, err := scanReceiptRows(rows)
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]string, len(receipts))
	for index, receipt := range receipts {
		ids[index] = receipt.ReceiptID
	}
	return ids
}

// The inbox reads the Audit's own receipts and the receipts it holds, never
// the rest of the receipt relation. With a fixed inbox and a growing
// population of other owners', other Audits' and ordinary Runs' receipts, the
// receipt rows an inbox page reads, as explaintest counts them, stay within a
// budget derived from the Audit's native receipts N, its holds H and the
// fetch limit L, under custom, generic and the service's cached-statement
// plans; the recorded original statement exceeds the same budget once the
// unrelated population is large.
//
// With holds of two Audits only, a generic plan estimates that each Audit
// holds half of them; the receipt budget must still hold. Hold rows stay
// within H once the hold statistics describe many Audits; with two, the
// planner may read the small hold relation whole, as the original did.
func TestPostgresAuditInboxRowBudgetIgnoresUnrelatedReceipts(t *testing.T) {
	for _, distribution := range []struct {
		name         string
		spreadAudits int
	}{{"two Audits", 0}, {"many Audits", 30}} {
		t.Run(distribution.name, func(t *testing.T) {
			testAuditInboxRowBudget(t, distribution.spreadAudits)
		})
	}
}

func testAuditInboxRowBudget(t *testing.T, spreadAudits int) {
	ctx, cancel := context.WithTimeout(t.Context(), 4*time.Minute)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	createInboxProject(t, ctx, pool, inboxOwner, inboxProject)
	createInboxProject(t, ctx, pool, inboxForeignOwner, inboxForeignProject)
	createInboxAudit(t, ctx, pool, inboxOwner, inboxProject, inboxAudit)
	createInboxAudit(t, ctx, pool, inboxOwner, inboxProject, inboxSiblingAudit)
	seedInboxCandidates(t, ctx, pool)
	// Other Audits of both owners each originate and hold receipts.
	for index := range spreadAudits {
		owner, project := inboxOwner, inboxProject
		if index%2 == 1 {
			owner, project = inboxForeignOwner, inboxForeignProject
		}
		auditID := fmt.Sprintf("spread-audit-%02d", index)
		createInboxAudit(t, ctx, pool, owner, project, auditID)
		seedSyntheticReceipts(t, ctx, pool, syntheticReceipts{
			prefix: auditID, owner: owner, project: project, run: auditID + "-run", audit: auditID,
			count: 100, offset: time.Duration(index) * time.Second, step: 89 * time.Second,
		})
		holdSyntheticReceipts(t, ctx, pool, auditID, auditID, project, 100)
	}
	unrelated := []syntheticReceipts{
		{prefix: "sibling", owner: inboxOwner, project: inboxProject, run: "y-run", audit: inboxSiblingAudit, offset: 3 * time.Second},
		{prefix: "plain", owner: inboxOwner, project: inboxProject, run: "plain", offset: 5 * time.Second},
		{prefix: "foreign", owner: inboxForeignOwner, project: inboxForeignProject, run: "b-run", offset: time.Second},
	}
	statements := []inboxStatement{
		{name: "inbox", sql: listAuditInboxSQL, original: originalAuditInboxSQL},
		{name: "original inbox", sql: originalAuditInboxSQL, original: originalAuditInboxSQL, recorded: true},
		{name: "held inbox", sql: listAuditHeldInboxSQL, original: originalAuditHeldInboxSQL, held: true},
		{name: "original held inbox", sql: originalAuditHeldInboxSQL, original: originalAuditHeldInboxSQL, held: true, recorded: true},
	}
	for stage, perGroup := range []int{300, 6300} {
		for index := range unrelated {
			group := unrelated[index]
			group.prefix = fmt.Sprintf("%s-%d", group.prefix, stage)
			group.count = perGroup
			// Spread each group over the inbox's time range, about 2.5 hours.
			group.step = 9 * time.Second * 1000 / time.Duration(perGroup)
			seedSyntheticReceipts(t, ctx, pool, group)
		}
		// The sibling Audit holds a tenth, then half of its receipts.
		holdSyntheticReceipts(t, ctx, pool, fmt.Sprintf("sibling-%d", stage), inboxSiblingAudit, inboxProject, perGroup/(10-8*stage))
		if _, err := pool.Exec(ctx, `ANALYZE finding_proposal_receipts; ANALYZE finding_proposal_audit_holds; ANALYZE audits`); err != nil {
			t.Fatal(err)
		}
		var total, native, holds int64
		if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM finding_proposal_receipts),
       (SELECT count(*) FROM finding_proposal_receipts WHERE audit_id = $1),
       (SELECT count(*) FROM finding_proposal_audit_holds WHERE audit_id = $1)`, inboxAudit,
		).Scan(&total, &native, &holds); err != nil {
			t.Fatal(err)
		}
		if stage == 1 && total < 20001 {
			t.Fatalf("large population has %d receipts, want at least 20001", total)
		}
		inbox := inboxIDs(t, ctx, pool, originalAuditInboxSQL, inboxOwner, inboxAudit, nil, "", 1000)
		held := inboxIDs(t, ctx, pool, originalAuditHeldInboxSQL, inboxOwner, inboxAudit, nil, "", 1000)
		if len(inbox) != 280 || len(held) != 130 {
			t.Fatalf("fixture inbox has %d receipts and %d held, want 280 and 130", len(inbox), len(held))
		}
		// The native branch reads at most the Audit's N native receipts, the
		// held branch probes at most one receipt per hold, and the page reads
		// L whole. This budget bounds the population's reads by the inbox;
		// TestPostgresAuditHeldInboxPageReadsOnlyThePage bounds each branch
		// by the page.
		// Holding a receipt rewrites its retention state, so a held receipt
		// has two row versions, and until VACUUM can remove the old one an
		// index may return an entry for each: every read that can reach a
		// held receipt counts twice, at most H of the native branch's reads,
		// all H of the held branch's and all L of the page's. Holds are never
		// rewritten.
		budget := inboxReads{
			receipts: native + holds + 2*holds + 2*inboxFetch + inboxPlanningRows,
			holds:    holds + inboxPlanningRows,
		}
		heldBudget := inboxReads{receipts: 2*holds + 2*inboxFetch + inboxPlanningRows, holds: holds + inboxPlanningRows}
		t.Logf("population %d receipts: N=%d H=%d L=%d; inbox budget %d receipt rows, held budget %d",
			total, native, holds, inboxFetch, budget.receipts, heldBudget.receipts)
		for _, statement := range statements {
			ids, statementBudget := inbox, budget
			if statement.held {
				ids, statementBudget = held, heldBudget
			}
			// The late cursor leaves more than one page after it.
			late := ids[len(ids)-inboxFetch-9]
			var lateAt time.Time
			if err := pool.QueryRow(ctx, `SELECT created_at FROM finding_proposal_receipts WHERE receipt_id = $1`, late).Scan(&lateAt); err != nil {
				t.Fatal(err)
			}
			for _, cursor := range []struct {
				name string
				args []any
			}{
				{"early", []any{inboxOwner, inboxAudit, nil, "", inboxFetch}},
				{"late", []any{inboxOwner, inboxAudit, lateAt, late, inboxFetch}},
			} {
				readings := map[string]inboxReads{}
				for _, mode := range explaintest.Modes {
					reads, plan := explainInbox(t, ctx, pool, statement.sql, mode, cursor.args)
					readings[string(mode)] = reads
					t.Logf("%s %s %s: %d receipt rows, %d hold rows:\n%s",
						statement.name, cursor.name, mode, reads.receipts, reads.holds, plan)
				}
				service, got := serviceInbox(t, ctx, pool, statement.sql, cursor.args, 8)
				readings["service"] = service
				t.Logf("%s %s service: at most %d receipt rows, %d hold rows",
					statement.name, cursor.name, service.receipts, service.holds)
				if want := inboxIDs(t, ctx, pool, statement.original, cursor.args...); !slices.Equal(got, want) {
					t.Fatalf("%s %s page = %v, want %v", statement.name, cursor.name, got, want)
				}
				for mode, reads := range readings {
					within := reads.receipts <= statementBudget.receipts
					if statement.recorded {
						if stage == 1 && !statement.held && within {
							t.Errorf("original inbox %s %s visited %d receipt rows within budget %d; the recorded case no longer fails",
								cursor.name, mode, reads.receipts, statementBudget.receipts)
						}
						continue
					}
					if !within || spreadAudits > 0 && reads.holds > statementBudget.holds {
						t.Errorf("%s %s %s visited %d receipt and %d hold rows, budget %d and %d",
							statement.name, cursor.name, mode, reads.receipts, reads.holds,
							statementBudget.receipts, statementBudget.holds)
					}
				}
			}
		}
	}
}

// inboxPage reads one page of statement after the receipt after, from the
// start when after is nil.
func inboxPage(
	t *testing.T, ctx context.Context, db querier, statement, owner, auditID string, after *Receipt, limit int,
) []Receipt {
	t.Helper()
	var afterAt *time.Time
	afterID := ""
	if after != nil {
		afterAt, afterID = &after.CreatedAt, after.ReceiptID
	}
	rows, err := db.Query(ctx, statement, owner, auditID, afterAt, afterID, limit)
	if err != nil {
		t.Fatal(err)
	}
	receipts, err := scanReceiptRows(rows)
	if err != nil {
		t.Fatal(err)
	}
	return receipts
}

func receiptIDs(receipts []Receipt) []string {
	ids := make([]string, len(receipts))
	for index, receipt := range receipts {
		ids[index] = receipt.ReceiptID
	}
	return ids
}

// For every page size, paging the inbox and the held-only inbox from the
// start yields exactly the pages of the original statements: the same
// receipts in the same order, a receipt that is both native and held once,
// receipts sharing one creation time across both branches split at any page
// boundary without loss, old receipts imported later in receipt-time order,
// and an empty page after the last.
func TestPostgresAuditInboxPagesMatchOriginalStatements(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	createInboxProject(t, ctx, pool, inboxOwner, inboxProject)
	createInboxProject(t, ctx, pool, inboxForeignOwner, inboxForeignProject)
	createInboxAudit(t, ctx, pool, inboxOwner, inboxProject, inboxAudit)
	createInboxAudit(t, ctx, pool, inboxOwner, inboxProject, inboxSiblingAudit)
	seedInboxCandidates(t, ctx, pool)
	// Receipts of every kind created at one instant inside the inbox range.
	const tie = 30 * time.Minute
	for _, group := range []syntheticReceipts{
		{prefix: "tie-native", owner: inboxOwner, project: inboxProject, run: "x-live", audit: inboxAudit, count: 3, offset: tie},
		{prefix: "tie-overlap", owner: inboxOwner, project: inboxProject, run: "x-live", audit: inboxAudit, count: 3, offset: tie},
		{prefix: "tie-imported", owner: inboxOwner, project: inboxProject, run: "plain", count: 3, offset: tie},
		{prefix: "tie-gone-held", owner: inboxOwner, project: inboxProject, run: "x-gone", audit: inboxAudit, count: 2, offset: tie, sourceDeleted: true, auditHeld: true},
		{prefix: "tie-gone", owner: inboxOwner, project: inboxProject, run: "x-gone", audit: inboxAudit, count: 2, offset: tie, sourceDeleted: true},
		{prefix: "tie-sibling", owner: inboxOwner, project: inboxProject, run: "y-run", audit: inboxSiblingAudit, count: 2, offset: tie},
	} {
		seedSyntheticReceipts(t, ctx, pool, group)
	}
	holdSyntheticReceipts(t, ctx, pool, "tie-overlap", inboxAudit, inboxProject, 3)
	holdSyntheticReceipts(t, ctx, pool, "tie-imported", inboxAudit, inboxProject, 3)
	holdSyntheticReceipts(t, ctx, pool, "tie-gone-held", inboxAudit, inboxProject, 2)
	holdSyntheticReceipts(t, ctx, pool, "tie-sibling", inboxSiblingAudit, inboxProject, 2)
	for _, pair := range []struct {
		name, statement, original string
		want                      int
	}{
		{"inbox", listAuditInboxSQL, originalAuditInboxSQL, 280 + 3 + 3 + 3 + 2},
		{"held inbox", listAuditHeldInboxSQL, originalAuditHeldInboxSQL, 130 + 3 + 3 + 2},
	} {
		all := receiptIDs(inboxPage(t, ctx, pool, pair.original, inboxOwner, inboxAudit, nil, 1000))
		if len(all) != pair.want {
			t.Fatalf("%s fixture has %d receipts, want %d", pair.name, len(all), pair.want)
		}
		for _, limit := range []int{1, 2, 3, 7, 21, 201} {
			var listed []string
			var after *Receipt
			for pages := 0; ; pages++ {
				if pages > len(all)+1 {
					t.Fatalf("%s at limit %d does not end", pair.name, limit)
				}
				page := inboxPage(t, ctx, pool, pair.statement, inboxOwner, inboxAudit, after, limit)
				want := inboxPage(t, ctx, pool, pair.original, inboxOwner, inboxAudit, after, limit)
				if !slices.Equal(receiptIDs(page), receiptIDs(want)) {
					t.Fatalf("%s at limit %d after %d receipts = %v, want %v",
						pair.name, limit, len(listed), receiptIDs(page), receiptIDs(want))
				}
				if len(page) == 0 {
					break
				}
				listed = append(listed, receiptIDs(page)...)
				after = &page[len(page)-1]
			}
			if !slices.Equal(listed, all) {
				t.Fatalf("%s at limit %d listed %v, want %v", pair.name, limit, listed, all)
			}
		}
	}
	// The statements check receipt ownership, not Audit ownership, for native
	// receipts; the public handler reads the owner's Audit first. Another
	// owner's receipts naming the Audit therefore stay theirs alone, held
	// receipts need the Audit's owner, and an unknown Audit lists nothing.
	for _, read := range []struct {
		owner, auditID string
		inbox, held    []string
	}{
		{inboxForeignOwner, inboxAudit, []string{"intruder-1", "intruder-2", "intruder-3", "intruder-4", "intruder-5"}, nil},
		{inboxOwner, "missing-audit", nil, nil},
	} {
		for _, pair := range []struct {
			statement, original string
			want                []string
		}{
			{listAuditInboxSQL, originalAuditInboxSQL, read.inbox},
			{listAuditHeldInboxSQL, originalAuditHeldInboxSQL, read.held},
		} {
			page := receiptIDs(inboxPage(t, ctx, pool, pair.statement, read.owner, read.auditID, nil, 201))
			original := receiptIDs(inboxPage(t, ctx, pool, pair.original, read.owner, read.auditID, nil, 201))
			if !slices.Equal(page, original) || len(page) != len(pair.want) {
				t.Fatalf("inbox of %s read by %s = %v, original %v, want %v",
					read.auditID, read.owner, page, original, pair.want)
			}
		}
	}
}
