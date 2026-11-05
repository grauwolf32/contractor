package pgstore

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/disktest"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/jackc/pgx/v5"
)

// originalClaimAuditSQL is claimAuditSQL as V414-028 found it: it joined
// every claim row with its Audit before it filtered and ordered them, so a
// pass read the claim of every retained Audit. It is the reference for the
// candidate equivalence test and the statement the scale test must reject.
var originalClaimAuditSQL = `
WITH candidates AS (
    SELECT claim.ctid AS claim_tid
      FROM audit_controller_claims AS claim
      JOIN audits AS audit USING (audit_id)
     WHERE (
               audit.state IN ('active', 'finalizing', 'cancelling', 'deleting')
               OR (
                   audit.state = 'paused'
                   AND EXISTS (
                       SELECT 1 FROM audit_executions AS execution
                        WHERE execution.audit_id = audit.audit_id
                          AND execution.state IN ('submitted', 'collecting')
                   )
               )

           )
       AND ` + auditClaims.AvailableCondition("claim") + `
     ORDER BY claim.epoch, audit.updated_at, audit.audit_id
     FOR UPDATE OF claim SKIP LOCKED
     LIMIT $3
)` + auditClaims.LeaseSQL()

// claimScaleGroup is one group of Audits the claim-scale fixture copies from
// its collecting Audit, each with its claim row.
type claimScaleGroup struct {
	name  string
	state auditstore.AuditState
	count int
	// claim is free, expired (another holder's lease ended an hour ago) or
	// live (another holder's lease ends in an hour).
	claim string
	// deadline, when set, places the deadline that long from now.
	deadline string
	// busy Audits have a collecting execution.
	busy bool
	// candidate Audits pass the reconcile predicate, whatever their claim.
	candidate bool
}

// claimScaleGroups are the Audits of the claim-scale fixture's first
// measurement besides its collecting Audit: a few hundred in the states of
// audits_reconcile_idx, with free, expired and live claims and with paused
// Audits on both sides of their conditions, and a history
// of drafts and 16,200 terminal Audits. The history is large enough that the
// shared lease step fetches the leased claims by their tuple identifiers in
// every plan measured here: while the claim table is a few thousand rows, the
// planner may rather hash the whole small table (it did for a batch of 64 at
// 6,712 claim rows), which reads it once but stops growing where the batch's
// tuple fetches become cheaper.
var claimScaleGroups = []claimScaleGroup{
	{name: "active-free", state: auditstore.AuditActive, count: 120, claim: "free", candidate: true},
	{name: "active-expired", state: auditstore.AuditActive, count: 60, claim: "expired", candidate: true},
	{name: "active-live", state: auditstore.AuditActive, count: 100, claim: "live", candidate: true},
	{name: "finalizing", state: auditstore.AuditFinalizing, count: 10, claim: "free", candidate: true},
	{name: "cancelling", state: auditstore.AuditCancelling, count: 10, claim: "free", candidate: true},
	{name: "deleting", state: auditstore.AuditDeleting, count: 10, claim: "free", candidate: true},
	{name: "paused-busy", state: auditstore.AuditPaused, count: 30, claim: "free", busy: true, candidate: true},
	{name: "paused-idle", state: auditstore.AuditPaused, count: 30, claim: "free"},
	{name: "draft", state: auditstore.AuditDraft, count: 101, claim: "free"},
	{name: "completed", state: auditstore.AuditCompleted, count: 16_000, claim: "free"},
	{name: "cancelled", state: auditstore.AuditCancelled, count: 100, claim: "free"},
	{name: "failed", state: auditstore.AuditFailed, count: 100, claim: "free"},
}

// claimScaleRetained are the completed Audits, with their claim rows, that
// the second measurement adds: every Audit keeps its claim row from
// CreateDraft until it is purged.
var claimScaleRetained = claimScaleGroup{name: "retained", state: auditstore.AuditCompleted, count: 48_000, claim: "free"}

// claimScaleCounts are the claim-scale fixture's Audits in the states of
// audits_reconcile_idx and those that pass the reconcile predicate; its
// collecting Audit, active with a free claim, counts in both.
func claimScaleCounts() (reconcilable, candidates int) {
	reconcilable, candidates = 1, 1
	for _, group := range claimScaleGroups {
		if !group.state.Terminal() && group.state != auditstore.AuditDraft {
			reconcilable += group.count
		}
		if group.candidate {
			candidates += group.count
		}
	}
	return reconcilable, candidates
}

// newClaimScaleFixture is a collecting Audit with its claim released and the
// claimScaleGroups copied from it, in a database of its own. The copies are
// read-model fixtures: they satisfy the schema, not every store rule.
func newClaimScaleFixture(t *testing.T) collectingAuditFixture {
	t.Helper()
	f := newCollectingAuditFixture(t, "claim-scale", 1<<20, collectingAuditFixtureOptions{ownDatabase: true})
	// The copies and the measurements outlast the collecting fixture's
	// deadline.
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	t.Cleanup(cancel)
	f.ctx = ctx
	if err := f.store.ReleaseClaim(f.ctx, f.claim); err != nil {
		t.Fatal(err)
	}
	addClaimScaleAudits(t, f, claimScaleGroups...)
	return f
}

// addClaimScaleAudits copies groups from the fixture's collecting Audit in
// one transaction and analyzes the tables the claim reads.
func addClaimScaleAudits(t *testing.T, f collectingAuditFixture, groups ...claimScaleGroup) {
	t.Helper()
	setup, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = setup.Rollback(context.WithoutCancel(f.ctx)) }()
	// The copied executions satisfy every constraint and foreign key, but
	// not the triggers that guard the store's own execution writes; the test
	// owns its schema, so it switches them off for the copies.
	if _, err := setup.Exec(f.ctx, `ALTER TABLE audit_executions DISABLE TRIGGER USER`); err != nil {
		t.Fatal(err)
	}
	for _, group := range groups {
		id := `'claim-scale-` + group.name + `-' || lpad(n::text, 5, '0')`
		overrides := map[string]string{
			"audit_id": id, "state": `'` + string(group.state) + `'`,
			"current_round_id": `NULL`, "next_event_sequence": `1`,
			// Fixed eligibility rows need no retained input payload. Avoid
			// 64,000 unrelated Artifact hold mutations in this read fixture.
			"baseline_snapshot": `jsonb_set(src.baseline_snapshot,'{inputs}','{}'::jsonb)`,
			"created_at":        `timestamptz '2026-09-01 00:00:00+00' + n * interval '1 second'`,
			"updated_at":        `timestamptz '2026-09-01 00:00:00+00' + n * interval '1 second'`,
		}
		switch {
		case group.state == auditstore.AuditDraft:
			overrides["baseline_snapshot"], overrides["started_at"], overrides["phase"] = `NULL`, `NULL`, `'not-started'`
			overrides["hold_state"], overrides["dispatch_state"], overrides["deadline_at"] = `'pending'`, `'open'`, `NULL`
		case group.state.Terminal():
			overrides["finished_at"] = `src.started_at + interval '1 hour'`
			overrides["hold_state"], overrides["dispatch_state"] = `'released'`, `'closed'`
		case group.state == auditstore.AuditDeleting:
			overrides["deletion_requested_at"] = `clock_timestamp()`
		}
		if group.deadline != "" {
			overrides["deadline_at"] = `clock_timestamp() + interval '` + group.deadline + `'`
		}
		cloneRows(t, f.ctx, setup, "audits", "audit_id", f.audit.AuditID, group.count, overrides)
		holder := map[string]string{
			"free":    `NULL, NULL, NULL`,
			"expired": `'claim-scale-crashed', clock_timestamp() - interval '2 hours', clock_timestamp() - interval '1 hour'`,
			"live":    `'claim-scale-other', clock_timestamp(), clock_timestamp() + interval '1 hour'`,
		}[group.claim]
		if holder == "" {
			t.Fatalf("group %s has claim %q", group.name, group.claim)
		}
		// Epochs 0 to 3, so the claim order mixes the groups.
		if _, err := setup.Exec(f.ctx, `
INSERT INTO audit_controller_claims (audit_id, epoch, holder_id, claimed_at, expires_at)
SELECT 'claim-scale-`+group.name+`-' || lpad(n::text, 5, '0'), n % 4, `+holder+`
  FROM generate_series(1, $1::integer) AS n`, group.count); err != nil {
			t.Fatal(err)
		}
		if group.busy {
			cloneRows(t, f.ctx, setup, "audit_executions", "execution_id", f.execution.ExecutionID, group.count, map[string]string{
				"execution_id": `'claim-scale-execution-' || n`, "audit_id": id,
				"round_id": `NULL`, "role": `'task'`, "role_attempt": `NULL`,
				"submission_key": `'claim-scale-submission-' || n`,
				"run_id":         `'claim-scale-run-' || n`,
				"run_provenance": `jsonb_set(src.run_provenance, '{runId}', to_jsonb('claim-scale-run-' || n))`,
			})
		}
	}
	if _, err := setup.Exec(f.ctx, `ALTER TABLE audit_executions ENABLE TRIGGER USER`); err != nil {
		t.Fatal(err)
	}
	if err := setup.Commit(f.ctx); err != nil {
		t.Fatal(err)
	}
	// INDEX_CLEANUP ON removes the index entries of the collecting Audit's
	// superseded versions too, which VACUUM skips for so few dead rows, and
	// no other test's snapshot can keep them in the fixture's own database,
	// so every measurement reads live entries only.
	for _, table := range []string{"audits", "audit_controller_claims", "audit_executions"} {
		if _, err := f.pool.Exec(f.ctx, `VACUUM (INDEX_CLEANUP ON, ANALYZE) `+pgx.Identifier{table}.Sanitize()); err != nil {
			t.Fatal(err)
		}
	}
}

// explainClaim analyzes one claim pass of sql leasing up to limit Audits in
// mode, rolled back. It first vacuums the claim table, index entries
// included, so the claim versions an earlier rolled-back pass left behind
// are not read again.
func explainClaim(t *testing.T, f collectingAuditFixture, sql string, mode explaintest.Mode, limit int) explaintest.Analysis {
	t.Helper()
	if _, err := f.pool.Exec(f.ctx, `VACUUM (INDEX_CLEANUP ON) audit_controller_claims`); err != nil {
		t.Fatal(err)
	}
	analysis, err := explaintest.ExplainPrepared(f.ctx, f.pool, mode, sql, "claim-scale-controller", int64(30_000), limit)
	if err != nil {
		t.Fatal(err)
	}
	if leased := int(analysis.Plan.Rows); leased != limit {
		t.Fatalf("a %s claim pass of %d leased %d Audits:\n%s", mode, limit, leased, analysis.Plan)
	}
	return analysis
}

// claimScaleMeasurement is what one claim pass read: rows of the claim table
// and of audits, and the shared blocks its candidate selection touched.
type claimScaleMeasurement struct {
	claims, audits, blocks int64
}

// measureClaim measures one analyzed claim pass. Its blocks are those of the
// candidates CTE, which reads the Audits and claims and locks the batch; the
// lease that the shared helper writes afterwards touches blocks for the new
// claim versions, as many as the batch needs wherever free space is, so they
// say nothing about the selection.
func measureClaim(t *testing.T, analysis explaintest.Analysis) claimScaleMeasurement {
	t.Helper()
	measured := claimScaleMeasurement{
		claims: analysis.Reads.Of("audit_controller_claims"), audits: analysis.Reads.Of("audits"), blocks: -1,
	}
	for _, node := range analysis.Plan.Nodes() {
		if node.Relationship == "InitPlan" && node.Type == "Limit" {
			if measured.blocks >= 0 {
				t.Fatalf("the claim plan has two limited InitPlans, want only the candidates:\n%s", analysis.Plan)
			}
			measured.blocks = node.Blocks()
		}
	}
	if measured.blocks < 0 {
		t.Fatalf("the claim plan has no limited candidates InitPlan:\n%s", analysis.Plan)
	}
	return measured
}

// claimBudget is what a claim pass of limit may read from the claim-scale
// fixture: the Audits in the states of audits_reconcile_idx, and one claim
// per Audit that passes the reconcile predicate plus one per Audit leased. A
// claim held by a live lease is read and passed over.
func claimBudget(limit int) claimScaleMeasurement {
	reconcilable, candidates := claimScaleCounts()
	return claimScaleMeasurement{claims: int64(candidates + limit), audits: int64(reconcilable)}
}

// claimBudgetViolations lists how measured exceeds the budget of a claim
// pass of limit, and, against the first measurement before, how it grew.
// Adding the retained Audits deepens the claim table's primary key by at most
// one level, since it holds four times as many entries afterwards and a page
// holds far more than four: the candidate selection descends it once per
// claim it reads, and each descent may touch one more page, and the claims
// it locks lie among more pages, so each lock may touch one more. The Audits
// it reads come from the unchanged partial index.
func claimBudgetViolations(limit int, before, measured claimScaleMeasurement) []string {
	budget := claimBudget(limit)
	var violations []string
	if measured.claims > budget.claims {
		violations = append(violations, fmt.Sprintf("read %d claims, more than %d", measured.claims, budget.claims))
	}
	if measured.audits > budget.audits {
		violations = append(violations, fmt.Sprintf("read %d Audits, more than %d", measured.audits, budget.audits))
	}
	if measured.claims != before.claims || measured.audits != before.audits {
		violations = append(violations, fmt.Sprintf("read %d claims and %d Audits, against %d and %d before the retained Audits",
			measured.claims, measured.audits, before.claims, before.audits))
	}
	if allowance := budget.claims + int64(limit); measured.blocks > before.blocks+allowance {
		violations = append(violations, fmt.Sprintf("touched %d blocks, more than the %d before the retained Audits and %d for deeper claim indexes",
			measured.blocks, before.blocks, allowance))
	}
	return violations
}

// TestPostgresAuditClaimReadsOnlyTheReconcilableAudits holds an Audit
// controller claim pass to the reconcilable Audits: in custom and generic
// plans and for batches of 8 and 64, it reads the Audits of
// audits_reconcile_idx and one claim per candidate plus the batch, and adding
// 48,000 retained terminal Audits with their claim rows changes neither
// count, and the shared blocks only by deeper claim indexes. The original
// statement, measured on the same fixture, must break that budget, so the
// check can fail.
func TestPostgresAuditClaimReadsOnlyTheReconcilableAudits(t *testing.T) {
	disktest.Require(t)
	f := newClaimScaleFixture(t)
	type measurement struct {
		mode  explaintest.Mode
		limit int
	}
	var measurements []measurement
	for _, mode := range explaintest.Modes {
		for _, limit := range []int{8, 64} {
			measurements = append(measurements, measurement{mode, limit})
		}
	}
	first := map[measurement]claimScaleMeasurement{}
	originalFirst := map[measurement]claimScaleMeasurement{}
	for _, m := range measurements {
		first[m] = measureClaim(t, explainClaim(t, f, claimAuditSQL, m.mode, m.limit))
		originalFirst[m] = measureClaim(t, explainClaim(t, f, originalClaimAuditSQL, m.mode, m.limit))
	}
	addClaimScaleAudits(t, f, claimScaleRetained)
	for _, m := range measurements {
		analysis := explainClaim(t, f, claimAuditSQL, m.mode, m.limit)
		measured := measureClaim(t, analysis)
		if violations := claimBudgetViolations(m.limit, first[m], measured); len(violations) != 0 {
			t.Errorf("a %s claim pass of %d with the retained Audits: %v\n%s", m.mode, m.limit, violations, analysis.Plan)
		}
		original := measureClaim(t, explainClaim(t, f, originalClaimAuditSQL, m.mode, m.limit))
		if violations := claimBudgetViolations(m.limit, originalFirst[m], original); len(violations) == 0 {
			t.Errorf("the original %s claim pass of %d kept the budget, so the check cannot fail: %+v, before %+v",
				m.mode, m.limit, original, originalFirst[m])
		}
		t.Logf("%s claim pass of %d: before the retained Audits %+v, after %+v; original before %+v, after %+v",
			m.mode, m.limit, first[m], measured, originalFirst[m], original)
	}
}
