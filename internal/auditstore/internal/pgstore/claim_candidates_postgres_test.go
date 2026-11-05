package pgstore

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

// claimCondition is one reconcile condition of the candidate matrix: an
// Audit state and executions that decide whether the Audit is a candidate.
type claimCondition struct {
	name  string
	state auditstore.AuditState
	// execution is the state of the Audit's one execution, if it has one.
	execution string
	// candidate is whether the Audit passes the reconcile predicate.
	candidate bool
}

// claimConditions covers every way an Audit becomes a claim candidate and
// the near misses next to each.
var claimConditions = []claimCondition{
	{name: "active", state: auditstore.AuditActive, candidate: true},
	{name: "finalizing", state: auditstore.AuditFinalizing, candidate: true},
	{name: "cancelling", state: auditstore.AuditCancelling, candidate: true},
	{name: "deleting", state: auditstore.AuditDeleting, candidate: true},
	{name: "paused without executions", state: auditstore.AuditPaused},
	{name: "paused with an intent", state: auditstore.AuditPaused, execution: "intent"},
	{name: "paused with a submitted execution", state: auditstore.AuditPaused, execution: "submitted", candidate: true},
	{name: "paused with a collecting execution", state: auditstore.AuditPaused, execution: "collecting", candidate: true},
	{name: "draft", state: auditstore.AuditDraft},
	{name: "completed", state: auditstore.AuditCompleted},
	{name: "cancelled", state: auditstore.AuditCancelled},
	{name: "failed", state: auditstore.AuditFailed},
}

// claimMatrixAudit is one Audit of the candidate matrix: a condition with a
// free, expired or live claim at an epoch.
type claimMatrixAudit struct {
	id        string
	condition claimCondition
	claim     string
	epoch     uint64
	// minute orders updated_at; Audits of the same minute fall back to their
	// identifiers, which differ only in fixed-width digits so every collation
	// orders them alike.
	minute int
}

// claimMatrix places every condition with each kind of claim, with epochs
// and update times that tie often enough to exercise each sort key.
func claimMatrix() []claimMatrixAudit {
	var audits []claimMatrixAudit
	for _, condition := range claimConditions {
		for _, claim := range []string{"free", "expired", "live"} {
			index := len(audits)
			audits = append(audits, claimMatrixAudit{
				id: fmt.Sprintf("claim-matrix-%03d", index), condition: condition, claim: claim,
				epoch: uint64(index*5) % 3, minute: (index * 3) % 7,
			})
		}
	}
	return audits
}

// expectedClaimOrder is the order in which a controller leases the matrix:
// candidates whose claim is free or expired, by (epoch, updated_at, audit_id).
func expectedClaimOrder(audits []claimMatrixAudit) []claimMatrixAudit {
	var expected []claimMatrixAudit
	for _, audit := range audits {
		if audit.condition.candidate && audit.claim != "live" {
			expected = append(expected, audit)
		}
	}
	slices.SortFunc(expected, func(a, b claimMatrixAudit) int {
		return cmp.Or(cmp.Compare(a.epoch, b.epoch), cmp.Compare(a.minute, b.minute), cmp.Compare(a.id, b.id))
	})
	return expected
}

// newClaimMatrixFixture copies the matrix from a collecting Audit, whose own
// claim another holder keeps for an hour. The copies are read-model
// fixtures: they satisfy the schema, not every store rule.
func newClaimMatrixFixture(t *testing.T, audits []claimMatrixAudit) collectingAuditFixture {
	t.Helper()
	f := newCollectingAuditFixture(t, "claim-matrix", 1<<20)
	if _, err := f.pool.Exec(f.ctx, `
UPDATE audit_controller_claims
   SET holder_id = 'claim-matrix-keeper', claimed_at = clock_timestamp(), expires_at = clock_timestamp() + interval '1 hour'
 WHERE audit_id = $1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}

	setup, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = setup.Rollback(context.WithoutCancel(f.ctx)) }()
	if _, err := setup.Exec(f.ctx, `ALTER TABLE audit_executions DISABLE TRIGGER USER`); err != nil {
		t.Fatal(err)
	}
	quote := func(value string) string { return `'` + value + `'` }
	for _, audit := range audits {
		condition, id := audit.condition, quote(audit.id)
		updated := fmt.Sprintf(`timestamptz '2026-09-01 00:00:00+00' + interval '%d minutes'`, audit.minute)
		overrides := map[string]string{
			"audit_id": id, "state": quote(string(condition.state)),
			"next_event_sequence": `1`, "created_at": updated, "updated_at": updated,
			"deadline_at": `clock_timestamp() + interval '1 hour'`,
		}
		switch {
		case condition.state == auditstore.AuditDraft:
			overrides["baseline_snapshot"], overrides["started_at"] = `NULL`, `NULL`
			overrides["hold_state"], overrides["dispatch_state"], overrides["deadline_at"] = `'pending'`, `'open'`, `NULL`
		case condition.state.Terminal():
			overrides["finished_at"] = `src.started_at + interval '1 hour'`
			overrides["hold_state"], overrides["dispatch_state"] = `'released'`, `'closed'`
		case condition.state == auditstore.AuditDeleting:
			overrides["deletion_requested_at"] = `clock_timestamp()`
		}
		cloneRows(t, f.ctx, setup, "audits", "audit_id", f.audit.AuditID, 1, overrides)
		holder := map[string]string{
			"free":    `NULL, NULL, NULL`,
			"expired": `'claim-matrix-crashed', clock_timestamp() - interval '2 hours', clock_timestamp() - interval '1 hour'`,
			"live":    `'claim-matrix-other', clock_timestamp(), clock_timestamp() + interval '1 hour'`,
		}[audit.claim]
		if _, err := setup.Exec(f.ctx, `
INSERT INTO audit_controller_claims (audit_id, epoch, holder_id, claimed_at, expires_at)
VALUES ($1, $2, `+holder+`)`, audit.id, int64(audit.epoch)); err != nil {
			t.Fatal(err)
		}
		if condition.execution != "" {
			execution := map[string]string{
				"execution_id": quote(audit.id + "-execution"), "audit_id": id, "state": quote(condition.execution),
				"submission_key": quote(audit.id + "-submission"), "run_id": quote(audit.id + "-run"),
				"run_provenance": `jsonb_set(src.run_provenance, '{runId}', to_jsonb(` + quote(audit.id+"-run") + `::text))`,
			}
			if condition.execution != "collecting" {
				for _, column := range []string{"terminal_outcome", "terminal_run_generation", "terminal_run_sequence", "terminal_observed_at"} {
					execution[column] = `NULL`
				}
			}
			if condition.execution == "intent" {
				execution["run_id"], execution["run_provenance"] = `NULL`, `NULL`
			}
			cloneRows(t, f.ctx, setup, "audit_executions", "execution_id", f.execution.ExecutionID, 1, execution)
		}

	}
	if _, err := setup.Exec(f.ctx, `ALTER TABLE audit_executions ENABLE TRIGGER USER`); err != nil {
		t.Fatal(err)
	}
	if err := setup.Commit(f.ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `ANALYZE audits, audit_controller_claims, audit_executions, audit_tasks`); err != nil {
		t.Fatal(err)
	}
	return f
}

// matrixClaimer leases up to limit Audits to the matrix controller in tx.
type matrixClaimer func(ctx context.Context, tx pgx.Tx, limit int) ([]auditstore.ControllerClaim, error)

// matrixClaimers are the rewritten statement, through the store, and the
// original one.
var matrixClaimers = map[string]matrixClaimer{
	"claimAuditSQL": func(ctx context.Context, tx pgx.Tx, limit int) ([]auditstore.ControllerClaim, error) {
		return newTestStore(tx).Claim(ctx, auditstore.ClaimParams{HolderID: "claim-matrix-controller", Lease: time.Minute, Limit: limit})
	},
	"original": func(ctx context.Context, tx pgx.Tx, limit int) ([]auditstore.ControllerClaim, error) {
		rows, err := tx.Query(ctx, originalClaimAuditSQL, "claim-matrix-controller", time.Minute.Milliseconds(), limit)
		if err != nil {
			return nil, err
		}
		return pgx.CollectRows(rows, func(row pgx.CollectableRow) (auditstore.ControllerClaim, error) { return scanClaim(row) })
	},
}

// errMatrixRollback rolls a matrix lease back.
var errMatrixRollback = errors.New("rolled back")

// leaseInRolledBackTx runs lease in a transaction that it rolls back, so
// every claimer starts from the same claims.
func leaseInRolledBackTx(t *testing.T, f collectingAuditFixture, lease func(pgx.Tx) error) {
	t.Helper()
	err := pgx.BeginFunc(f.ctx, f.pool, func(tx pgx.Tx) error {
		if err := lease(tx); err != nil {
			return err
		}
		return errMatrixRollback
	})
	if !errors.Is(err, errMatrixRollback) {
		t.Fatal(err)
	}
}

// leasedOneByOne leases one Audit at a time until none is left, which is
// the order the claimer leases them in, and checks each lease.
func leasedOneByOne(t *testing.T, f collectingAuditFixture, claim matrixClaimer) []string {
	t.Helper()
	var leased []string
	leaseInRolledBackTx(t, f, func(tx pgx.Tx) error {
		for range len(claimConditions) * 3 {
			claims, err := claim(f.ctx, tx, 1)
			if err != nil || len(claims) == 0 {
				return err
			}
			leased = append(leased, describeMatrixClaims(claims)...)
		}
		return errors.New("the claimer never ran out of Audits")
	})
	return leased
}

// leasedAtOnce leases up to limit Audits in one pass and returns them in
// identifier order.
func leasedAtOnce(t *testing.T, f collectingAuditFixture, claim matrixClaimer, limit int) []string {
	t.Helper()
	var leased []string
	leaseInRolledBackTx(t, f, func(tx pgx.Tx) error {
		claims, err := claim(f.ctx, tx, limit)
		leased = describeMatrixClaims(claims)
		slices.Sort(leased)
		return err
	})
	return leased
}

// describeMatrixClaims renders each lease as its Audit and new epoch, and
// marks a lease that is not the matrix controller's.
func describeMatrixClaims(claims []auditstore.ControllerClaim) []string {
	described := make([]string, len(claims))
	for index, claim := range claims {
		described[index] = fmt.Sprintf("%s@%d", claim.AuditID, claim.Epoch)
		if claim.HolderID != "claim-matrix-controller" || !claim.ExpiresAt.After(claim.ClaimedAt) {
			described[index] += " (lease " + claim.HolderID + ")"
		}
	}
	return described
}

// describeExpected renders the expected leases of audits like
// describeMatrixClaims.
func describeExpected(audits []claimMatrixAudit) []string {
	described := make([]string, len(audits))
	for index, audit := range audits {
		described[index] = fmt.Sprintf("%s@%d", audit.id, audit.epoch+1)
	}
	return described
}

// TestPostgresAuditClaimLeasesTheOriginalCandidatesInOrder holds the
// rewritten claim to the original statement's candidates: on a matrix of
// every reconcile condition and its near misses, each with a free, an
// expired and a live claim, both lease the same Audits in the same order with
// the same epochs, one at a time and in batches, and both skip claims that
// another transaction holds locked.
func TestPostgresAuditClaimLeasesTheOriginalCandidatesInOrder(t *testing.T) {
	audits := claimMatrix()
	f := newClaimMatrixFixture(t, audits)
	expected := expectedClaimOrder(audits)
	if len(expected) < 12 {
		t.Fatalf("the matrix has %d candidates, want enough to order", len(expected))
	}
	names := make(map[string]string, len(audits))
	for _, audit := range audits {
		names[audit.id] = audit.condition.name + ", " + audit.claim + " claim"
	}
	sorted := func(values []string) []string { values = slices.Clone(values); slices.Sort(values); return values }
	named := func(lease string) string {
		id, _, _ := strings.Cut(lease, "@")
		return lease + " (" + names[id] + ")"
	}
	check := func(what string, got, want []string) {
		t.Helper()
		if slices.Equal(got, want) {
			return
		}
		var missing, extra []string
		for _, lease := range want {
			if !slices.Contains(got, lease) {
				missing = append(missing, named(lease))
			}
		}
		for _, lease := range got {
			if !slices.Contains(want, lease) {
				extra = append(extra, named(lease))
			}
		}
		t.Errorf("%s leased\n%v\nwant\n%v\nmissing %v\nextra %v", what, got, want, missing, extra)
	}
	for name, claim := range matrixClaimers {
		check(name+" one at a time", leasedOneByOne(t, f, claim), describeExpected(expected))
		for _, limit := range []int{5, 100} {
			check(fmt.Sprintf("%s in a batch of %d", name, limit), leasedAtOnce(t, f, claim, limit),
				sorted(describeExpected(expected[:min(limit, len(expected))])))
		}
	}

	// Another transaction holds the first and third candidates' claims, as
	// a controller write proving its claim does.
	blocker, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = blocker.Rollback(context.WithoutCancel(f.ctx)) }()
	if _, err := blocker.Exec(f.ctx, `SELECT audit_id FROM audit_controller_claims WHERE audit_id = ANY($1) FOR UPDATE`,
		[]string{expected[0].id, expected[2].id}); err != nil {
		t.Fatal(err)
	}
	unlocked := slices.Concat(expected[1:2], expected[3:])
	for name, claim := range matrixClaimers {
		check(name+" one at a time beside locked claims", leasedOneByOne(t, f, claim), describeExpected(unlocked))
		check(name+" in a batch of 3 beside locked claims", leasedAtOnce(t, f, claim, 3), sorted(describeExpected(unlocked[:3])))
	}
}
