//go:build integration

package auditcontroller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5/pgxpool"
)

// writeCountingStore is the real claim store counting the Controller writes
// whose invalid refusal must close the Audit instead of being retried, so a
// test sees whether a later pass attempts a refused write again.
type writeCountingStore struct {
	Store
	mu    sync.Mutex
	calls map[string]int
}

func countWrites(c *Controller) *writeCountingStore {
	store := &writeCountingStore{Store: c.store, calls: map[string]int{}}
	c.store = store
	return store
}

func (s *writeCountingStore) count(operation string) {
	s.mu.Lock()
	s.calls[operation]++
	s.mu.Unlock()
}

func (s *writeCountingStore) attempts(operation string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls[operation]
}

func (s *writeCountingStore) AcceptInitialRound(
	ctx context.Context, params auditstore.AcceptInitialRoundParams,
) (auditstore.Round, bool, error) {
	s.count("AcceptInitialRound")
	return s.Store.AcceptInitialRound(ctx, params)
}

func (s *writeCountingStore) CreateExecutionIntent(
	ctx context.Context, params auditstore.CreateExecutionIntentParams,
) (auditstore.Execution, bool, error) {
	s.count("CreateExecutionIntent")
	return s.Store.CreateExecutionIntent(ctx, params)
}

// lowerStoredBound replaces constraint on table with a CHECK of the same name
// and measure, octet_length(column::text) <= 1, so the production constraint
// refuses every later write of column. NOT VALID keeps the rows already
// stored.
func lowerStoredBound(t *testing.T, ctx context.Context, pool *pgxpool.Pool, table, constraint, column string) {
	t.Helper()
	if _, err := pool.Exec(ctx, fmt.Sprintf(
		`ALTER TABLE %s DROP CONSTRAINT %s, ADD CONSTRAINT %s CHECK (octet_length(%s::text) <= 1) NOT VALID`,
		table, constraint, constraint, column,
	)); err != nil {
		t.Fatal(err)
	}
}

type passResult struct {
	worked bool
	err    error
}

// reconcilePasses runs count Controller passes and returns each outcome
// without judging it, so a test can first count the attempts of a refused
// write.
func reconcilePasses(ctx context.Context, c *Controller, count int) []passResult {
	results := make([]passResult, count)
	for index := range results {
		results[index].worked, results[index].err = c.RunOnce(ctx)
	}
	return results
}

func requireNoFailedPass(t *testing.T, results []passResult) {
	t.Helper()
	for index, result := range results {
		if result.err != nil {
			t.Fatalf("pass %d failed: %v", index+1, result.err)
		}
	}
}

func requireStopReason(t *testing.T, audit auditstore.Audit, state auditstore.AuditState, code, message string) {
	t.Helper()
	if audit.State != state || audit.Dispatch != auditstore.DispatchClosed || audit.StopReason == nil ||
		audit.StopReason.Code != code || audit.StopReason.Message != message {
		t.Fatalf("Audit = state %s, dispatch %s, stop reason %+v; want %s with %s: %q",
			audit.State, audit.Dispatch, audit.StopReason, state, code, message)
	}
}

// A first Round that the store refuses through its production CHECK closes
// the prepared Audit with invalid_inventory after one attempt. No later pass
// accepts it again, the claim is released after the refusing pass, and the
// Audit drains to failed without a Round or items (AUX-R99, ASC-R51).
func TestPostgresControllerClosesAuditWhoseInitialRoundIsRefused(t *testing.T) {
	ctx, h, c := generatedInventoryHarness(t)
	readyGeneratedInventory(t, ctx, h, c)
	c.initialRoundBuilder = h.service
	writes := countWrites(c)
	lowerStoredBound(t, ctx, h.pool, "audit_items", "audit_items_coverage_gaps_check", "coverage_gaps")

	first := reconcilePasses(ctx, c, 1)
	if holder, _ := h.auditClaim(t, ctx); holder != nil {
		t.Fatalf("the refusing pass left the claim held by %q", *holder)
	}
	results := append(first, reconcilePasses(ctx, c, 5)...)
	if got := writes.attempts("AcceptInitialRound"); got != 1 {
		t.Fatalf("refused initial Round was attempted %d times, want 1", got)
	}
	requireNoFailedPass(t, results)
	if !results[0].worked {
		t.Fatal("the refusing pass reported no progress")
	}
	audit, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	requireStopReason(t, audit, auditstore.AuditFailed, "invalid_inventory",
		"The prepared Audit inventory failed validation: the Audit store refused its initial Round.")
	var rounds, items int
	if err := h.pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM audit_rounds WHERE audit_id = $1),
       (SELECT count(*) FROM audit_items WHERE audit_id = $1)`, audit.AuditID).Scan(&rounds, &items); err != nil ||
		rounds != 0 || items != 0 || audit.CurrentRoundID != nil {
		t.Fatalf("refused initial Round left rounds=%d items=%d current=%v (%v)", rounds, items, audit.CurrentRoundID, err)
	}
}

// A preparation role intent that the store refuses through its production
// CHECK closes the Audit after one attempt, with the preparation stop reason
// a role whose dispatch data fails validation has, and reserves no Run
// (AUX-R99, ASC-R59).
func TestPostgresControllerClosesAuditWhoseRoleIntentIsRefused(t *testing.T) {
	ctx, h, c := preparationHarness(t, false)
	writes := countWrites(c)
	lowerStoredBound(t, ctx, h.pool, "audit_executions", "audit_executions_manifest_ref_check", "manifest_ref")

	first := reconcilePasses(ctx, c, 1)
	if holder, _ := h.auditClaim(t, ctx); holder != nil {
		t.Fatalf("the refusing pass left the claim held by %q", *holder)
	}
	results := append(first, reconcilePasses(ctx, c, 5)...)
	if got := writes.attempts("CreateExecutionIntent"); got != 1 {
		t.Fatalf("refused role intent was attempted %d times, want 1", got)
	}
	requireNoFailedPass(t, results)
	audit, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	requireStopReason(t, audit, auditstore.AuditFailed, "preparation_failed",
		"Pinned Audit role dispatch data failed deterministic validation.")
	if audit.ReservedRunCount != 0 {
		t.Fatalf("refused role intent reserved %d Runs", audit.ReservedRunCount)
	}
	preparationExecutions(t, ctx, h, 0)
}

// A check execution intent that the store refuses through its production
// CHECK closes the Audit with dispatch_contract_invalid after one attempt;
// its undispatched item then settles as excluded (AUX-R99, AUD-R186).
func TestPostgresControllerClosesAuditWhoseCheckIntentIsRefused(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	h := newPostgresControllerHarness(t, ctx, 1)
	c := h.controller(t)
	writes := countWrites(c)
	lowerStoredBound(t, ctx, h.pool, "audit_executions", "audit_executions_manifest_ref_check", "manifest_ref")

	results := reconcilePasses(ctx, c, 6)
	if got := writes.attempts("CreateExecutionIntent"); got != 1 {
		t.Fatalf("refused check intent was attempted %d times, want 1", got)
	}
	requireNoFailedPass(t, results)
	if holder, _ := h.auditClaim(t, ctx); holder != nil {
		t.Fatalf("the claim is still held by %q", *holder)
	}
	audit, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	requireStopReason(t, audit, auditstore.AuditFinalizing, "dispatch_contract_invalid",
		"Pinned Audit dispatch data failed deterministic validation.")
	executions, err := h.audits.ListExecutions(ctx, audit.AuditID)
	if err != nil || len(executions) != 0 || audit.ReservedRunCount != 0 {
		t.Fatalf("refused check intent left executions %+v and %d reserved Runs (%v)",
			executions, audit.ReservedRunCount, err)
	}
	items, err := h.audits.Items(ctx, audit.OwnerID, audit.AuditID)
	if err != nil || len(items) != 1 || items[0].State != auditstore.ItemSettled ||
		items[0].FinalDisposition == nil || *items[0].FinalDisposition != auditstore.FinalExcluded {
		t.Fatalf("undispatched item after closure = (%+v, %v), want settled as excluded", items, err)
	}
}

// injectWriteFailureSQL makes every insert into audit_executions fail the way
// contractor_test_inject_failure names: a serialization failure (40001), or
// the loss of the connection, which terminates the backend mid-statement.
const injectWriteFailureSQL = `
CREATE TABLE contractor_test_inject_failure (failure text NOT NULL);
CREATE FUNCTION contractor_test_inject_write_failure() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    injected text;
BEGIN
    SELECT failure INTO injected FROM contractor_test_inject_failure LIMIT 1;
    IF injected = 'serialization' THEN
        RAISE EXCEPTION 'injected serialization failure' USING ERRCODE = 'serialization_failure';
    ELSIF injected = 'connection' THEN
        PERFORM pg_terminate_backend(pg_backend_pid());
        PERFORM pg_sleep(30);
    END IF;
    RETURN NULL;
END $$;
CREATE TRIGGER contractor_test_inject_write_failure BEFORE INSERT ON audit_executions
FOR EACH STATEMENT EXECUTE FUNCTION contractor_test_inject_write_failure();`

// A transient failure of an execution intent, a serialization failure or a
// lost connection, is not an invalid refusal: the Audit stays active, a later
// pass attempts the same intent again, and it dispatches once the failure is
// gone (AUX-R99, FAIL-AUX-4).
func TestPostgresControllerRetriesTransientIntentFailures(t *testing.T) {
	for _, failure := range []string{"serialization", "connection"} {
		t.Run(failure, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			h := newPostgresControllerHarness(t, ctx, 1)
			c := h.controller(t)
			writes := countWrites(c)
			if _, err := h.pool.Exec(ctx, injectWriteFailureSQL); err != nil {
				t.Fatal(err)
			}
			if worked, err := c.RunOnce(ctx); err != nil || !worked {
				t.Fatalf("start the Round = (%t, %v)", worked, err)
			}
			if _, err := h.pool.Exec(ctx, `INSERT INTO contractor_test_inject_failure VALUES ($1)`, failure); err != nil {
				t.Fatal(err)
			}
			for attempt := 1; attempt <= 2; attempt++ {
				worked, err := c.RunOnce(ctx)
				if worked || err == nil || errors.Is(err, auditstore.ErrInvalid) ||
					persistencepostgres.Classify(err) == persistencepostgres.ClassInvalid ||
					failure == "serialization" && !persistencepostgres.IsTransactionConflict(err) {
					t.Fatalf("transient attempt %d = (%t, %v), want a retryable failure", attempt, worked, err)
				}
				if got := writes.attempts("CreateExecutionIntent"); got != attempt {
					t.Fatalf("after transient attempt %d the intent was attempted %d times", attempt, got)
				}
			}
			audit, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
			if err != nil || audit.State != auditstore.AuditActive || audit.Dispatch != auditstore.DispatchOpen ||
				audit.StopReason != nil {
				t.Fatalf("transient failures changed the Audit: %+v (%v)", audit, err)
			}
			if _, err := h.pool.Exec(ctx, `DELETE FROM contractor_test_inject_failure`); err != nil {
				t.Fatal(err)
			}
			if worked, err := c.RunOnce(ctx); err != nil || !worked {
				t.Fatalf("dispatch after the failure = (%t, %v)", worked, err)
			}
			executions, err := h.audits.ListExecutions(ctx, audit.AuditID)
			if err != nil || len(executions) != 1 || executions[0].RunID == nil ||
				executions[0].State != auditstore.ExecutionSubmitted {
				t.Fatalf("retried intent = (%+v, %v), want one submitted execution", executions, err)
			}
		})
	}
}

// fullCoverageGaps is a gap array whose jsonb text is exactly the stored
// bound of audit_items_coverage_gaps_check: 4096 distinct values of 252
// bytes, 1,048,576 bytes of jsonb text and 1,044,481 bytes of compact JSON,
// which the store's validation of a collected coverage accepts.
func fullCoverageGaps(t *testing.T) []byte {
	t.Helper()
	gaps := make([]string, 4096)
	for index := range gaps {
		gaps[index] = fmt.Sprintf("gap-%04d-%s", index, strings.Repeat("x", 252-9))
	}
	encoded, err := json.Marshal(gaps)
	if err != nil || len(encoded) > auditstore.MaxCoverageArrayBytes {
		t.Fatalf("full gap array = %d bytes (%v)", len(encoded), err)
	}
	return encoded
}

// An undispatched item whose coverage gaps already fill their stored bound
// still settles when its Audit closes: settlement keeps the full gap array
// instead of appending a closure gap the CHECK would refuse on every pass, so
// the cancelled Audit drains to cancelled (AUX-R100, AUD-R186).
func TestPostgresControllerSettlesAnItemWhoseCoverageGapsAreFull(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	h := newPostgresControllerHarness(t, ctx, 1)
	c := h.controller(t)
	var storedBytes int
	if err := h.pool.QueryRow(ctx, `
UPDATE audit_items SET coverage_gaps = $2::jsonb WHERE audit_id = $1
RETURNING octet_length(coverage_gaps::text)`, h.started.Audit.AuditID, fullCoverageGaps(t)).Scan(&storedBytes); err != nil ||
		storedBytes != auditstore.MaxCoverageArrayBytes {
		t.Fatalf("full gap array stored as %d bytes of jsonb text (%v)", storedBytes, err)
	}
	audit, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.service.Cancel(ctx, preparationMutation(audit, "cancel-full-gaps")); err != nil {
		t.Fatal(err)
	}

	requireNoFailedPass(t, reconcilePasses(ctx, c, 5))
	audit, err = h.audits.Get(ctx, audit.OwnerID, audit.AuditID)
	if err != nil || audit.State != auditstore.AuditCancelled {
		t.Fatalf("cancelled Audit = %+v (%v), want cancelled", audit, err)
	}
	items, err := h.audits.Items(ctx, audit.OwnerID, audit.AuditID)
	if err != nil || len(items) != 1 || items[0].State != auditstore.ItemSettled ||
		items[0].FinalDisposition == nil || *items[0].FinalDisposition != auditstore.FinalExecutionCancelled {
		t.Fatalf("item after settlement = (%+v, %v), want settled as cancelled", items, err)
	}
	var gaps int
	if err := h.pool.QueryRow(ctx, `
SELECT jsonb_array_length(coverage_gaps) FROM audit_items WHERE audit_id = $1`, audit.AuditID).Scan(&gaps); err != nil ||
		gaps != 4096 {
		t.Fatalf("settled item has %d gaps (%v), want its 4096 unchanged", gaps, err)
	}
}
