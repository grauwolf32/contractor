package pgstore

import (
	"context"
	"maps"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// auditTriggerProbe is a transaction of a collecting Audit fixture whose
// statements count the triggers they call, with the named deferred
// constraint triggers made immediate so that they are counted too.
type auditTriggerProbe struct {
	collectingAuditFixture
	tx    pgx.Tx
	probe *explaintest.Probe
	store *testStore
}

func newAuditTriggerProbe(t *testing.T, name string, immediate ...string) auditTriggerProbe {
	t.Helper()
	f := newCollectingAuditFixture(t, name, 1<<20, collectingAuditFixtureOptions{extraPending: true})
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tx.Rollback(context.WithoutCancel(f.ctx)) })
	if len(immediate) > 0 {
		if _, err := tx.Exec(f.ctx, `SET CONSTRAINTS `+strings.Join(immediate, ", ")+` IMMEDIATE`); err != nil {
			t.Fatal(err)
		}
	}
	probe := explaintest.NewProbe(t, tx)
	return auditTriggerProbe{collectingAuditFixture: f, tx: tx, probe: probe, store: newTestStore(probe)}
}

// expect runs action and compares the triggers it called on each relation in
// want with the wanted calls.
func (p auditTriggerProbe) expect(t *testing.T, step string, want map[string]map[string]int64, action func() error) {
	t.Helper()
	p.probe.Reset()
	if err := action(); err != nil {
		t.Fatalf("%s: %v", step, err)
	}
	for relation, triggers := range want {
		if fired := p.probe.Fired(relation); !maps.Equal(fired, triggers) {
			t.Errorf("%s fired the %s triggers %v, want %v", step, relation, fired, triggers)
		}
	}
}

// executionScans counts the scans of audit_executions in the transaction.
func (p auditTriggerProbe) executionScans(t *testing.T) int64 {
	t.Helper()
	var scans int64
	if err := p.tx.QueryRow(p.ctx, `
SELECT COALESCE(seq_scan, 0) + COALESCE(idx_scan, 0)
  FROM pg_stat_xact_user_tables WHERE relid = 'audit_executions'::regclass`).Scan(&scans); err != nil {
		t.Fatal(err)
	}
	return scans
}

const (
	preparationGuard   = "audits_protect_preparation_baseline"
	executionAdmission = "audit_executions_project_admission"
	collectionGuard    = "audit_executions_protect_collection"
	preparationCheck   = "audit_executions_validate_preparation"
	matchingRun        = "audit_executions_require_matching_run"
	matchingRunUpdate  = "audit_executions_require_matching_run_on_update"
	childProjection    = "eval_audit_children_projection"

	runAssociationMessage = "AuditExecution Run association is not authoritative"
)

// TestPostgresAuditUpdatesSkipThePreparationProbe counts the work of the
// Audit preparation guard (PER-R22). An event append, which every Audit
// write makes, assigns no baseline column and calls no guard; assigning the
// phase without changing it calls the guard but no longer probes
// audit_executions for a prepare execution, which only a changed baseline
// needs, and such a change still probes.
func TestPostgresAuditUpdatesSkipThePreparationProbe(t *testing.T) {
	p := newAuditTriggerProbe(t, "preparation-probe")
	event := `{"kind":"review.requested","entityId":"review-probe","entityRevision":1,"summary":{}}`
	before := p.executionScans(t)
	p.expect(t, "event append", map[string]map[string]int64{"audits": {}}, func() error {
		var last int64
		return p.probe.QueryRow(p.ctx, `
WITH changed AS (
    UPDATE audits AS audit SET next_event_sequence = audit.next_event_sequence + 1
     WHERE audit.audit_id = $1
    RETURNING contractor_append_audit_events(audit.audit_id, $2::jsonb) AS last_event_sequence
)
SELECT last_event_sequence FROM changed`, p.audit.AuditID, "["+event+"]").Scan(&last)
	})
	p.expect(t, "unchanged phase", map[string]map[string]int64{"audits": {preparationGuard: 1}}, func() error {
		_, err := p.probe.Exec(p.ctx, `UPDATE audits SET phase = phase WHERE audit_id = $1`, p.audit.AuditID)
		return err
	})
	if scans := p.executionScans(t) - before; scans != 0 {
		t.Errorf("an event append and an unchanged phase scanned audit_executions %d times, want none", scans)
	}
	// An Audit whose preparation never began may still change its input
	// selection, which the guard decides by probing for a prepare execution.
	before = p.executionScans(t)
	if _, err := explaintest.Explain(p.ctx, p.tx, `UPDATE audits SET input_selection = '{"changed":true}'
WHERE audit_id = $1`, p.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	if scans := p.executionScans(t) - before; scans != 1 {
		t.Errorf("a changed input selection scanned audit_executions %d times, want 1", scans)
	}
}

// TestPostgresAuditExecutionWritesFireOnlyTheTriggersTheyConcern counts the
// triggers each AuditExecution write calls from intent to collection
// (PER-R22). The deferred Run association check runs for the intent and the
// binding, which set what it compares, and no longer for the terminal
// observation or the collection; the preparation check runs on INSERT and for
// the collection receipt it validates; the Eval Audit child projection runs
// for every write.
func TestPostgresAuditExecutionWritesFireOnlyTheTriggersTheyConcern(t *testing.T) {
	p := newAuditTriggerProbe(t, "execution-triggers", matchingRun, matchingRunUpdate)
	page, err := p.store.ListTasks(p.ctx, auditstore.ListTasksParams{OwnerID: p.audit.OwnerID, AuditID: p.audit.AuditID, Limit: 3})
	if err != nil || len(page.Tasks) != 2 {
		t.Fatalf("ready Tasks: %+v %v", page, err)
	}
	executionTaskID := page.Tasks[1].Document.TaskID
	var execution auditstore.Execution
	p.expect(t, "intent", map[string]map[string]int64{
		"audits":           {},
		"audit_executions": {executionAdmission: 1, preparationCheck: 1, matchingRun: 1, childProjection: 1},
	}, func() error {
		var err error
		execution, _, err = p.store.SelectTasks(p.ctx, auditstore.SelectTasksParams{Claim: p.claim, TaskIDs: []string{page.Tasks[1].Document.TaskID}})

		return err
	})
	const runID = "run-second"
	if err := insertTestRun(p.ctx, p.tx, runID, p.audit.OwnerID, p.project.ProjectID, &execution); err != nil {
		t.Fatal(err)
	}
	p.expect(t, "Run binding", map[string]map[string]int64{
		"audit_executions": {collectionGuard: 1, matchingRunUpdate: 1, childProjection: 1},
	}, func() error {
		var err error
		execution, err = p.store.BindRun(p.ctx, auditstore.BindRunParams{Claim: p.claim, ExecutionID: execution.ExecutionID, RunID: runID})
		return err
	})
	// The Run association is checked again when the Run changes. A Run
	// that references the execution keeps its submission key and identity
	// through its foreign key.
	if _, err := explaintest.Explain(p.ctx, p.tx, `
UPDATE audit_executions
   SET run_id = 'run-elsewhere', run_provenance = jsonb_set(run_provenance, '{runId}', '"run-elsewhere"')
 WHERE execution_id = $1`, execution.ExecutionID); persistencepostgres.SQLState(err) != persistencepostgres.SQLStateCheckViolation ||
		!strings.Contains(err.Error(), runAssociationMessage) {
		t.Errorf("moving the execution to another Run = %v, want the Run association rejected", err)
	}
	var generation string
	var sequence int64
	if _, err := p.tx.Exec(p.ctx, `
UPDATE workflow_runs
   SET state = 'failed', state_reason_code = 'test-terminal', started_at = clock_timestamp(),
       finished_at = clock_timestamp(), updated_at = clock_timestamp()
 WHERE run_id = $1`, runID); err != nil {
		t.Fatal(err)
	}
	if err := p.tx.QueryRow(p.ctx, `SELECT run_event_generation, next_run_event_sequence - 1 FROM workflow_runs
WHERE run_id = $1`, runID).Scan(&generation, &sequence); err != nil {
		t.Fatal(err)
	}
	p.expect(t, "terminal observation", map[string]map[string]int64{
		"audit_executions": {collectionGuard: 1, childProjection: 1},
	}, func() error {
		var err error
		execution, err = p.store.ObserveTerminal(p.ctx, auditstore.ObserveTerminalParams{
			Claim: p.claim, ExecutionID: execution.ExecutionID, RunID: runID, Generation: generation, Sequence: uint64(sequence),
		})
		return err
	})
	code := "run_failed"
	p.expect(t, "collection", map[string]map[string]int64{
		"audit_executions": {collectionGuard: 1, preparationCheck: 1, childProjection: 1},
	}, func() error {
		_, _, err := p.store.Collect(p.ctx, auditstore.CollectParams{
			Claim: p.claim, ReceiptID: "receipt-second", ExecutionID: execution.ExecutionID,
			Disposition: auditstore.CollectionExecutionFailed, ErrorCode: &code, RequestDigest: testDigest("9"),
			Retained: []auditstore.ArtifactLink{},
			Tasks:    []auditstore.TaskCollection{{ExecutionItemID: auditdomain.DeterministicID("attempt", execution.ExecutionID, executionTaskID), TaskID: executionTaskID, GapCode: &code, Retryable: true}},
		})
		return err
	})
	// A deletion mark while the Run still exists is rejected.
	if _, err := explaintest.Explain(p.ctx, p.tx, `UPDATE audit_executions SET run_deleted_at = clock_timestamp()
WHERE execution_id = $1`, execution.ExecutionID); persistencepostgres.SQLState(err) != persistencepostgres.SQLStateCheckViolation ||
		!strings.Contains(err.Error(), runAssociationMessage) {
		t.Errorf("marking an existing Run deleted = %v, want the Run association rejected", err)
	}
	p.expect(t, "updated_at only", map[string]map[string]int64{
		"audit_executions": {childProjection: 1},
	}, func() error {
		_, err := p.probe.Exec(p.ctx, `UPDATE audit_executions SET updated_at = clock_timestamp() WHERE execution_id = $1`,
			execution.ExecutionID)
		return err
	})

	// Every column of a collection receipt, changed alone, is still rejected.
	for _, column := range []string{
		"execution_id", "audit_id", "run_id", "terminal_outcome", "terminal_run_generation",
		"terminal_run_sequence", "collection_receipt_id", "collection_disposition",
		"collection_source_output_ref", "collection_source_output_digest", "collection_retained_refs",
		"collection_error_code", "collection_request_digest", "collected_at",
	} {
		value, err := explaintest.Changed(p.ctx, p.tx, "audit_executions", column)
		if err != nil {
			t.Fatal(err)
		}
		statement := `UPDATE audit_executions SET ` + column + ` = ` + value + ` WHERE execution_id = $1`
		if _, err := explaintest.Explain(p.ctx, p.tx, statement, execution.ExecutionID); err == nil ||
			!strings.Contains(err.Error(), "Audit collection receipts are immutable") {
			t.Errorf("%s = %v, want the collection receipt protected", statement, err)
		}
	}
}
