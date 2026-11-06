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

const (
	executionAdmission = "audit_executions_project_admission"
	collectionGuard    = "audit_executions_protect_collection"
	matchingRun        = "audit_executions_require_matching_run"
	matchingRunUpdate  = "audit_executions_require_matching_run_on_update"
	childProjection    = "eval_audit_children_projection"

	runAssociationMessage = "AuditExecution Run association is not authoritative"
)

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
		"audit_executions": {executionAdmission: 1, matchingRun: 1, childProjection: 1},
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
		"audit_executions": {collectionGuard: 1, childProjection: 1},
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
