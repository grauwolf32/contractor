package runstore

import (
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestPostgresStageDecisionSingleWriteAndExactTarget(t *testing.T) {
	ctx := t.Context()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool, artifactpgtest.NewStorage())
	createTestRun(t, ctx, store, "decision-run")
	if _, err := store.CreateStageExecution(ctx, testActiveStageParams("decision-run", "decision-source", "source")); err != nil {
		t.Fatal(err)
	}
	params := RecordStageTransitionDecisionParams{SourceExecutionID: "decision-source", RunID: "decision-run", Action: StageTransitionFail}
	// Independent callers must get one immutable decision, including when the
	// losing request would otherwise replace it with a different action.
	start := make(chan struct{})
	results := make(chan error, 8)
	var workers sync.WaitGroup
	for i := range 8 {
		workers.Go(func() {
			<-start
			p := params
			if i%2 == 0 {
				p.Action = StageTransitionSucceed
			}
			_, err := store.RecordStageTransitionDecision(ctx, p)
			results <- err
		})
	}
	close(start)
	workers.Wait()
	close(results)
	successes := 0
	for err := range results {
		if err == nil {
			successes++
		} else if !errors.Is(err, ErrConflict) {
			t.Fatal(err)
		}
	}
	if successes != 1 {
		t.Fatalf("accepted %d decisions, want one", successes)
	}
	before, err := store.ListStageTransitionDecisions(ctx, "decision-run")
	if err != nil || len(before) != 1 {
		t.Fatalf("decisions=%v: %v", before, err)
	}
	for _, change := range []string{
		`transition_action=NULL,transition_decided_at=NULL`,
		`transition_decided_at=transition_decided_at+interval '1 second'`,
		`transition_escalation_ordinal=1,transition_escalation_exhausted=true`,
		`transition_target_stage_name='other',transition_target_execution_id='other'`,
	} {
		if _, err := pool.Exec(ctx, `UPDATE stage_executions SET `+change+` WHERE stage_execution_id='decision-source'`); persistencepostgres.SQLState(err) != persistencepostgres.SQLStateCheckViolation {
			t.Fatalf("decision rewrite %s: %v", change, err)
		}
	}
	after, err := store.ListStageTransitionDecisions(ctx, "decision-run")
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("decision changed: %v", err)
	}
	params.SourceExecutionID = "missing-source"
	if _, err := store.RecordStageTransitionDecision(ctx, params); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing source=%v", err)
	}
	// A target in another Run may exist, but is not an exact progression target.
	createTestRun(t, ctx, store, "foreign-run")
	if _, err := store.CreateStageExecution(ctx, testActiveStageParams("foreign-run", "foreign-stage", "foreign")); err != nil {
		t.Fatal(err)
	}
	name, target := "foreign", "foreign-stage"
	// Use a fresh source so uniqueness cannot hide the exact-target check.
	createTestRun(t, ctx, store, "other-run")
	if _, err := store.CreateStageExecution(ctx, testActiveStageParams("other-run", "other-stage", "other")); err != nil {
		t.Fatal(err)
	}
	if _, err := store.RecordStageTransitionDecision(ctx, RecordStageTransitionDecisionParams{
		SourceExecutionID: "other-stage", RunID: "other-run", Action: StageTransitionNext,
		TargetStageName: &name, TargetExecutionID: &target,
	}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("foreign target=%v", err)
	}
}

func TestPostgresManualReceiptCreatedWithAttemptAndRollback(t *testing.T) {
	ctx := t.Context()
	pool := isolatedRunStorePool(t, ctx)
	store, source := failedResumeFixture(t, ctx, pool, "receipt-run")
	before, err := store.GetRun(ctx, source.RunID)
	if err != nil {
		t.Fatal(err)
	}
	// New targets cannot borrow a historical transaction or another owner.
	for _, authority := range []string{"'1'::xid8, 'user-1'", "pg_current_xact_id(), 'other-owner'"} {
		_, err := pool.Exec(ctx, `INSERT INTO stage_executions(stage_execution_id,run_id,stage_name,attempt,previous_execution_id,
 resume_source_execution_id,execution_config_variant,escalation_ordinal,stage_spec_schema_version,stage_spec_snapshot,
 stage_context_schema_version,stage_context_snapshot,state,state_reason_code,resume_requested_at,resume_transaction_id,resume_requested_by)
 SELECT 'forged-target',run_id,stage_name,attempt+1,stage_execution_id,stage_execution_id,execution_config_variant,escalation_ordinal,
 stage_spec_schema_version,stage_spec_snapshot,stage_context_schema_version,stage_context_snapshot,'preparing','created',clock_timestamp(),`+authority+`
 FROM stage_executions WHERE stage_execution_id=$1`, source.StageExecutionID)
		if persistencepostgres.SQLState(err) != persistencepostgres.SQLStateCheckViolation {
			t.Fatalf("forged receipt authority %s: %v", authority, err)
		}
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, err := NewPostgresStore(tx, artifactpgtest.NewStorage()).ResumeFailedRun(ctx, before.OwnerID, source.RunID, source.StageExecutionID, "rolled-back-target"); err != nil {
		t.Fatal(err)
	}
	var current bool
	if err := tx.QueryRow(ctx, `SELECT resume_transaction_id=pg_current_xact_id() AND resume_requested_by=$1 AND resume_requested_at IS NOT NULL FROM stage_executions WHERE stage_execution_id='rolled-back-target'`, before.OwnerID).Scan(&current); err != nil || !current {
		t.Fatalf("receipt not owned by creating transaction: %v", err)
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	after, err := store.GetRun(ctx, source.RunID)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("rolled-back continuation changed Run: %v", err)
	}
	if _, err := store.GetStageExecution(ctx, "rolled-back-target"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("rolled-back target survived: %v", err)
	}
	var frozen bool
	if err := pool.QueryRow(ctx, `SELECT frozen FROM artifact_bindings WHERE scope_kind='run' AND scope_id=$1 AND namespace='outputs' AND name='upstream'`, source.RunID).Scan(&frozen); err != nil || !frozen {
		t.Fatalf("rolled-back continuation thawed outputs: %v", err)
	}
	// A receipt cannot be attached to a terminal attempt after its creation.
	if _, err := pool.Exec(ctx, `UPDATE stage_executions SET resume_requested_by=$1,resume_requested_at=clock_timestamp(),resume_transaction_id=pg_current_xact_id() WHERE stage_execution_id=$2`, before.OwnerID, source.StageExecutionID); persistencepostgres.SQLState(err) != persistencepostgres.SQLStateCheckViolation {
		t.Fatalf("late receipt attachment=%v", err)
	}
	result, err := store.ResumeFailedRun(ctx, before.OwnerID, source.RunID, source.StageExecutionID, "committed-target")
	if err != nil {
		t.Fatal(err)
	}
	for _, change := range []string{
		`resume_requested_by='other-owner'`,
		`resume_requested_at=resume_requested_at+interval '1 second'`,
		`resume_transaction_id=pg_current_xact_id()`,
		`resume_source_execution_id=NULL,resume_requested_by=NULL,resume_requested_at=NULL,resume_transaction_id=NULL`,
	} {
		if _, err := pool.Exec(ctx, `UPDATE stage_executions SET `+change+` WHERE stage_execution_id=$1`, result.StageExecutionID); persistencepostgres.SQLState(err) != persistencepostgres.SQLStateCheckViolation {
			t.Fatalf("receipt rewrite %s: %v", change, err)
		}
	}
	// The retained target remains preparing and its Run pending. Even in this
	// eligible-looking state, a new transaction cannot reuse its old receipt.
	service := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	if err := artifactstest.FreezeOutputs(ctx, service, source.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE artifact_bindings SET frozen=false WHERE scope_kind='run' AND scope_id=$1 AND namespace='outputs'`, source.RunID); persistencepostgres.SQLState(err) != persistencepostgres.SQLStateCheckViolation {
		t.Fatalf("old receipt permitted thaw: %v", err)
	}
}
