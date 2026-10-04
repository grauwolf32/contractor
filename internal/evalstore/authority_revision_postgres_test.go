package evalstore

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestPostgresEvalCoordinatorObservationsKeepAuthorityRevision(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	putDataset(t, pool, scope)
	experiment := createExperiment(t, pool, scope, "exp", "trace-1", "create-workflow")
	prepare := evaldomain.Command{Kind: "prepare"}
	prepareDoc := freeze(t, "Command", prepare)
	mustTx(t, pool, func(s *Store) error {
		_, err := s.Command(ctx, CommandParams{Scope: scope, ExperimentID: experiment.ID,
			CommandID: "prepare-command", Command: prepare,
			Mutation: mutation(t, "prepare", experiment.Revision, prepareDoc)})
		return err
	})
	beforePreparation, err := NewPostgresStore(pool).Get(ctx, scope.OwnerID, experiment.ID)
	if err != nil {
		t.Fatal(err)
	}
	claim := oneClaim(t, pool)
	var registration evaldomain.ExternalRegistration
	if err := json.Unmarshal(fixture(t, "registration", "ExternalRegistration").Bytes(), &registration); err != nil {
		t.Fatal(err)
	}
	setup := bytesOf(map[string]any{
		"variants": registration.Variants, "checks": registration.Checks,
		"comparison": registration.Comparison, "budgets": registration.Budgets,
	})
	cases := make(map[string]evaldomain.Case, len(registration.Recipes))
	for _, recipe := range registration.Recipes {
		cases[recipe.MemberID] = recipe.Case
	}
	plan := fixture(t, "portable-plan", "playground.plan/v1")
	mustTx(t, pool, func(s *Store) error {
		return s.FreezePrepared(ctx, scope, experiment.ID, claim, plan, setup, cases)
	})
	prepared, err := NewPostgresStore(pool).Get(ctx, scope.OwnerID, experiment.ID)
	if err != nil || prepared.Revision != beforePreparation.Revision {
		t.Fatalf("preparation changed authority revision: before=%+v after=%+v err=%v", beforePreparation, prepared, err)
	}
	start := evaldomain.Command{Kind: "start", PlanSHA256: plan.Digest()}
	startDoc := freeze(t, "Command", start)
	mustTx(t, pool, func(s *Store) error {
		_, err := s.Command(ctx, CommandParams{Scope: scope, ExperimentID: experiment.ID,
			CommandID: "start-command", Command: start,
			Mutation: mutation(t, "start", prepared.Revision, startDoc)})
		return err
	})
	running, err := NewPostgresStore(pool).Get(ctx, scope.OwnerID, experiment.ID)
	if err != nil || running.Revision != prepared.Revision+1 {
		t.Fatalf("start authority revision = %+v (%v)", running, err)
	}

	var result evaldomain.ResultInput
	if err := json.Unmarshal(fixture(t, "result", "ResultInput").Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	result.PlanSHA256 = plan.Digest()
	resultDoc := freeze(t, "ResultInput", result)
	if receipt, err := admit(t, pool, running, result.MemberID, &claim, "submit"); err != nil {
		t.Fatal(err)
	} else if admitted, err := receipt.Submission(); err != nil || admitted.ExperimentRevision != running.Revision {
		t.Fatalf("admission receipt consumed authority CAS: %+v (%v)", admitted, err)
	}
	mustTx(t, pool, func(s *Store) error {
		return s.ObserveTokens(ctx, scope, experiment.ID, result.MemberID, claim, 7)
	})
	mustTx(t, pool, func(s *Store) error {
		return s.SelectFirstNative(ctx, scope, experiment.ID, result.MemberID, claim, resultDoc, evaldomain.Frozen{})
	})
	var selected int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM eval_selections
WHERE experiment_id=$1 AND member_id=$2`, experiment.ID, result.MemberID).Scan(&selected); err != nil || selected != 1 {
		t.Fatalf("native selections = %d (%v), want 1", selected, err)
	}
	operation := Suboperation{Kind: "run-create", Key: "rejected-run", Request: json.RawMessage(`{"workflow":"trace-a@1"}`)}
	mustTx(t, pool, func(s *Store) error {
		return s.PutSuboperation(ctx, scope, experiment.ID, result.MemberID, claim, operation)
	})
	mustTx(t, pool, func(s *Store) error {
		return s.ResolveSuboperation(ctx, scope, experiment.ID, result.MemberID, operation.Kind, claim, []byte(`{"rejected":true}`), true)
	})
	mustTx(t, pool, func(s *Store) error {
		return s.Settle(ctx, scope, experiment.ID, result.MemberID, claim)
	})
	observed, err := NewPostgresStore(pool).Get(ctx, scope.OwnerID, experiment.ID)
	if err != nil || observed.Revision != running.Revision || observed.Outstanding != 0 || observed.ObservedTokens != 7 {
		t.Fatalf("coordinator observations consumed authority CAS: %+v (%v)", observed, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE eval_experiments SET name='tampered',
    updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond')
WHERE experiment_id=$1`, experiment.ID); err == nil {
		t.Fatal("database accepted an authoring edit without advancing authority revision")
	}
	pause := evaldomain.Command{Kind: "pause", PlanSHA256: plan.Digest()}
	pauseDoc := freeze(t, "Command", pause)
	mustTx(t, pool, func(s *Store) error {
		_, err := s.Command(ctx, CommandParams{Scope: scope, ExperimentID: experiment.ID,
			CommandID: "pause-command", Command: pause,
			Mutation: mutation(t, "pause", running.Revision, pauseDoc)})
		return err
	})
	mustTx(t, pool, func(s *Store) error {
		return s.Transition(ctx, scope, experiment.ID, claim, evaldomain.StatePausing, evaldomain.StatePaused, 7, nil)
	})
	paused, err := NewPostgresStore(pool).Get(ctx, scope.OwnerID, experiment.ID)
	if err != nil || paused.Revision != running.Revision+1 || paused.State != evaldomain.StatePaused {
		t.Fatalf("pause/transition authority revision = %+v (%v)", paused, err)
	}
	cancel := evaldomain.Command{Kind: "cancel", PlanSHA256: plan.Digest()}
	cancelDoc := freeze(t, "Command", cancel)
	stale := transaction(pool, func(s *Store) error {
		_, err := s.Command(ctx, CommandParams{Scope: scope, ExperimentID: experiment.ID,
			CommandID: "stale-cancel", Command: cancel,
			Mutation: mutation(t, "stale-cancel", running.Revision, cancelDoc)})
		return err
	})
	if !code(stale, "eval_revision_mismatch") {
		t.Fatalf("cancel after authority command = %v, want 412", stale)
	}
	mustTx(t, pool, func(s *Store) error {
		_, err := s.Command(ctx, CommandParams{Scope: scope, ExperimentID: experiment.ID,
			CommandID: "cancel-command", Command: cancel,
			Mutation: mutation(t, "cancel", paused.Revision, cancelDoc)})
		return err
	})
	cancelled, err := NewPostgresStore(pool).Get(ctx, scope.OwnerID, experiment.ID)
	if err != nil || cancelled.Revision != paused.Revision+1 || cancelled.State != evaldomain.StateCancelling {
		t.Fatalf("cancel with authority revision = %+v (%v)", cancelled, err)
	}
}

func TestPostgresEvalExecutionTombstoneKeepsAuthorityRevision(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	createDoc := fixture(t, "external-workflow", "CreateExperiment")
	var created Receipt
	mustTx(t, pool, func(s *Store) error {
		var err error
		created, err = s.Create(ctx, CreateParams{Scope: scope, ID: "exp", PortableID: "trace-1",
			Document: createDoc, Mutation: mutation(t, "create", 0, createDoc)})
		return err
	})
	experiment, err := NewPostgresStore(pool).Get(ctx, scope.OwnerID, "exp")
	if err != nil {
		t.Fatal(err)
	}
	createdResponse, err := created.Experiment()
	if err != nil || createdResponse.Revision != experiment.Revision {
		t.Fatalf("external registration receipt revision = %+v, stored=%d (%v)", createdResponse, experiment.Revision, err)
	}
	member := members(t, pool, experiment)[0]
	if _, err := admit(t, pool, experiment, member.MemberID, nil, "submit"); err != nil {
		t.Fatal(err)
	}
	claim := oneClaim(t, pool)
	operation := Suboperation{Kind: "run-create", Key: "run-create-key", Request: json.RawMessage(`{"workflow":"trace-a@1"}`)}
	mustTx(t, pool, func(s *Store) error {
		return s.PutSuboperation(ctx, scope, experiment.ID, member.MemberID, claim, operation)
	})
	const runtimeSnapshot = `{"default":{"label":"default","explicit":false,"bindingRevision":1,"config":{"name":"contractor-empty","version":"1","digest":"sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f"}},"labels":[],"llmCredentialIds":[],"runtimeCredentialIds":[]}`
	_, err = pool.Exec(ctx, `
INSERT INTO workflow_runs (run_id,owner_id,project_id,workflow_name,workflow_version,
    workflow_schema_version,workflow_snapshot,runtime_labels,runtime_config_snapshot,
    state,state_reason_code,finished_at,request_idempotency_key,request_digest)
VALUES ('run-to-delete',$1,$2,'trace','1','1','{}','{}',$3,'failed','test_failure',clock_timestamp(),$4,$5)
`, scope.OwnerID, scope.ProjectID, runtimeSnapshot, operation.Key, evaldomain.Digest(operation.Request))
	if err != nil {
		t.Fatal(err)
	}
	before, err := NewPostgresStore(pool).Get(ctx, scope.OwnerID, experiment.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `DELETE FROM workflow_runs WHERE run_id='run-to-delete'`); err != nil {
		t.Fatal(err)
	}
	after, err := NewPostgresStore(pool).Get(ctx, scope.OwnerID, experiment.ID)
	if err != nil || after.Revision != before.Revision || after.ViewGeneration != before.ViewGeneration+1 || !after.UpdatedAt.After(before.UpdatedAt) {
		t.Fatalf("execution tombstone changed authority CAS: before=%+v after=%+v err=%v", before, after, err)
	}
	if _, err := NewPostgresStore(pool).Tombstone(ctx, scope.OwnerID, experiment.ID, member.MemberID); err != nil {
		t.Fatal(err)
	}
}
