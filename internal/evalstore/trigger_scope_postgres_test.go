package evalstore

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// evalTestRuntimeSnapshot is the Runtime configuration of the Runs these
// tests insert directly.
const evalTestRuntimeSnapshot = `{"default":{"label":"default","explicit":false,"bindingRevision":1,"config":{"name":"contractor-empty","version":"1","digest":"sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f"}},"labels":[],"llmCredentialIds":[],"runtimeCredentialIds":[]}`

// TestSharedPostgresEvalProjectionFollowsOnlyRunOutputChanges checks that
// the Eval output projection keeps every effect after its WHEN conditions
// (PER-R22): creating a Run output, changing its revision, freezing it and
// deleting it mark the member that ran the Run dirty, while writing another
// namespace of the Run, touching only an output's update time and freezing a
// frozen output do not. Deleting the member's Run still calls the retention
// trigger; deleting a Run that no Eval suboperation can have created does not.
func TestSharedPostgresEvalProjectionFollowsOnlyRunOutputChanges(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	experiment := createExperiment(t, pool, scope, "exp", "trace-1", "external-workflow")
	member := members(t, pool, experiment)[0]
	if _, err := admit(t, pool, experiment, member.MemberID, nil, "submit"); err != nil {
		t.Fatal(err)
	}
	claim := oneClaim(t, pool)
	operation := Suboperation{Kind: "run-create", Key: "run-create-key", Request: json.RawMessage(`{"workflow":"trace-a@1"}`)}
	mustTx(t, pool, func(s *Store) error {
		return s.RecoverMember(scope, experiment.ID, member.MemberID, claim).PutSuboperation(ctx, operation)
	})
	const runID = "run-eval-outputs"
	if _, err := pool.Exec(ctx, `
INSERT INTO workflow_runs (run_id, owner_id, project_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, runtime_labels, runtime_config_snapshot, scheduler_lane_class,
    state, state_reason_code, finished_at, request_idempotency_key, request_digest)
VALUES ($1, $2, $3, 'trace', '1', '1', '{}', '{}', $4, 'model', 'failed', 'test_failure', clock_timestamp(), $5, $6)`,
		runID, scope.OwnerID, scope.ProjectID, evalTestRuntimeSnapshot, operation.Key, evaldomain.Digest(operation.Request)); err != nil {
		t.Fatal(err)
	}
	mustTx(t, pool, func(s *Store) error {
		return s.RecoverMember(scope, experiment.ID, member.MemberID, claim).BindExecution(ctx, runID)
	})
	revision := func(db pg.DBTX) int64 {
		t.Helper()
		var value int64
		if err := db.QueryRow(ctx, `SELECT revision FROM eval_member_projections
WHERE experiment_id = $1 AND member_id = $2`, experiment.ID, member.MemberID).Scan(&value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	expectDirty := func(step string, db pg.DBTX, dirty bool, action func() error) {
		t.Helper()
		before := revision(db)
		if err := action(); err != nil {
			t.Fatalf("%s: %v", step, err)
		}
		if changed := revision(db) != before; changed != dirty {
			t.Errorf("%s marked the member dirty = %t, want %t", step, changed, dirty)
		}
	}
	// inRolledBackTx runs action in a transaction that it rolls back, so the
	// deferred revision key of a binding is never checked.
	inRolledBackTx := func(step string, dirty bool, action func(pgx.Tx) error) {
		t.Helper()
		tx, err := pool.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = tx.Rollback(ctx) }()
		expectDirty(step, tx, dirty, func() error { return action(tx) })
	}
	service := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	run, _ := service.Run(runID)
	var written artifacts.WriteResult
	expectDirty("intermediate write", pool, false, func() error {
		var err error
		written, err = run.Write(ctx, artifacts.ArtifactRef{Namespace: "scratch", Name: "result"},
			artifacts.Payload{MediaType: "text/plain", Data: []byte("result")}, nil)
		return err
	})
	expectDirty("output binding", pool, true, func() error {
		_, err := artifactstest.BindOutput(ctx, service, runID, "result", written.Ref)
		return err
	})
	inRolledBackTx("output revision", true, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, `UPDATE artifact_bindings SET current_revision = 'rev-elsewhere'
WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = 'outputs'`, runID)
		return err
	})
	inRolledBackTx("output update time", false, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, `UPDATE artifact_bindings SET updated_at = clock_timestamp()
WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = 'outputs'`, runID)
		return err
	})
	expectDirty("output freeze", pool, true, func() error { return artifactstest.FreezeOutputs(ctx, service, runID) })
	expectDirty("repeated freeze", pool, false, func() error { return artifactstest.FreezeOutputs(ctx, service, runID) })
	inRolledBackTx("Run purge", true, func(tx pgx.Tx) error {
		purger, err := artifactpgtest.NewStorage().Purger(tx)
		if err != nil {
			return err
		}
		if err := purger.PurgeRun(ctx, runID); err != nil {
			return err
		}
		// The member's Run is matched to its suboperation when deleted.
		deletion, err := explaintest.Explain(ctx, tx, `DELETE FROM workflow_runs WHERE run_id = $1`, runID)
		if err != nil {
			return err
		}
		if calls := deletion.Triggers[explaintest.Trigger{Relation: "workflow_runs", Name: "workflow_runs_eval_tombstone"}]; calls != 1 {
			t.Errorf("deleting the member's Run called the retention trigger %d times, want 1", calls)
		}
		return nil
	})
	// A Run without a Project and request key cannot be an Eval execution.
	if _, err := pool.Exec(ctx, `
INSERT INTO workflow_runs (run_id, owner_id, workflow_name, workflow_version, workflow_schema_version,
    workflow_snapshot, runtime_labels, runtime_config_snapshot, scheduler_lane_class, state, state_reason_code, finished_at)
VALUES ('run-standalone', $1, 'trace', '1', '1', '{}', '{}', $2, 'model', 'failed', 'test_failure', clock_timestamp())`,
		scope.OwnerID, evalTestRuntimeSnapshot); err != nil {
		t.Fatal(err)
	}
	deletion, err := explaintest.Explain(ctx, pool, `DELETE FROM workflow_runs WHERE run_id = 'run-standalone'`)
	if err != nil {
		t.Fatal(err)
	}
	if calls := deletion.Triggers[explaintest.Trigger{Relation: "workflow_runs", Name: "workflow_runs_eval_tombstone"}]; calls != 0 {
		t.Errorf("deleting a standalone Run called the retention trigger %d times, want none", calls)
	}
}
