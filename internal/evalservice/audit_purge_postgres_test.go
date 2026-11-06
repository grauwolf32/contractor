package evalservice

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/findingintake"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/contracts"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestSharedPostgresEvalAuditWithDecidedHistoryPurges drains an Eval-managed Audit
// that holds a collected check result through the
// ordinary purge, which must also record the Eval execution tombstone.
func TestSharedPostgresEvalAuditWithDecidedHistoryPurges(t *testing.T) {
	h := newHarness(t)
	e := h.prepared(t, "audit")
	h.command(t, e, "start")
	c := h.coordinator(t, "controller")
	tick(t, c)
	tick(t, c)
	var auditID string
	if err := h.pool.QueryRow(t.Context(), `SELECT audit_id FROM audits`).Scan(&auditID); err != nil {
		t.Fatal("Eval-managed Audit was not created", err)
	}
	audit, err := h.audit.Get(t.Context(), h.scope.OwnerID, auditID)
	if err != nil {
		t.Fatal(err)
	}
	seedEvalAuditDecidedHistory(t, h.pool, audit)
	e = h.get(t, e.ID)
	h.command(t, e, "cancel")
	tick(t, c)
	store := auditstoretest.NewStore(h.pool, artifactpgtest.NewStorage()).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
	claims, err := store.Claim(t.Context(), auditstore.ClaimParams{HolderID: "audit-controller", Lease: time.Minute, Limit: 10})
	if err != nil || len(claims) != 1 || claims[0].AuditID != auditID {
		t.Fatal(claims, err)
	}
	if err = store.PurgeClaimed(t.Context(), claims[0], auditdomain.ArtifactNamespace(auditID)); err != nil {
		t.Fatalf("purge Eval-managed Audit with decided history: %v", err)
	}
	tick(t, c)
	e = h.get(t, e.ID)
	if e.State != "cancelled" || e.Outstanding != 0 || count(t, h.pool, "audits") != 0 || count(t, h.pool, "audit_execution_items") != 0 ||
		count(t, h.pool, "eval_execution_tombstones") != 1 {
		t.Fatal("Eval-managed Audit with decided history failed to drain", e.State)
	}
}

// seedEvalAuditDecidedHistory records the rows an Audit keeps after its child
// Run is deleted: a collected check attempt.
// Project Finding state is independent of the execution rows being purged.
func seedEvalAuditDecidedHistory(t *testing.T, pool *pgxpool.Pool, audit auditstore.Audit) {
	t.Helper()
	digest := "sha256:" + strings.Repeat("d", 64)
	workflow := &auditstore.WorkflowProvenance{
		Name: "check", Version: "1", SchemaVersion: contracts.APIVersion, ClosureDigest: digest,
	}
	workflow.ConfigurationRef.Name, workflow.ConfigurationRef.Version = workflow.Name, workflow.Version
	runProvenance, err := json.Marshal(auditstore.RunProvenance{
		Schema: "contractor.audit.run-provenance.v1", RunID: "deleted-run-decided", Workflow: workflow,
	})
	if err != nil {
		t.Fatal(err)
	}
	resultRef := `{"namespace":"audit-results","name":"check-decided","revision":"result-r1"}`
	statements := []struct {
		sql  string
		args []any
	}{
		{`INSERT INTO audit_tasks(task_id,audit_id,source_key,source,document,batch_digest,ordinal,iteration)
VALUES('task-decided',$1,$2,'{}',json_build_object('schema','contractor.audit.task.v1','task_id','task-decided','audit_id',$1::text,'workflow','check'),$2,1,1)`, []any{audit.AuditID, digest}},
		{`
INSERT INTO audit_executions (
    lane_class, execution_id, audit_id, workflow_role,
    submission_key, request_digest, run_id, state, terminal_outcome,
    terminal_run_generation, terminal_run_sequence, terminal_observed_at,
    run_provenance, run_deleted_at
) VALUES ('model', 'execution-decided', $1, 'check-role', 'submission-decided', $2, 'deleted-run-decided', 'collected', 'succeeded',
          'generation-one', 1, clock_timestamp(), $3::jsonb, clock_timestamp())`,
			[]any{audit.AuditID, digest, runProvenance}},
		{`INSERT INTO audit_execution_items(execution_item_id,execution_id,audit_id,task_id,batch_ordinal,item_attempt,state,collection_disposition,result_ref,result_digest,collected_at)
VALUES('execution-item-decided','execution-decided',$1,'task-decided',0,1,'settled','accepted-result',$2::jsonb,$3,clock_timestamp())`, []any{audit.AuditID, resultRef, digest}},
		{`
UPDATE audit_executions
   SET collection_receipt_id = 'collection-decided', collection_disposition = 'accepted-result',
       collection_source_output_ref = $2::jsonb, collection_source_output_digest = $3,
       collection_retained_refs = '[]'::jsonb, collection_request_digest = $3,
       collected_at = clock_timestamp()
 WHERE audit_id = $1 AND execution_id = 'execution-decided'`,
			[]any{audit.AuditID, resultRef, digest}},
		{`UPDATE audit_tasks SET state='settled',last_execution_id='execution-decided',attempts=1,result=$2::json
WHERE audit_id=$1 AND task_id='task-decided'`, []any{audit.AuditID, `{"scope_kind":"audit","scope_id":"` + audit.AuditID + `","ref":` + resultRef + `,"digest":"` + digest + `","media_type":"application/json","size_bytes":1}`}},
	}
	err = pg.InTx(t.Context(), pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, statement := range statements {
			if _, err := tx.Exec(t.Context(), statement.sql, statement.args...); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
