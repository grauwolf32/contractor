package evalservice

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

func TestSharedPostgresEvalInventoryOnlyLabelsUniformTaskIterations(t *testing.T) {
	for _, tc := range []struct {
		name  string
		extra []int
		mixed bool
	}{
		{name: "single"},
		{name: "uniform", extra: []int{1}},
		{name: "mixed", extra: []int{2}, mixed: true},
		{name: "mixed-after-uniform-prefix", extra: []int{1, 2}, mixed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, e, memberID, claim, execution := evalAuditExecution(t)
			ctx := t.Context()
			// Seed membership directly to isolate the read contract from dispatch:
			// compatible Tasks can come from different iterations.
			for index, iteration := range tc.extra {
				taskID := fmt.Sprintf("inventory-task-%d", index)
				if _, err := h.pool.Exec(ctx, `INSERT INTO audit_tasks
(task_id,audit_id,source_key,source,document,batch_digest,ordinal,iteration,state)
SELECT $2,t.audit_id,$3,t.source,jsonb_set(t.document::jsonb,'{task_id}',to_jsonb($2::text)),
 t.batch_digest,t.ordinal+$4,$5,'selected'
FROM audit_execution_items m JOIN audit_tasks t ON t.task_id=m.task_id
WHERE m.execution_id=$1 AND m.batch_ordinal=0`, execution.ExecutionID, taskID,
					evaldomain.Digest([]byte(taskID)), index+1, iteration); err != nil {
					t.Fatal(err)
				}
				if _, err := h.pool.Exec(ctx, `INSERT INTO audit_execution_items
(execution_item_id,execution_id,audit_id,task_id,batch_ordinal,item_attempt,state)
VALUES ($1,$2,$3,$1,$4,1,'submitted')`, taskID, execution.ExecutionID, claim.AuditID, index+1); err != nil {
					t.Fatal(err)
				}
			}
			audit, err := h.audit.Get(ctx, e.OwnerID, claim.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			runID := "run-" + execution.ExecutionID
			if err := pg.InTx(ctx, h.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				if _, err := runstore.NewPostgresStore(tx, artifactpgtest.NewStorage()).CreateAuditRun(ctx, runstore.CreateAuditRunParams{
					CreateRunParams: runstore.CreateRunParams{
						LaneClass: resolved.LaneClassModel, RunID: runID, OwnerID: e.OwnerID, ProjectID: &audit.ProjectID,
						WorkflowName: "fixture", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion,
						WorkflowSnapshot: json.RawMessage(`{"name":"fixture"}`), Parameters: map[string]string{},
						RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
					}, AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey,
				}); err != nil {
					return err
				}
				_, err := auditstoretest.NewStore(tx, artifactpgtest.NewStorage()).BindRun(ctx, auditstore.BindRunParams{
					Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID,
				})
				return err
			}); err != nil {
				t.Fatal(err)
			}
			// An unresolved intent without members must not invent an iteration.
			if _, err := h.pool.Exec(ctx, `INSERT INTO audit_executions
(lane_class,execution_id,audit_id,workflow_role,submission_key,request_digest,state)
SELECT lane_class,'zz-empty',audit_id,workflow_role,'empty-key',request_digest,'intent'
FROM audit_executions WHERE execution_id=$1`, execution.ExecutionID); err != nil {
				t.Fatal(err)
			}

			check := func(t *testing.T, entries []evalstore.InventoryEntry) {
				t.Helper()
				if len(entries) != 3 {
					t.Fatalf("inventory entries = %+v", entries)
				}
				var seen int
				for _, entry := range entries {
					if entry.IntentID != execution.ExecutionID {
						if entry.Iteration != nil {
							t.Fatalf("non-member iteration = %d", *entry.Iteration)
						}
						continue
					}
					seen++
					if entry.Execution == nil || entry.Execution.ID != runID || !entry.Available || entry.Workflow == nil || *entry.Workflow != execution.WorkflowRole {
						t.Fatalf("lost execution association: %+v", entry)
					}
					if tc.mixed {
						if entry.Iteration != nil {
							t.Fatalf("mixed batch reported iteration %d", *entry.Iteration)
						}
					} else if entry.Iteration == nil || *entry.Iteration != 1 {
						t.Fatalf("uniform batch iteration = %v", entry.Iteration)
					}
				}
				if seen != 1 {
					t.Fatalf("matched execution %d times", seen)
				}
			}
			store := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage())
			inventory, err := store.Inventory(ctx, e.OwnerID, e.ID, memberID)
			if err != nil {
				t.Fatal(err)
			}
			t.Run("collection", func(t *testing.T) { check(t, inventory.Entries) })
			var entries []evalstore.InventoryEntry
			after := ""
			for range 4 {
				page, err := store.InventoryPage(ctx, e.OwnerID, e.ID, memberID, after, 1, nil)
				if err != nil {
					t.Fatal(err)
				}
				entries = append(entries, page.Items...)
				if !page.HasMore {
					break
				}
				after = page.LastKey
			}
			t.Run("paged", func(t *testing.T) { check(t, entries) })
		})
	}
}
