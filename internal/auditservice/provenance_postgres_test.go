package auditservice

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestAuditReadsRejectHistoricalProvenanceWithoutRewritingIt(t *testing.T) {
	ctx, pool, _, started, _ := newTimeControlAudit(t, 7200)
	store := auditstore.NewPostgresStore(pool)
	items, err := store.ListItems(ctx, started.Audit.AuditID)
	if err != nil || len(items) == 0 {
		t.Fatalf("current items = %+v, %v", items, err)
	}
	for _, kind := range []string{"item", "run"} {
		for _, shape := range []string{"current", "historical", "mixed-true", "mixed-false", "missing"} {
			t.Run(kind+"/"+shape, func(t *testing.T) {
				tx, err := pool.Begin(ctx)
				if err != nil {
					t.Fatal(err)
				}
				defer tx.Rollback(ctx)
				var document map[string]any
				if kind == "item" {
					encoded, err := json.Marshal(items[0].Origin)
					if err != nil || json.Unmarshal(encoded, &document) != nil {
						t.Fatal("cannot encode current origin")
					}
					document["entryKey"] = "historical"
				} else {
					document = map[string]any{
						"schema": "contractor.audit.run-provenance.v1", "runId": "deleted-run",
						"workflow": map[string]any{
							"name": "check", "version": "1", "schemaVersion": "contractor/v1alpha1",
							"configurationRef": map[string]string{"name": "check", "version": "1"},
							"closureDigest":    serviceTestDigest("check-workflow"),
						},
					}
				}
				switch shape {
				case "historical":
					if kind == "item" {
						document = map[string]any{"schema": auditstore.ItemOriginSchema, "entryKey": "historical"}
					} else {
						delete(document, "workflow")
					}
					document["provenanceIncomplete"] = true
				case "mixed-true", "mixed-false":
					document["provenanceIncomplete"] = shape == "mixed-true"
				case "missing":
					if kind == "item" {
						delete(document, "sourceRef")
					} else {
						delete(document, "workflow")
					}
				}
				encoded, err := json.Marshal(document)
				if err != nil {
					t.Fatal(err)
				}
				if kind == "item" {
					_, err = tx.Exec(ctx, `INSERT INTO audit_items (
item_id,audit_id,round_id,item_key,ordinal,kind,subject_key,task_ref,task_digest,origin,workflow_role,state)
SELECT 'historical-item',audit_id,round_id,'historical',10,kind,subject_key,task_ref,task_digest,$1::jsonb,workflow_role,'ready'
FROM audit_items WHERE item_id=$2`, encoded, items[0].ItemID)
				} else {
					_, err = tx.Exec(ctx, `INSERT INTO audit_executions (
execution_id,audit_id,round_id,role,workflow_role,manifest_ref,manifest_digest,
submission_key,request_digest,run_id,state,terminal_outcome,terminal_run_generation,
terminal_run_sequence,terminal_observed_at,run_provenance,run_deleted_at)
SELECT 'historical-execution',audit_id,round_id,'check',workflow_role,task_ref,task_digest,
'historical-submission',task_digest,'deleted-run','collected','succeeded','generation',1,
clock_timestamp(),$1::jsonb,clock_timestamp() FROM audit_items WHERE item_id=$2`, encoded, items[0].ItemID)
				}
				if err != nil {
					t.Fatal(err)
				}
				const snapshot = `SELECT jsonb_build_object(
'audit',(SELECT to_jsonb(a) FROM audits a WHERE audit_id='audit-time'),
'items',(SELECT jsonb_agg(to_jsonb(i) ORDER BY item_id) FROM audit_items i),
'executions',(SELECT jsonb_agg(to_jsonb(e) ORDER BY execution_id) FROM audit_executions e)
)::text`
				var before, after string
				if err := tx.QueryRow(ctx, snapshot).Scan(&before); err != nil {
					t.Fatal(err)
				}
				reader := auditstore.NewPostgresStore(tx)
				for range 2 {
					if kind == "item" {
						rows, readErr := reader.ListItems(ctx, started.Audit.AuditID)
						if shape == "current" {
							if readErr != nil || len(rows) != len(items)+1 {
								t.Fatalf("current origin read = %+v, %v", rows, readErr)
							}
						} else if readErr == nil || rows != nil {
							t.Fatalf("invalid origin read = %+v, %v", rows, readErr)
						}
					} else {
						rows, readErr := reader.ListExecutions(ctx, started.Audit.AuditID)
						if shape == "current" {
							if readErr != nil || len(rows) != 1 || rows[0].RunDeletedAt == nil || rows[0].RunProvenance.Workflow == nil {
								t.Fatalf("current deleted Run provenance = %+v, %v", rows, readErr)
							}
						} else if readErr == nil || rows != nil {
							t.Fatalf("invalid Run provenance read = %+v, %v", rows, readErr)
						}
					}
				}
				if err := tx.QueryRow(ctx, snapshot).Scan(&after); err != nil || before != after {
					t.Fatalf("provenance reads rewrote stored state: %v", err)
				}
			})
		}
	}
}
