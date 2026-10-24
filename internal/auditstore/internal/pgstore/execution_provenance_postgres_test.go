package pgstore

import (
	"context"
	"encoding/json"
	"testing"
)

// TestPostgresExecutionReadsRejectHistoricalRunProvenance stores a collected
// execution of a deleted Run whose provenance has each historical or mixed
// shape, and reads it through the store's execution read (getExecution,
// which every execution replay uses, and scanExecution behind every
// execution read). Only the current shape decodes; the others fail without
// the read rewriting what is stored.
func TestPostgresExecutionReadsRejectHistoricalRunProvenance(t *testing.T) {
	f := newCollectingAuditFixture(t, "historical-provenance", 1<<20)
	for _, shape := range []string{"current", "historical", "mixed-true", "mixed-false", "missing"} {
		t.Run(shape, func(t *testing.T) {
			tx, err := f.pool.Begin(f.ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = tx.Rollback(context.WithoutCancel(f.ctx)) }()
			document := map[string]any{
				"schema": "contractor.audit.run-provenance.v1", "runId": "deleted-run",
				"workflow": map[string]any{
					"name": "check", "version": "1", "schemaVersion": "contractor/v1alpha1",
					"configurationRef": map[string]string{"name": "check", "version": "1"},
					"closureDigest":    testDigest("8"),
				},
			}
			switch shape {
			case "historical":
				delete(document, "workflow")
				document["provenanceIncomplete"] = true
			case "mixed-true", "mixed-false":
				document["provenanceIncomplete"] = shape == "mixed-true"
			case "missing":
				delete(document, "workflow")
			}
			encoded, err := json.Marshal(document)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := tx.Exec(f.ctx, `
INSERT INTO audit_executions (
    lane_class, execution_id, audit_id, round_id, role, workflow_role, manifest_ref, manifest_digest,
    submission_key, request_digest, run_id, state, terminal_outcome, terminal_run_generation,
    terminal_run_sequence, terminal_observed_at, run_provenance, run_deleted_at)
SELECT 'model', 'historical-execution', audit_id, round_id, 'check', workflow_role, task_ref, task_digest,
       'historical-submission', task_digest, 'deleted-run', 'collected', 'succeeded', 'generation', 1,
       clock_timestamp(), $1::jsonb, clock_timestamp()
  FROM audit_items WHERE item_id = $2`, encoded, f.itemID); err != nil {
				t.Fatal(err)
			}
			const snapshot = `SELECT jsonb_agg(to_jsonb(execution) ORDER BY execution_id)::text
  FROM audit_executions AS execution WHERE audit_id = $1`
			var before, after string
			if err := tx.QueryRow(f.ctx, snapshot, f.audit.AuditID).Scan(&before); err != nil {
				t.Fatal(err)
			}
			for range 2 {
				execution, err := NewClaimStore(tx).getExecution(f.ctx, f.audit.AuditID, "historical-execution")
				if shape == "current" {
					if err != nil || execution.RunDeletedAt == nil || execution.RunProvenance == nil || execution.RunProvenance.Workflow == nil {
						t.Fatalf("current deleted Run provenance = %+v, %v", execution, err)
					}
				} else if err == nil {
					t.Fatalf("invalid Run provenance read = %+v", execution)
				}
			}
			if err := tx.QueryRow(f.ctx, snapshot, f.audit.AuditID).Scan(&after); err != nil || before != after {
				t.Fatalf("the provenance read rewrote stored state: %v", err)
			}
		})
	}
}
