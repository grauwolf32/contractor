package auditservice

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
)

// TestAuditReadsRejectHistoricalProvenanceWithoutRewritingIt reads item
// origins of each historical or mixed shape through the owner's item pages:
// only the current shape decodes, and the reads rewrite nothing. Execution
// Run provenance has no whole-Audit production read; the store's execution
// read is checked with the same shapes by pgstore's
// TestPostgresExecutionReadsRejectHistoricalRunProvenance.
func TestAuditReadsRejectHistoricalProvenanceWithoutRewritingIt(t *testing.T) {
	ctx, pool, _, started, _ := newTimeControlAudit(t, 7200)
	store := auditstoretest.NewStore(pool)
	items, err := store.Items(ctx, started.Audit.OwnerID, started.Audit.AuditID)
	if err != nil || len(items) == 0 {
		t.Fatalf("current items = %+v, %v", items, err)
	}
	for _, shape := range []string{"current", "historical", "mixed-true", "mixed-false", "missing"} {
		t.Run(shape, func(t *testing.T) {
			tx, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer tx.Rollback(ctx)
			var document map[string]any
			encoded, err := json.Marshal(items[0].Origin)
			if err != nil || json.Unmarshal(encoded, &document) != nil {
				t.Fatal("cannot encode current origin")
			}
			document["entryKey"] = "historical"
			switch shape {
			case "historical":
				document = map[string]any{"schema": auditstore.ItemOriginSchema, "entryKey": "historical"}
				document["provenanceIncomplete"] = true
			case "mixed-true", "mixed-false":
				document["provenanceIncomplete"] = shape == "mixed-true"
			case "missing":
				delete(document, "sourceRef")
			}
			encoded, err = json.Marshal(document)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := tx.Exec(ctx, `INSERT INTO audit_items (
item_id,audit_id,round_id,item_key,ordinal,kind,subject_key,task_ref,task_digest,origin,workflow_role,state,
coverage_status,coverage_requested,coverage_completed,coverage_gaps)
SELECT 'historical-item',audit_id,round_id,'historical',10,kind,subject_key,task_ref,task_digest,$1::jsonb,workflow_role,'ready',
coverage_status,coverage_requested,coverage_completed,coverage_gaps
FROM audit_items WHERE item_id=$2`, encoded, items[0].ItemID); err != nil {
				t.Fatal(err)
			}
			const snapshot = `SELECT jsonb_build_object(
'audit',(SELECT to_jsonb(a) FROM audits a WHERE audit_id='audit-time'),
'items',(SELECT jsonb_agg(to_jsonb(i) ORDER BY item_id) FROM audit_items i)
)::text`
			var before, after string
			if err := tx.QueryRow(ctx, snapshot).Scan(&before); err != nil {
				t.Fatal(err)
			}
			reader := auditstoretest.NewStore(tx)
			for range 2 {
				rows, readErr := reader.Items(ctx, started.Audit.OwnerID, started.Audit.AuditID)
				if shape == "current" {
					if readErr != nil || len(rows) != len(items)+1 {
						t.Fatalf("current origin read = %+v, %v", rows, readErr)
					}
				} else if readErr == nil || rows != nil {
					t.Fatalf("invalid origin read = %+v, %v", rows, readErr)
				}
			}
			if err := tx.QueryRow(ctx, snapshot).Scan(&after); err != nil || before != after {
				t.Fatalf("provenance reads rewrote stored state: %v", err)
			}
		})
	}
}
