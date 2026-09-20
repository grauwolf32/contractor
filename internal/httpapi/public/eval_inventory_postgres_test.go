package public

import (
	"fmt"
	"net/url"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func TestEvalPostgresAuditInventoryIncludesNonItemRolesAndDeletedChildren(t *testing.T) {
	h := newEvalAPIHarness(t)
	draft, _ := h.dataset(t, "audit")
	created := apiDecode[struct {
		ID string `json:"experimentId"`
	}](t, h.request(t, "POST", "/v1/projects/evaluation/eval-experiments", evaldomain.CreateExperiment{Name: "Audit inventory", ControlMode: "server", Draft: &draft}, "create", "", 201))
	h.command(t, created.ID, "prepare")
	h.tick(t)
	h.command(t, created.ID, "start")
	h.tick(t)
	h.tick(t)
	var member, audit string
	if err := h.pool.QueryRow(t.Context(), `SELECT member_id,execution_id FROM eval_submissions WHERE experiment_id=$1 AND execution_id IS NOT NULL LIMIT 1`, created.ID).Scan(&member, &audit); err != nil {
		t.Fatal(err)
	}
	digest := evaldomain.Digest([]byte("inventory fixture"))
	// Seed retained association facts, including a deleted Run tombstone. These
	// are read-model fixtures; Run/Audit admission itself is tested separately.
	for i := 1; i <= 2; i++ {
		_, err := h.pool.Exec(t.Context(), `INSERT INTO audit_rounds(round_id,audit_id,ordinal,manifest_ref,manifest_digest,state,expected_item_count) SELECT $1,$2,COALESCE(max(ordinal),0)+1,'{}'::jsonb,$3,'closed',0 FROM audit_rounds WHERE audit_id=$2`, fmt.Sprintf("inventory-round-%d", i), audit, digest)
		if err != nil {
			t.Fatal(err)
		}
	}
	for i, role := range []string{"discovery", "assessment"} {
		_, err := h.pool.Exec(t.Context(), `INSERT INTO audit_executions(execution_id,audit_id,round_id,role,workflow_role,role_attempt,manifest_ref,manifest_digest,submission_key,request_digest) VALUES($1,$2,$3,$4,$4,$5,'{}'::jsonb,$6,$1,$6)`, "inventory-"+role, audit, fmt.Sprintf("inventory-round-%d", i+1), role, i+1, digest)
		if err != nil {
			t.Fatal(err)
		}
	}
	_, err := h.pool.Exec(t.Context(), `
INSERT INTO audit_executions(execution_id,audit_id,round_id,role,workflow_role,manifest_ref,manifest_digest,submission_key,request_digest,run_id,state,terminal_outcome,terminal_run_generation,terminal_run_sequence,terminal_observed_at,run_provenance,run_deleted_at)
VALUES('inventory-deleted',$1,'inventory-round-1','check','check','{}'::jsonb,$2,'inventory-deleted',$2,'deleted-child','collected','failed','generation',1,clock_timestamp(),'{"schema":"contractor.audit.run-provenance.v1","runId":"deleted-child"}'::jsonb,clock_timestamp())`, audit, digest)
	if err != nil {
		t.Fatal(err)
	}
	base := "/v1/eval-experiments/" + created.ID + "/members/" + member + "/executions?limit=1"
	type inventoryPage struct {
		Revision int64                      `json:"inventoryRevision"`
		Complete bool                       `json:"inventoryComplete"`
		Items    []evalstore.InventoryEntry `json:"items"`
		Gaps     []string                   `json:"gaps"`
		Page     evalPageInfo               `json:"page"`
	}
	first := apiDecode[inventoryPage](t, h.request(t, "GET", base, nil, "", "", 200))
	if first.Complete || len(first.Gaps) == 0 || first.Page.NextCursor == nil || len(first.Items) != 1 || first.Items[0].Execution.Kind != "audit" {
		t.Fatal("parent inventory", first)
	}
	roles := map[string]evalstore.InventoryEntry{}
	next := first.Page.NextCursor
	for next != nil {
		page := apiDecode[inventoryPage](t, h.request(t, "GET", base+"&cursor="+url.QueryEscape(*next), nil, "", "", 200))
		if page.Revision != first.Revision || len(page.Items) != 1 || page.Complete {
			t.Fatal("inventory pagination changed context")
		}
		entry := page.Items[0]
		if entry.Role == nil || entry.Round == nil || entry.Parent == nil || entry.Parent.ID != audit {
			t.Fatal("association fields missing", entry)
		}
		roles[*entry.Role] = entry
		next = page.Page.NextCursor
	}
	if len(roles) != 3 || roles["check"].Available || roles["check"].Execution == nil || roles["check"].Execution.ID != "deleted-child" || roles["discovery"].Execution != nil {
		t.Fatal("non-item/unresolved/deleted children lost", roles)
	}
	_, err = h.pool.Exec(t.Context(), `INSERT INTO audit_executions(execution_id,audit_id,role,workflow_role,role_attempt,manifest_ref,manifest_digest,submission_key,request_digest) VALUES('inventory-retry',$1,'discovery','discovery',2,'{}'::jsonb,$2,'inventory-retry',$2)`, audit, digest)
	if err != nil {
		t.Fatal(err)
	}
	h.request(t, "GET", base+"&cursor="+url.QueryEscape(*first.Page.NextCursor), nil, "", "", 409)
	if _, err = h.service.Executions(t.Context(), "user-2", created.ID, member, "", 1, nil); !evaldomain.IsCode(err, "eval_not_found") {
		t.Fatal("foreign inventory", err)
	}
}
