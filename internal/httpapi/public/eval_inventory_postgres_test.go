package public

import (
	"net/url"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func TestSharedEvalPostgresAuditInventoryIncludesTaskWorkflowsAndDeletedChildren(t *testing.T) {
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
	for i, workflow := range []string{"observe", "generate", "verify"} {
		id := "inventory-" + workflow
		var err error
		if workflow == "verify" {
			_, err = h.pool.Exec(t.Context(), `INSERT INTO audit_executions(lane_class,execution_id,audit_id,workflow_role,submission_key,request_digest,run_id,state,terminal_outcome,terminal_run_generation,terminal_run_sequence,terminal_observed_at,run_provenance,run_deleted_at)
VALUES('model',$1,$2,$3,$1,$4,'deleted-child','collected','failed','generation',1,clock_timestamp(),'{"schema":"contractor.audit.run-provenance.v1","runId":"deleted-child"}',clock_timestamp())`, id, audit, workflow, digest)
		} else {
			_, err = h.pool.Exec(t.Context(), `INSERT INTO audit_executions(lane_class,execution_id,audit_id,workflow_role,submission_key,request_digest) VALUES('model',$1,$2,$3,$1,$4)`, id, audit, workflow, digest)
		}
		if err != nil {
			t.Fatal(err)
		}
		if _, err = h.pool.Exec(t.Context(), `INSERT INTO audit_tasks(task_id,audit_id,source_key,source,document,batch_digest,ordinal,iteration)
VALUES($1,$2,'sha256:'||encode(sha256(convert_to($1,'UTF8')),'hex'),'{}',json_build_object('schema','contractor.audit.task.v1','task_id',$1::text,'audit_id',$2::text,'workflow',$3::text),$4,$5,$5)`, id, audit, workflow, digest, i+1); err != nil {
			t.Fatal(err)
		}
		if _, err = h.pool.Exec(t.Context(), `INSERT INTO audit_execution_items(execution_item_id,execution_id,audit_id,task_id,batch_ordinal,item_attempt) VALUES($1,$1,$2,$1,0,1)`, id, audit); err != nil {
			t.Fatal(err)
		}
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
		if entry.Workflow == nil || entry.Iteration == nil || entry.Parent == nil || entry.Parent.ID != audit {
			t.Fatal("association fields missing", entry)
		}
		roles[*entry.Workflow] = entry
		next = page.Page.NextCursor
	}
	if len(roles) != 3 || roles["verify"].Available || roles["verify"].Execution == nil || roles["verify"].Execution.ID != "deleted-child" || roles["observe"].Execution != nil {
		t.Fatal("common-workflow/unresolved/deleted children lost", roles)
	}
	_, err := h.pool.Exec(t.Context(), `INSERT INTO audit_executions(lane_class,execution_id,audit_id,workflow_role,submission_key,request_digest) VALUES('model','inventory-retry',$1,'observe','inventory-retry',$2)`, audit, digest)
	if err != nil {
		t.Fatal(err)
	}
	h.request(t, "GET", base+"&cursor="+url.QueryEscape(*first.Page.NextCursor), nil, "", "", 409)
	if _, err = h.service.Executions(t.Context(), "user-2", created.ID, member, "", 1, nil); !evaldomain.IsCode(err, "eval_not_found") {
		t.Fatal("foreign inventory", err)
	}
}
