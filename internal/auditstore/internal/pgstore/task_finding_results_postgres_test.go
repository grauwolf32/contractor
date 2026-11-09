package pgstore

import (
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"testing"
	"time"
)

func TestPostgresCanonicalTaskResultsKeepMembersAndEmptyResultsWithoutArtifacts(t *testing.T) {
	f := newCommonTaskFixture(t, 1)
	ctx := t.Context()
	accepted, _, err := f.owners.AcceptTasks(ctx, f.acceptance(2, "canonical-results"))
	if err != nil {
		t.Fatal(err)
	}
	claims := NewClaimStore(f.pool, artifactpgtest.NewStorage())
	owned, err := claims.Claim(ctx, auditstore.ClaimParams{HolderID: "finding-collector", Lease: time.Minute, Limit: 1})
	if err != nil || len(owned) != 1 {
		t.Fatalf("claim %v %v", owned, err)
	}
	for index, id := range accepted.TaskIDs {
		execution, _, err := claims.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: owned[0], TaskIDs: []string{id}})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := claims.ObserveSubmissionFailure(ctx, auditstore.ObserveSubmissionFailureParams{Claim: owned[0], ExecutionID: execution.ExecutionID}); err != nil {
			t.Fatal(err)
		}
		ids := []string{}
		if index == 0 {
			ids = []string{"finding-canonical"}
		}
		result := &auditstore.CanonicalFindingResult{IDs: ids}
		gap := "execution-failed"
		p := auditstore.CollectParams{Claim: owned[0], ExecutionID: execution.ExecutionID, ReceiptID: "collection-" + id, RequestDigest: contentdigest.Bytes([]byte(id)), Disposition: auditstore.CollectionExecutionFailed, Retained: []auditstore.ArtifactLink{}, Tasks: []auditstore.TaskCollection{{ExecutionItemID: auditdomain.DeterministicID("attempt", execution.ExecutionID, id), TaskID: id, Findings: result, GapCode: &gap}}}
		if _, changed, err := claims.Collect(ctx, p); err != nil || !changed {
			t.Fatalf("collect %t %v", changed, err)
		}
		if _, changed, err := claims.Collect(ctx, p); err != nil || changed {
			t.Fatalf("replay %t %v", changed, err)
		}
		stored, err := f.reads.GetTask(ctx, "owner", f.audit.AuditID, id)
		if err != nil || stored.Result != nil || stored.Findings == nil || len(stored.Findings.IDs) != len(ids) {
			t.Fatalf("Task %+v %v", stored, err)
		}
		encoded, _ := json.Marshal(result)
		if _, err := f.pool.Exec(ctx, `UPDATE audit_tasks SET finding_result=$2 WHERE task_id=$1`, id, encoded); err == nil {
			t.Fatal("settled canonical result mutable")
		}
	}
	var copies int
	if err := f.pool.QueryRow(ctx, `SELECT count(*) FROM audit_artifact_links WHERE audit_id=$1 AND logical_key LIKE 'execution/%'`, f.audit.AuditID).Scan(&copies); err != nil || copies != 0 {
		t.Fatalf("Artifact export=%d %v", copies, err)
	}
}
