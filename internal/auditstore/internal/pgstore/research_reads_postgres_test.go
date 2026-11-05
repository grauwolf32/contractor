package pgstore

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
)

func TestSharedPostgresResearchMaterialsTraverseLongKeys(t *testing.T) {
	f := newCommonTaskFixture(t, 1)
	ctx := t.Context()
	ref, _ := json.Marshal(f.material.Ref)
	// A real containing collection; every accepted alias addresses frozen bytes.
	if _, err := f.pool.Exec(ctx, `INSERT INTO audit_artifact_links(audit_id,artifact_scope,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance)
 SELECT $1,'audit',lpad(n::text,512,'x'),$2,$3,$4,$5,'{}' FROM generate_series(1,5000) AS n`, f.audit.AuditID, ref, f.material.Digest, f.material.MediaType, f.material.SizeBytes); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(ctx, `ANALYZE audit_artifact_links`); err != nil {
		t.Fatal(err)
	}
	p := auditstore.MaterialListParams{OwnerID: "owner", AuditID: f.audit.AuditID, Limit: 200}
	seen := map[string]bool{}
	for {
		page, err := f.reads.ListMaterials(ctx, p)
		if err != nil {
			t.Fatal(err)
		}
		for _, item := range page.Items {
			if len(item.Key) != 512 || seen[item.Key] || item.Material.Digest != f.material.Digest || item.Material.ScopeID != f.audit.AuditID {
				t.Fatalf("invalid page member: %+v", item)
			}
			seen[item.Key] = true
		}
		if len(page.Items) > p.Limit {
			t.Fatal("row budget exceeded")
		}
		if page.NextOrdinal == nil {
			break
		}
		if *page.NextOrdinal <= p.AfterOrdinal {
			t.Fatal("cursor did not advance")
		}
		p.AfterOrdinal = *page.NextOrdinal
		p.AuditRevision = &page.AuditRevision
	}
	if len(seen) != 5000 {
		t.Fatalf("traversed %d materials", len(seen))
	}
	for _, mode := range explaintest.Modes {
		plan := explainPrepared(t, ctx, f.pool, mode, listResearchMaterialsSQL, f.audit.AuditID, int64(2500), 11)
		if n := plan.Reads.Of("audit_artifact_links"); n > 32 {
			t.Fatalf("%s read %d rows: %s", mode, n, plan.Plan.String())
		}
	}
	p.OwnerID = "foreign"
	if _, err := f.reads.ListMaterials(ctx, p); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign owner: %v", err)
	}
	p.OwnerID = "owner"
	stale := uint64(1)
	p.AuditRevision = &stale
	if _, err := f.reads.ListMaterials(ctx, p); !errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("stale revision: %v", err)
	}
	p.AuditRevision = nil
	p.Limit = 201
	if _, err := f.reads.ListMaterials(ctx, p); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("oversized page: %v", err)
	}
}

func TestSharedPostgresTaskAttemptReadAndCoordination(t *testing.T) {
	f := newCommonTaskFixture(t, 1)
	ctx := t.Context()
	accepted, _, err := f.owners.AcceptTasks(ctx, f.acceptance(2, "accept"))
	if err != nil {
		t.Fatal(err)
	}
	claims := NewClaimStore(f.pool, artifactpgtest.NewStorage())
	claimed, err := claims.Claim(ctx, auditstore.ClaimParams{HolderID: "reader", Lease: time.Minute, Limit: 1})
	if err != nil || len(claimed) != 1 {
		t.Fatalf("claim: %+v %v", claimed, err)
	}
	execution, _, err := claims.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claimed[0], TaskIDs: accepted.TaskIDs[:1]})
	if err != nil {
		t.Fatal(err)
	}
	p := auditstore.TaskAttemptListParams{OwnerID: "owner", AuditID: f.audit.AuditID, TaskID: accepted.TaskIDs[0], Limit: 1}
	page, err := f.reads.ListTaskAttempts(ctx, p)
	if err != nil || len(page.Items) != 1 || page.Items[0].ExecutionID != execution.ExecutionID || page.Items[0].Attempt != 1 || page.Items[0].RunID != nil {
		t.Fatalf("attempt: %+v %v", page, err)
	}
	p.TaskID = accepted.TaskIDs[1]
	if page, err := f.reads.ListTaskAttempts(ctx, p); err != nil || len(page.Items) != 0 {
		t.Fatalf("unselected Task: %+v %v", page, err)
	}
	p.TaskID = "foreign-task"
	if _, err := f.reads.ListTaskAttempts(ctx, p); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign task: %v", err)
	}
	p.TaskID = accepted.TaskIDs[0]
	p.OwnerID = "foreign"
	if _, err := f.reads.ListTaskAttempts(ctx, p); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign owner: %v", err)
	}
	summary, err := f.reads.GetWorkspace(ctx, "owner", f.audit.AuditID)
	if err != nil || summary.TotalTasks != 2 || summary.OutstandingRuns != 1 || summary.SubmittedRuns != 0 {
		t.Fatalf("coordination: %+v %v", summary, err)
	}
	data, _ := json.Marshal(summary)
	for _, old := range []string{"roundId", "completedChecks", "issues", "gaps"} {
		if strings.Contains(string(data), fmt.Sprintf("%q:", old)) {
			t.Fatalf("legacy semantics: %s", data)
		}
	}
}
