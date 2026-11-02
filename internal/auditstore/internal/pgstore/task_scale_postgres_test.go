package pgstore

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func TestSharedPostgresCommonTaskPagesAndPurgeAtCeiling(t *testing.T) {
	f := newCommonTaskFixture(t, 1)
	ctx := t.Context()
	base := f.acceptance(1, "seed").Tasks[0]
	batch, err := taskBatchDigest(base.Document)
	if err != nil {
		t.Fatal(err)
	}
	// Insert fixture history in finite blocks. A query budget is measured on
	// newly inserted rows, before any Task update can leave obsolete versions.
	const total = 100000
	for start := 1; start <= total; start += 1000 {
		rows := make([][]any, 0, 1000)
		for ordinal := start; ordinal < start+1000; ordinal++ {
			entry := base
			entry.Document.TaskID = fmt.Sprintf("task-%06d", ordinal)
			entry.Source.Key = entry.Document.TaskID
			document, err := strictjson.Canonical(entry.Document)
			if err != nil {
				t.Fatal(err)
			}
			source, err := strictjson.Canonical(entry.Source)
			if err != nil {
				t.Fatal(err)
			}
			state := "settled"
			var gap any = "fixture-complete"
			if ordinal > total-3 {
				state = "ready"
				gap = nil
			}
			rows = append(rows, []any{entry.Document.TaskID, f.audit.AuditID, contentdigest.Bytes(source), source, document, batch, ordinal, (ordinal-1)/3125 + 1, state, gap})
		}
		if _, err := f.pool.CopyFrom(ctx, pgx.Identifier{"audit_tasks"}, []string{"task_id", "audit_id", "source_key", "source", "document", "batch_digest", "ordinal", "iteration", "state", "gap_code"}, pgx.CopyFromRows(rows)); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := f.pool.Exec(ctx, `UPDATE audits SET task_count=$2,task_iteration=32,iteration_task_count=3125 WHERE audit_id=$1`, f.audit.AuditID, total); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(ctx, `VACUUM (ANALYZE) audit_tasks`); err != nil {
		t.Fatal(err)
	}
	ready := auditstore.TaskReady
	page, err := f.reads.ListTasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: f.audit.AuditID, State: &ready, Limit: 10})
	if err != nil || len(page.Tasks) != 3 || page.NextOrdinal != nil || page.Tasks[0].Ordinal != total-2 {
		t.Fatalf("sparse ready selection: %+v %v", page, err)
	}
	for _, mode := range []explaintest.Mode{explaintest.Custom, explaintest.Generic} {
		origin, _ := strictjson.Canonical(auditstore.TaskSource{Kind: "manual", Key: "task-100000"})
		sources, err := explaintest.ExplainPrepared(ctx, f.pool, mode, tasksBySourceSQL, f.audit.AuditID, []string{contentdigest.Bytes(origin), contentdigest.Bytes([]byte("absent"))})
		if err != nil {
			t.Fatal(err)
		}
		if n := sources.Reads.Of("audit_tasks"); n > 16 || int(sources.Plan.Rows) != 1 {
			t.Fatalf("%s admission source read %d tasks:\n%s", mode, n, sources.Plan)
		}
		selection, err := explaintest.ExplainPrepared(ctx, f.pool, mode, readyTasksSQL, f.audit.AuditID, []string{"inspect"})
		if err != nil {
			t.Fatal(err)
		}
		if n := selection.Reads.Of("audit_tasks"); n > 16 || int(selection.Plan.Rows) != 3 {
			t.Fatalf("%s ready dispatch read %d tasks:\n%s", mode, n, selection.Plan)
		}
		analysis, err := explaintest.ExplainPrepared(ctx, f.pool, mode, fmt.Sprintf(taskPageSQL, taskStateFilterSQL(&ready)), f.audit.AuditID, 0, 11)
		if err != nil {
			t.Fatal(err)
		}
		if n := analysis.Reads.Of("audit_tasks"); n > 11 || int(analysis.Plan.Rows) != 3 {
			t.Fatalf("%s sparse page read %d tasks:\n%s", mode, n, analysis.Plan)
		}
		analysis, err = explaintest.ExplainPrepared(ctx, f.pool, mode, fmt.Sprintf(taskPageSQL, ""), f.audit.AuditID, 0, 201)
		if err != nil {
			t.Fatal(err)
		}
		if n := analysis.Reads.Of("audit_tasks"); n > 209 {
			t.Fatalf("%s page read %d tasks:\n%s", mode, n, analysis.Plan)
		}
	}
	if _, err := f.pool.Exec(ctx, `UPDATE audit_tasks SET state='settled',gap_code='cancelled' WHERE audit_id=$1 AND state='ready'`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(ctx, `UPDATE audits SET state='deleting',dispatch_state='closed',hold_state='released',deletion_requested_at=clock_timestamp() WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	claims := NewClaimStore(f.pool).WithPurgeDependencies(func(context.Context, pgx.Tx, string, int) (int, error) { return 0, nil })
	claimed, err := claims.Claim(ctx, auditstore.ClaimParams{HolderID: "ceiling-purger", Lease: 5 * time.Minute, Limit: 1})
	if err != nil || len(claimed) != 1 {
		t.Fatalf("claim: %+v %v", claimed, err)
	}
	previous := total
	for pass := 0; pass < 105; pass++ {
		if err := claims.PurgeClaimed(ctx, claimed[0], auditdomain.ArtifactNamespace(f.audit.AuditID)); err != nil {
			t.Fatal(err)
		}
		var remaining int
		if err := f.pool.QueryRow(ctx, `SELECT count(*) FROM audit_tasks WHERE audit_id=$1`, f.audit.AuditID).Scan(&remaining); err != nil {
			t.Fatal(err)
		}
		if removed := previous - remaining; removed < 0 || removed > postgres.CleanupBatchRows {
			t.Fatalf("purge pass removed %d tasks", removed)
		}
		previous = remaining
		var exists bool
		if err := f.pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audits WHERE audit_id=$1)`, f.audit.AuditID).Scan(&exists); err != nil {
			t.Fatal(err)
		}
		if !exists {
			if remaining != 0 {
				t.Fatal("parent cascaded Task history")
			}
			return
		}
	}
	t.Fatalf("bounded purge failed to drain %d tasks", previous)
}
