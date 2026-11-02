package evalstore

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	// inventoryScaleExecutions child executions belong to the inventoried
	// Audit; each of inventoryScaleBackground other Audits of the same owner
	// holds inventoryScaleBackgroundExecutions more, whose execution IDs
	// interleave with them, so the inventoried Audit owns a tenth of the
	// table.
	inventoryScaleExecutions           = 5_000
	inventoryScaleBackground           = 3
	inventoryScaleBackgroundExecutions = 15_000
	inventoryScaleOwner                = "owner-inventory-scale"
)

// inventoryScaleFixture creates the inventoried Audit and the background
// Audits, each with collected check executions whose IDs carry the
// punctuation an execution ID may hold, and returns the pool and the
// inventoried Audit's ID.
func inventoryScaleFixture(t *testing.T) (context.Context, *pgxpool.Pool, string) {
	t.Helper()
	pool := testPool(t)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	t.Cleanup(cancel)
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-inventory-scale", OwnerID: inventoryScaleOwner, Kind: projectstore.KindProject,
		Name: "Inventory scale", IdempotencyKey: "project-inventory-scale", RequestDigest: evaldomain.Digest([]byte("project")),
	}); err != nil {
		t.Fatal(err)
	}
	audits := auditstoretest.NewStore(pool, artifactpgtest.NewStorage())
	digest := evaldomain.Digest([]byte("inventory scale"))
	auditIDs := []string{"audit-inventory-scale"}
	for index := range inventoryScaleBackground {
		auditIDs = append(auditIDs, fmt.Sprintf("audit-inventory-background-%d", index+1))
	}
	for index, auditID := range auditIDs {
		count := inventoryScaleExecutions
		if index != 0 {
			count = inventoryScaleBackgroundExecutions
		}
		if _, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
			AuditID: auditID, OwnerID: inventoryScaleOwner, ProjectID: "project-inventory-scale",
			Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: digest},
			ProfileSnapshot: json.RawMessage(`{"name":"profile"}`), InputSelection: json.RawMessage(`{}`),
			Limits: auditstore.Limits{
				MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
				MaxSubmittedRuns: 1_000_000, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
			},
			IdempotencyKey: auditID, RequestDigest: digest,
		}); err != nil {
			t.Fatal(err)
		}
		// Read-model fixtures, as in the public inventory test: a closed
		// round and collected executions whose Runs are deleted.
		if _, err := pool.Exec(ctx, `
INSERT INTO audit_rounds (round_id, audit_id, ordinal, manifest_ref, manifest_digest, state, expected_item_count)
VALUES ($1 || '-round', $1, 1, '{}'::jsonb, $2, 'closed', 0)`, auditID, digest); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
INSERT INTO audit_executions (lane_class, execution_id, audit_id, round_id, role, workflow_role,
       manifest_ref, manifest_digest, submission_key, request_digest, run_id, state, terminal_outcome,
       terminal_run_generation, terminal_run_sequence, terminal_observed_at, run_provenance, run_deleted_at)
SELECT 'model', key.id, $1, $1 || '-round', 'check', 'check', '{}'::jsonb, $2, key.id, $2,
       'run-' || key.id, 'collected', 'failed', 'generation', 1, clock_timestamp(),
       jsonb_build_object('schema', 'contractor.audit.run-provenance.v1', 'runId', 'run-' || key.id),
       clock_timestamp()
  FROM generate_series(1, $3::integer) AS n,
       LATERAL (SELECT substr(md5($4 || n), 1, 6) || (ARRAY['-', '.', '_', ':'])[n % 4 + 1]
                       || substr(md5($4 || n), 7, 10) AS id) AS key`,
			auditID, digest, count, string(rune('a'+index))); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := pool.Exec(ctx, `ANALYZE audit_executions; ANALYZE audits; ANALYZE audit_rounds`); err != nil {
		t.Fatal(err)
	}
	return ctx, pool, "audit-inventory-scale"
}

// originalInventoryKeysetSQL is the child keyset the inventory compared
// before V409-004: the whole page key '1' || execution_id against the cursor
// key. The test keeps it only to prove that the bare execution-ID keyset
// selects the same children under the database collation.
const originalInventoryKeysetSQL = `
SELECT x.execution_id FROM audit_executions AS x
 WHERE x.audit_id = $1 AND '1' || x.execution_id > $2
 ORDER BY x.execution_id LIMIT $3`

// TestPostgresInventoryChildPagesProbeTheExecutionIDRange pages a 5,000-child
// Audit inventory whose Audit owns a tenth of audit_executions. Each child
// page, from the parent boundary to the last child, starts its scan at the
// cursor's execution ID instead of filtering every earlier execution, in the
// custom and the generic plan: it reads at most twice the rows it fetches,
// where the original '1' || execution_id comparison read up to every
// execution of the table. Every page holds the same children, in the same
// order, as the original comparison selected, and another owner reads none.
func TestPostgresInventoryChildPagesProbeTheExecutionIDRange(t *testing.T) {
	ctx, pool, auditID := inventoryScaleFixture(t)
	rows, err := pool.Query(ctx, `SELECT execution_id FROM audit_executions WHERE audit_id = $1 ORDER BY execution_id`, auditID)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil || len(ids) != inventoryScaleExecutions {
		t.Fatalf("inventoried children = %d, %v", len(ids), err)
	}
	const fetch = 51 // a page of 50 children and one more to detect the next
	read := func(owner, child string, limit int) []string {
		t.Helper()
		rows, err := pool.Query(ctx, inventoryPageAuditExecutionsSQL, owner, auditID, child, limit)
		if err != nil {
			t.Fatal(err)
		}
		defer rows.Close()
		var result []string
		for rows.Next() {
			var id string
			var run, role, state, outcome *string
			var round *int
			var available bool
			if err := rows.Scan(&id, &run, &role, &round, &state, &available, &outcome); err != nil {
				t.Fatal(err)
			}
			result = append(result, id)
		}
		if err := rows.Err(); err != nil {
			t.Fatal(err)
		}
		return result
	}
	for _, position := range []struct {
		name, key string
		next      int // index in ids of the first child the page holds
	}{
		{"first page", "", 0},
		{"parent boundary", "0", 0},
		{"first child", "1" + ids[0], 1},
		{"early child", "1" + ids[100], 101},
		{"middle child", "1" + ids[2_500], 2_501},
		{"late child", "1" + ids[4_960], 4_961},
		{"last child", "1" + ids[len(ids)-1], len(ids)},
	} {
		t.Run(position.name, func(t *testing.T) {
			child, err := inventoryChildAfter(position.key)
			if err != nil {
				t.Fatal(err)
			}
			want := ids[position.next:min(len(ids), position.next+fetch)]
			got := read(inventoryScaleOwner, child, fetch)
			if !slices.Equal(got, want) {
				t.Fatalf("page after %q holds %d children from %v, want %d from %v", position.key, len(got), got[:min(1, len(got))], len(want), want[:min(1, len(want))])
			}
			original, err := pgx.CollectRows(must(pool.Query(ctx, originalInventoryKeysetSQL, auditID, position.key, fetch)), pgx.RowTo[string])
			if err != nil || !slices.Equal(original, got) {
				t.Fatalf("original keyset after %q selects %d children (%v), the execution-ID keyset %d", position.key, len(original), err, len(got))
			}
			if foreign := read("owner-inventory-foreign", child, fetch); len(foreign) != 0 {
				t.Fatalf("another owner reads %d children", len(foreign))
			}
			for _, mode := range explaintest.Modes {
				analysis, err := explaintest.ExplainPrepared(ctx, pool, mode, inventoryPageAuditExecutionsSQL, inventoryScaleOwner, auditID, child, fetch)
				if err != nil {
					t.Fatalf("%s: explain: %v", mode, err)
				}
				// The child keyset bounds an index scan of the executions,
				// plain or bitmap, by the execution ID.
				probe := slices.IndexFunc(analysis.Plan.Nodes(), func(node explaintest.Node) bool {
					return strings.HasPrefix(node.Index, "audit_executions_") && strings.Contains(node.IndexCond, "execution_id > ")
				})
				if read := analysis.Reads.Of("audit_executions"); read > 2*fetch || probe < 0 {
					t.Fatalf("%s: inventory page read %d executions (budget %d):\n%s", mode, read, 2*fetch, analysis.Plan)
				}
				t.Logf("%s: read %d executions:\n%s", mode, analysis.Reads.Of("audit_executions"), analysis.Plan)
			}
		})
	}
	for _, key := range []string{"1", "2" + ids[0], "00"} {
		if _, err := inventoryChildAfter(key); !code(err, "eval_invalid") {
			t.Fatalf("page key %q error = %v, want eval_invalid", key, err)
		}
	}
}

func must[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}
