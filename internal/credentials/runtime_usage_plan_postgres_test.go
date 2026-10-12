package credentials

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The usage-plan history has several owners and is mostly terminal: one Run
// in fifty is active, and every third Run belongs to its owner's Project and
// pins an HTTP target credential. usageHistoryRun mirrors
// populateUsageHistorySQL so the test knows which Runs the read must return.
const (
	usageHistorySize   = 5000
	usageHistoryOwners = 7
	usageHistoryKeys   = 30
)

type usageHistoryRun struct {
	id                string
	active            bool
	runtime, httpCred string
}

func usageHistoryEntry(i int) usageHistoryRun {
	run := usageHistoryRun{
		id: fmt.Sprintf("run-usage-%05d", i), active: i%50 == 0,
		runtime: fmt.Sprintf("runtime-%d", i%usageHistoryKeys),
	}
	if i%3 == 0 {
		run.httpCred = fmt.Sprintf("runtime-%d", (i+10)%usageHistoryKeys)
	}
	return run
}

// populateUsageHistorySQL loads the history in one statement; see
// usageHistoryRun. $1 is a valid RuntimeConfig snapshot whose Runtime
// credential list is replaced per Run, $2 the owner count, $3 the credential
// count and $4 the Run count.
const populateUsageHistorySQL = `
INSERT INTO workflow_runs (
    run_id, owner_id, project_id, workflow_name, workflow_version, workflow_schema_version,
    workflow_snapshot, parameters, runtime_labels, runtime_config_snapshot,
    project_http_target_snapshot, state, state_reason_code,
    created_at, updated_at, started_at, finished_at
)
SELECT 'run-usage-' || lpad(i::text, 5, '0'), 'owner-' || (i % $2),
       CASE WHEN i % 3 = 0 THEN 'project-' || (i % $2) END,
       'history', '1', 'contractor/v1alpha1', '{}'::jsonb, '{}'::jsonb, '{}'::text[],
       jsonb_set($1::jsonb, '{runtimeCredentialIds}', jsonb_build_array('runtime-' || (i % $3))),
       CASE WHEN i % 3 = 0 THEN jsonb_build_object(
           'url', 'https://target.example.test/',
           'credential', jsonb_build_object('kind', 'http-origin-bearer@1', 'credentialId', 'runtime-' || ((i + 10) % $3))
       ) END,
       run.state, 'history', run.created_at, run.created_at,
       CASE WHEN run.state NOT IN ('initializing', 'pending') THEN run.created_at END,
       CASE WHEN run.state IN ('succeeded', 'failed') THEN run.created_at + interval '1 minute' END
FROM generate_series(1, $4) AS i
CROSS JOIN LATERAL (
    SELECT CASE
               WHEN i % 50 = 0 THEN (ARRAY['initializing', 'pending', 'running', 'waiting'])[(i / 50) % 4 + 1]
               WHEN i % 2 = 1 THEN 'failed'
               ELSE 'succeeded'
           END AS state,
           timestamptz '2026-09-01 00:00:00+00' + i * interval '1 second' AS created_at
) AS run`

func populateUsageHistory(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	projects := projectstore.NewPostgresStore(pool)
	for owner := range usageHistoryOwners {
		projectID := fmt.Sprintf("project-%d", owner)
		if _, _, err := projects.Create(ctx, projectstore.CreateParams{
			ProjectID: projectID, OwnerID: fmt.Sprintf("owner-%d", owner), Kind: projectstore.KindProject,
			Name: projectID, IdempotencyKey: projectID, RequestDigest: fmt.Sprintf("sha256:%064x", owner+1),
		}); err != nil {
			t.Fatalf("create Project %q: %v", projectID, err)
		}
	}
	snapshot, err := json.Marshal(runtimeconfigtest.BuiltInRunSnapshot())
	if err != nil {
		t.Fatal(err)
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	// The read under test ignores Run events, and appending one per Run would
	// dominate the load; deferred constraint triggers would otherwise block
	// re-enabling the event trigger.
	for _, statement := range []struct {
		sql  string
		args []any
	}{
		{sql: `SET CONSTRAINTS ALL IMMEDIATE`},
		{sql: `ALTER TABLE workflow_runs DISABLE TRIGGER workflow_runs_capture_lifecycle_event`},
		{sql: populateUsageHistorySQL, args: []any{snapshot, usageHistoryOwners, usageHistoryKeys, usageHistorySize}},
		{sql: `ALTER TABLE workflow_runs ENABLE TRIGGER workflow_runs_capture_lifecycle_event`},
	} {
		if _, err := tx.Exec(ctx, statement.sql, statement.args...); err != nil {
			t.Fatalf("populate Run history: %v", err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatalf("commit Run history: %v", err)
	}
	if _, err := pool.Exec(ctx, `VACUUM ANALYZE workflow_runs`); err != nil {
		t.Fatal(err)
	}
}

// explainUsagePlan returns the custom plan for args, or with generic the plan
// a cached prepared statement may reuse; GENERIC_PLAN keeps the $n
// placeholders, which only the simple query protocol passes unbound.
func explainUsagePlan(t *testing.T, ctx context.Context, pool *pgxpool.Pool, sql string, args []any, generic bool) string {
	t.Helper()
	var lines []string
	if generic {
		conn, err := pool.Acquire(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer conn.Release()
		results, err := conn.Conn().PgConn().Exec(ctx, "EXPLAIN (GENERIC_PLAN) "+sql).ReadAll()
		if err != nil {
			t.Fatalf("explain generic plan: %v", err)
		}
		for _, row := range results[0].Rows {
			lines = append(lines, string(row[0]))
		}
		return strings.Join(lines, "\n")
	}
	rows, err := pool.Query(ctx, "EXPLAIN "+sql, args...)
	if err != nil {
		t.Fatalf("explain: %v", err)
	}
	lines, err = pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		t.Fatal(err)
	}
	return strings.Join(lines, "\n")
}

// TestPostgresRuntimeCredentialRunUsageUsesActiveStateIndexes checks the
// Runtime credential deletion check on a history that is mostly terminal: it
// reports exactly the active Runs that pin the credential through their
// RuntimeConfig snapshot or their Project HTTP target, pending and waiting
// ones included, and it combines the two partial indexes over the active
// states instead of reading the Run history.
func TestPostgresRuntimeCredentialRunUsageUsesActiveStateIndexes(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	pool := isolatedCredentialPool(t, ctx, databaseURL)
	populateUsageHistory(t, ctx, pool)

	const credentialID = "runtime-10"
	var want []string
	viaRuntime, viaTarget := 0, 0
	for i := 1; i <= usageHistorySize; i++ {
		run := usageHistoryEntry(i)
		if !run.active || (run.runtime != credentialID && run.httpCred != credentialID) {
			continue
		}
		want = append(want, run.id)
		if run.runtime == credentialID {
			viaRuntime++
		}
		if run.httpCred == credentialID {
			viaTarget++
		}
	}
	if viaRuntime < 4 || viaTarget < 4 || len(want) > maximumRuntimeCredentialReferences {
		t.Fatalf("history pins %s in %d RuntimeConfig and %d HTTP target Runs", credentialID, viaRuntime, viaTarget)
	}
	usage, err := NewRuntimeCredentialRepository(pool).InspectRuntimeCredentialUsage(
		ctx, credentialID, maximumRuntimeCredentialReferences,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(usage.RunIDs, want) {
		t.Errorf("Runs pinning %s = %v, want %v", credentialID, usage.RunIDs, want)
	}

	for _, generic := range []bool{false, true} {
		plan := explainUsagePlan(t, ctx, pool, runtimeCredentialRunUsageSQL,
			[]any{credentialID, maximumRuntimeCredentialReferences}, generic)
		for _, index := range []string{
			"workflow_runs_runtime_credentials_idx", "workflow_runs_project_http_target_credential_idx",
		} {
			if !strings.Contains(plan, index) {
				t.Errorf("Runtime credential Runs (generic plan %v) do not use %s:\n%s", generic, index, plan)
			}
		}
		if strings.Contains(plan, "Seq Scan on workflow_runs") {
			t.Errorf("Runtime credential Runs (generic plan %v) read the whole Run history:\n%s", generic, plan)
		}
	}
}
