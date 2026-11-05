package pgstore

import (
	"context"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/helpers/disktest"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestPostgresAuditReadsAtFindingScale reads the workspace, the finding and
// finding pages and the uncollected-execution probe of an Audit with 4,001
// findings and 20,001 collected executions among
// background Audits of the same owner, in the custom and the generic plan.
// Changed statements have budgets that their original form exceeds: the
// workspace counts the Audit's findings in one pass where it took two (8,002
// rows), a finding count or page reads only the Audit's findings where the
// generic plan read every finding of the owner (12,001), and the uncollected
// probe reads only uncollected executions where it read every collected one
// (20,001). Kept statements have their measured budgets: an exact count reads
// each row it counts, and a finding page reads one fetch. Every result equals
// an independent read of the fixture, which has a database of its own, so
// the budgets count live rows only. Each budget adds planningEntries.
func TestPostgresAuditReadsAtFindingScale(t *testing.T) {
	disktest.Require(t)
	f := newFindingScaleFixture(t)
	owner, auditID := f.audit.OwnerID, f.audit.AuditID
	var findings, unreviewed, proposed, collected int64
	if err := f.pool.QueryRow(f.ctx, `
SELECT (SELECT count(*) FROM project_finding_audits WHERE audit_id = $1),
       (SELECT count(*) FROM project_findings JOIN project_finding_audits USING(project_id,finding_id) WHERE audit_id = $1 AND analyst_verdict IS NULL),
       (SELECT count(*) FROM project_findings JOIN project_finding_audits USING(project_id,finding_id) WHERE audit_id = $1 AND state IS NULL),
       (SELECT count(*) FROM audit_executions WHERE audit_id = $1 AND state = 'collected')`, auditID).
		Scan(&findings, &unreviewed, &proposed, &collected); err != nil {
		t.Fatal(err)
	}
	if findings != findingScaleFindings+1 ||
		collected != findingScaleExecutions+1 {
		t.Fatalf("fixture holds %d findings and %d collected executions", findings, collected)
	}
	conn, err := f.pool.Acquire(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Release()
	store := NewQueryStore(conn, artifactpgtest.NewStorage())
	// inModes runs read under each plan cache mode on the store's
	// connection, so its prepared statements run as custom and as generic
	// plans.
	inModes := func(t *testing.T, read func(t *testing.T, mode explaintest.Mode)) {
		t.Helper()
		for _, mode := range explaintest.Modes {
			if _, err := conn.Exec(f.ctx, "SET plan_cache_mode = "+string(mode)); err != nil {
				t.Fatal(err)
			}
			read(t, mode)
		}
		if _, err := conn.Exec(f.ctx, "RESET plan_cache_mode"); err != nil {
			t.Fatal(err)
		}
	}
	within := func(t *testing.T, mode explaintest.Mode, statement string, args []any, budgets map[string]int64) {
		t.Helper()
		plan := explainPrepared(t, f.ctx, f.pool, mode, statement, args...)
		for table, rows := range budgets {
			budget, read := rows+planningEntries, plan.Reads.Of(table)
			if read > budget {
				t.Fatalf("%s: read %d %s rows, budget %d\n%s", mode, read, table, budget, plan.Plan)
			}
			t.Logf("%s: read %d %s rows (budget %d)", mode, read, table, budget)
		}
	}

	t.Run("workspace", func(t *testing.T) {
		inModes(t, func(t *testing.T, mode explaintest.Mode) {
			summary, err := store.GetWorkspace(f.ctx, owner, auditID)
			if err != nil || summary.TotalTasks != 0 ||
				summary.AsOf.IsZero() {
				t.Fatalf("%s: workspace = %+v, %v", mode, summary, err)
			}
			// Coordination reads do not visit findings or executions.
			within(t, mode, getWorkspaceSQL, []any{owner, auditID}, map[string]int64{
				"project_findings": 0, "project_finding_audits": 0, "audit_executions": 0,
			})
		})
	})

	t.Run("uncollected probe", func(t *testing.T) {
		requireUncollectedProbe(t, f.ctx, f.pool, auditID, f.execution.ExecutionID)
	})
}

// uncollectedProbeSQL holds when Audit $1 has no uncollected execution, the
// gate of report proposal, report commit and purge.
const uncollectedProbeSQL = `SELECT NOT ` + uncollectedExecutionSQL + ` FROM audits AS audit WHERE audit.audit_id = $1`

// originalUncollectedProbeSQL is the gate as it read before V409-004.
const originalUncollectedProbeSQL = `
SELECT NOT EXISTS (
           SELECT 1 FROM audit_executions AS execution
            WHERE execution.audit_id = audit.audit_id AND execution.state <> 'collected')
  FROM audits AS audit WHERE audit.audit_id = $1`

// requireUncollectedProbe checks that the gate reads no collected execution
// of an Audit whose executions are all collected, and that it agrees with
// the original gate once one execution in each declared state is added.
func requireUncollectedProbe(t *testing.T, ctx context.Context, pool *pgxpool.Pool, auditID, collectedID string) {
	t.Helper()
	for _, mode := range explaintest.Modes {
		plan := explainPrepared(t, ctx, pool, mode, uncollectedProbeSQL, auditID)
		if read := plan.Reads.Of("audit_executions"); read > planningEntries {
			t.Fatalf("%s: the probe read %d executions of an Audit without uncollected ones\n%s", mode, read, plan.Plan)
		}
	}
	rows, err := pool.Query(ctx, `
SELECT attname::text FROM pg_attribute
 WHERE attrelid = 'audit_executions'::regclass AND attnum > 0 AND NOT attisdropped
   AND (attname LIKE 'collection\_%' OR attname = 'collected_at')`)
	if err != nil {
		t.Fatal(err)
	}
	collection, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil || len(collection) == 0 {
		t.Fatalf("collection columns = %v, %v", collection, err)
	}
	states := statesOfExecutions(t)
	for _, state := range states {
		t.Run(state, func(t *testing.T) {
			tx, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
			if _, err := tx.Exec(ctx, `ALTER TABLE audit_executions DISABLE TRIGGER USER`); err != nil {
				t.Fatal(err)
			}
			id := "probe-" + state
			overrides := map[string]string{
				"execution_id": `'` + id + `'`, "submission_key": `'` + id + `'`, "round_id": `NULL`,
				"role": `'task'`, "role_attempt": `NULL`, "state": `'` + state + `'`,
				"run_id": `'` + id + `-run'`, "run_provenance": `jsonb_set(src.run_provenance, '{runId}', to_jsonb('` + id + `-run'::text))`,
			}
			if state == string(auditstore.ExecutionCollected) {
				overrides["collection_receipt_id"] = `'` + id + `-collection'`
			} else {
				for _, column := range collection {
					overrides[column] = `NULL`
				}
			}
			if state == string(auditstore.ExecutionIntent) || state == string(auditstore.ExecutionSubmitted) {
				for _, column := range []string{"terminal_outcome", "terminal_run_generation", "terminal_run_sequence", "terminal_observed_at"} {
					overrides[column] = `NULL`
				}
			}
			if state == string(auditstore.ExecutionIntent) {
				overrides["run_id"], overrides["run_provenance"] = `NULL`, `NULL`
			}
			cloneRows(t, ctx, tx, "audit_executions", "execution_id", collectedID, 1, overrides)
			var probe, original bool
			if err := tx.QueryRow(ctx, uncollectedProbeSQL, auditID).Scan(&probe); err != nil {
				t.Fatal(err)
			}
			if err := tx.QueryRow(ctx, originalUncollectedProbeSQL, auditID).Scan(&original); err != nil {
				t.Fatal(err)
			}
			if probe != original || probe != (state == string(auditstore.ExecutionCollected)) {
				t.Fatalf("with a %s execution the probe holds %t and the original %t", state, probe, original)
			}
		})
	}
}
