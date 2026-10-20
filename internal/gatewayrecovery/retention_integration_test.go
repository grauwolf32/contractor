//go:build integration

package gatewayrecovery

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// recoveryTestPool returns a pool on a fresh migrated schema. It skips only
// while CONTRACTOR_TEST_DATABASE_URL is unset; an unreachable database fails.
func recoveryTestPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := cryptorand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := pgx.Identifier{"gateway_recovery_" + hex.EncodeToString(random)}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+schema); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = strings.Trim(schema, `"`)
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		_, err := admin.Exec(context.Background(), `DROP SCHEMA `+schema+` CASCADE`)
		admin.Close()
		if err != nil {
			t.Error(err)
		}
	})
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}

func createPendingRun(t *testing.T, pool *pgxpool.Pool, runID string) {
	t.Helper()
	runs := runstore.NewPostgresStore(pool)
	if _, err := runstoretest.CreateRun(t.Context(), runs, runstore.CreateRunParams{
		RunID: runID, OwnerID: "user-1", WorkflowName: "test", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
		RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(t.Context(), runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
}

// TestProbeHolderLookupUsesItsIndex checks that the statements which find the
// probes a Run holds (Admit, EndStage, ReleaseAdmission) can read them through
// the partial probe holder index instead of every retained route.
func TestProbeHolderLookupUsesItsIndex(t *testing.T) {
	pool := recoveryTestPool(t)
	err := persistencepostgres.InTx(t.Context(), pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(t.Context(), `SET LOCAL enable_seqscan = off`); err != nil {
			return err
		}
		for _, query := range []string{
			`SELECT route_key FROM gateway_recovery_routes WHERE probe_run_id='run-1' AND probe_id IS NULL`,
			`SELECT route_key FROM gateway_recovery_routes WHERE probe_run_id='run-1' ORDER BY route_key FOR UPDATE`,
		} {
			var plan string
			if err := tx.QueryRow(t.Context(), `EXPLAIN (FORMAT JSON) `+query).Scan(&plan); err != nil {
				return err
			}
			if !strings.Contains(plan, `"Index Name": "gateway_recovery_routes_probe_run_idx"`) {
				t.Errorf("plan of %q does not use the probe holder index: %s", query, plan)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestRecoveryRoutesOutliveRunsAndReceiptsDoNot pins the retention decision
// (SCH-R169): ending a Stage drops the Run's waits, route membership and
// probe but keeps the route with its outage history for other Runs; hard
// deletion of the Run removes its failure receipts and keeps the route.
func TestRecoveryRoutesOutliveRunsAndReceiptsDoNot(t *testing.T) {
	ctx := t.Context()
	pool := recoveryTestPool(t)
	service, err := New(pool, runstore.GatewayRecoveryRuns{}, DefaultPolicy(), time.Minute, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	route := Route{OwnerID: "user-1", GatewayDigest: "sha256:" + strings.Repeat("1", 64), Model: "worker"}
	createPendingRun(t, pool, "run-1")
	if allowed, err := service.Admit(ctx, "run-1", []Route{route}); err != nil || !allowed {
		t.Fatalf("admit = %t, %v", allowed, err)
	}
	runs := runstore.NewPostgresStore(pool)
	if _, err := runs.TransitionRun(ctx, "run-1", runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	if err := runstoretest.CreateRunningStage(ctx, runs, "run-1", "stage-1", contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
	participant := service.Planner("run-1", "stage-1", route, llmgateway.DefaultGatewayFailureSignatures())
	if decision, err := participant.Update(ctx, Request{RequestID: "request-1", Action: "acquire"}); err != nil || !decision.Allowed {
		t.Fatalf("acquire = %+v, %v", decision, err)
	}
	if _, err := participant.Update(ctx, Request{RequestID: "request-1", Action: "failed", Code: "gateway_unavailable"}); err != nil {
		t.Fatal(err)
	}
	count := func(query string) int {
		t.Helper()
		var rows int
		if err := pool.QueryRow(ctx, query).Scan(&rows); err != nil {
			t.Fatal(err)
		}
		return rows
	}
	if got := count(`SELECT count(*) FROM gateway_recovery_failures WHERE run_id='run-1'`); got != 1 {
		t.Fatalf("failure receipts = %d, want 1", got)
	}
	// Retention is bounded because a route key carries configuration only:
	// another Run on the same configuration shares the route row.
	createPendingRun(t, pool, "run-2")
	if _, err := service.Admit(ctx, "run-2", []Route{route}); err != nil {
		t.Fatal(err)
	}
	if got := count(`SELECT count(*) FROM gateway_recovery_routes`); got != 1 {
		t.Fatalf("two Runs on one configuration created %d routes, want 1", got)
	}

	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `SELECT 1 FROM workflow_runs WHERE run_id='run-1' FOR UPDATE`); err != nil {
			return err
		}
		return EndStage(ctx, tx, runstore.GatewayRecoveryRuns{}, "run-1")
	}); err != nil {
		t.Fatal(err)
	}
	if got := count(`SELECT count(*) FROM gateway_recovery_waits WHERE run_id='run-1'`) +
		count(`SELECT count(*) FROM gateway_run_routes WHERE run_id='run-1'`) +
		count(`SELECT count(*) FROM gateway_recovery_routes WHERE probe_run_id='run-1'`); got != 0 {
		t.Fatalf("the ended Stage left %d waits, memberships or probes", got)
	}
	if got := count(`SELECT count(*) FROM gateway_recovery_routes WHERE blocked AND failure_count=1`); got != 1 {
		t.Fatal("ending the Stage dropped the route or its outage history")
	}
	if got := count(`SELECT count(*) FROM gateway_recovery_failures WHERE run_id='run-1'`); got != 1 {
		t.Fatalf("failure receipts after the Stage ended = %d, want 1 until the Run is deleted", got)
	}

	if _, err := pool.Exec(ctx, `DELETE FROM workflow_runs WHERE run_id='run-1'`); err != nil {
		t.Fatal(err)
	}
	if got := count(`SELECT count(*) FROM gateway_recovery_failures`); got != 0 {
		t.Fatalf("failure receipts after Run deletion = %d, want 0", got)
	}
	if got := count(`SELECT count(*) FROM gateway_recovery_routes`); got != 1 {
		t.Fatalf("routes after Run deletion = %d, want the retained route", got)
	}
}
