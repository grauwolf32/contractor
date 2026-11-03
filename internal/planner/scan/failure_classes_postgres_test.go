package scan

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The retained plan/report reads use the production Artifact service and a real
// lock timeout. The scanner and journal harness retain the completed attempt so
// the retry can also prove that this storage failure never dispatches it again.
func TestPostgresAuditResultReadLockTimeoutRetainsCompletedScan(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedScanPool(t, ctx)
	h, history := newAuditFactoryHarness(t, "sqlmap")
	h.invocation.Deadline, _ = ctx.Deadline()
	instance := h.instance(t).(*execution)
	if _, err := instance.Run(ctx); err != nil {
		t.Fatal(err)
	}
	_, err := runstoretest.CreateRun(ctx, runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()), runstore.CreateRunParams{
		LaneClass: resolved.LaneClassTool, RunID: h.invocation.RunID, OwnerID: "user",
		WorkflowName: "scan", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: json.RawMessage(`{"name":"scan"}`), Parameters: map[string]string{},
		RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := NewServiceArtifacts(artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)))
	if err != nil {
		t.Fatal(err)
	}
	for _, ref := range []*contracts.ArtifactRef{h.sessions.state.Plan, h.sessions.state.Jobs[0].Report} {
		payload := h.artifacts.payload(*ref)
		target := ref.Clone()
		target.Revision = nil
		stored, err := store.Create(ctx, h.invocation.RunID, target, payload)
		if err != nil {
			t.Fatal(err)
		}
		*ref = stored
	}
	retained := cloneFactoryValue(h.sessions.state)
	retainedReads := 0
	h.factory.artifacts = &scanFaultArtifacts{ArtifactStore: h.artifacts,
		read: func(ctx context.Context, ref contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
			if retained.Plan.SameExact(ref) || retained.Jobs[0].Report.SameExact(ref) {
				retainedReads++
				return store.Read(ctx, h.invocation.RunID, ref, limit)
			}
			return h.artifacts.Read(ctx, h.invocation.RunID, ref, limit)
		}}
	attempts, err := history.ReadAuditScanHistory(ctx, h.invocation.RunID)
	if err != nil {
		t.Fatal(err)
	}
	lock, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = lock.Rollback(context.Background()) }()
	if _, err := lock.Exec(ctx, `LOCK TABLE artifact_binding_revisions IN ACCESS EXCLUSIVE MODE`); err != nil {
		t.Fatal(err)
	}
	_, err = instance.finishAudit(ctx, planner.ScanSessionIdentity{}, attempts)
	failure := planner.FailureFrom(err)
	var databaseError *pgconn.PgError
	if !errors.As(err, &databaseError) || databaseError.Code != persistencepostgres.SQLStateLockNotAvailable ||
		failure.Code != "audit_scan_result_unavailable" || !failure.Retryable || len(h.invoker.calls) != 1 {
		t.Fatalf("real retained read failure=%+v, cause=%v, scanner calls=%d", failure, err, len(h.invoker.calls))
	}
	if err := lock.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	history.prior = []planner.ScanAttempt{{StageExecutionID: history.stageID, StageName: "execute", Terminal: true, State: retained}}
	h.invocation.StageExecutionID = "postgres-result-read-retry"
	history.stageID = h.invocation.StageExecutionID
	h.sessions.state, h.sessions.completion = planner.ScanState{}, nil
	result := h.run(t)
	pkg := auditFactoryResult(t, h, result)
	if len(h.invoker.calls) != 1 || retainedReads != 3 || !pkg.ExecutionCompleted {
		t.Fatalf("retry lost evidence or rescanned: calls=%d, retained reads=%d, result=%+v", len(h.invoker.calls), retainedReads, pkg)
	}
}

func isolatedScanPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{
		Configure: func(poolConfig *pgxpool.Config) {
			poolConfig.ConnConfig.RuntimeParams["lock_timeout"] = "50ms"
		},
	})
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}
