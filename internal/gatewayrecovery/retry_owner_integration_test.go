//go:build integration

package gatewayrecovery

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// runReadTrace counts the statements that name the Run table.
type runReadTrace struct {
	mu   sync.Mutex
	runs int
}

func (trace *runReadTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	if strings.Contains(data.SQL, "workflow_runs") {
		trace.mu.Lock()
		trace.runs++
		trace.mu.Unlock()
	}
	return ctx
}

func (*runReadTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

func (trace *runReadTrace) take() int {
	trace.mu.Lock()
	defer trace.mu.Unlock()
	runs := trace.runs
	trace.runs = 0
	return runs
}

// Manual retry authorizes the owner the Run row lock returns: it reads the
// Run once, refuses another owner and a missing Run as unavailable without
// touching the route, and reopens the automatic window for the owner. It
// read the owner again after the lock before.
func TestRetryChecksTheOwnerFromItsRunLock(t *testing.T) {
	ctx := t.Context()
	pool := recoveryTestPool(t)
	trace := &runReadTrace{}
	configuration := pool.Config().Copy()
	configuration.ConnConfig.Tracer = trace
	counted, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(counted.Close)
	service, err := New(counted, runstore.GatewayRecoveryRuns{}, DefaultPolicy(), time.Minute, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	route := Route{OwnerID: "user-1", GatewayDigest: "sha256:" + strings.Repeat("2", 64), Model: "worker"}
	createPendingRun(t, pool, "run-1")
	if allowed, err := service.Admit(ctx, "run-1", []Route{route}); err != nil || !allowed {
		t.Fatalf("admit = %t, %v", allowed, err)
	}
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
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
	if _, err := pool.Exec(ctx, `UPDATE gateway_recovery_routes SET automatic_until=clock_timestamp()-interval '1 second'`); err != nil {
		t.Fatal(err)
	}
	window := func() time.Time {
		t.Helper()
		var until time.Time
		if err := pool.QueryRow(ctx, `SELECT automatic_until FROM gateway_recovery_routes`).Scan(&until); err != nil {
			t.Fatal(err)
		}
		return until
	}
	expired := window()

	for _, test := range []struct{ owner, runID string }{{"user-2", "run-1"}, {"user-1", "run-missing"}} {
		trace.take()
		if err := service.Retry(ctx, test.owner, test.runID); !IsUnavailable(err) {
			t.Fatalf("retry of %s by %s = %v, want unavailable", test.runID, test.owner, err)
		}
		if runReads := trace.take(); runReads != 1 {
			t.Fatalf("refused retry read the Run %d times, want once", runReads)
		}
		if !window().Equal(expired) {
			t.Fatal("refused retry changed the route")
		}
	}
	trace.take()
	if err := service.Retry(ctx, "user-1", "run-1"); err != nil {
		t.Fatal(err)
	}
	if runReads := trace.take(); runReads != 1 {
		t.Fatalf("owner's retry read the Run %d times, want once", runReads)
	}
	if !window().After(expired) {
		t.Fatal("owner's retry did not reopen the automatic window")
	}
}
