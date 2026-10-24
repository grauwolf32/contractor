package app

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/grauwolf32/contractor/internal/persistence/leasetest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// A server-side termination of the active Server's lease session releases
// the lease at once, while that Server may still commit a lease-fenced write
// until it notices. The standby that takes the lease returns from
// awaitControlPlaneLease, after which Run starts credential recovery, only
// once that write ended, and a write the previous holder starts afterwards
// commits nothing (PER-R34).
func TestPostgresControlPlaneLeaseStandbyTakesOverOnlyAfterTheHandover(t *testing.T) {
	ctx, pool := handoverTestPool(t)
	previous := activeLeaseForHandover(t, ctx, pool)
	writer, commit := openFencedWrite(t, ctx, pool, previous, 1)

	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	standbyAddress := probe.Addr().String()
	_ = probe.Close()
	standbyCtx, cancelStandby := context.WithCancel(ctx)
	defer cancelStandby()
	type result struct {
		lease *controlPlaneLease
		err   error
	}
	standbyDone := make(chan result, 1)
	go func() {
		lease, err := awaitControlPlaneLease(
			standbyCtx, pool, standbyAddress, time.Second, slog.New(slog.NewTextHandler(io.Discard, nil)),
		)
		standbyDone <- result{lease, err}
	}()
	received := false
	t.Cleanup(func() {
		cancelStandby()
		if !received {
			if completed := <-standbyDone; completed.lease != nil {
				completed.lease.Close()
			}
		}
	})
	client := &http.Client{Timeout: 200 * time.Millisecond}
	awaitStandbyReadiness(t, ctx, client, standbyAddress)

	terminateLeaseBackend(t, ctx, pool, previous)
	// The standby took the lease and its handover waits for the write.
	leasetest.AwaitWaiterBlockedBy(t, pool, writer)
	select {
	case completed := <-standbyDone:
		received = true
		t.Fatalf("the standby took over while the previous holder's write was open: %v", completed.err)
	default:
	}
	if response, err := client.Get("http://" + standbyAddress + "/readyz"); err != nil {
		t.Fatalf("the standby stopped answering during the handover: %v", err)
	} else {
		_ = response.Body.Close()
		if response.StatusCode != http.StatusServiceUnavailable {
			t.Fatalf("standby readiness during the handover = %d, want 503", response.StatusCode)
		}
	}

	commit()
	completed := <-standbyDone
	received = true
	if completed.err != nil || completed.lease == nil {
		t.Fatalf("the standby did not take over after the handover: %v", completed.err)
	}
	defer completed.lease.Close()
	assertHandoverOutcome(t, ctx, pool, previous, completed.lease)
}

// A Server that starts after the previous holder's session ended takes the
// lease at its first probe and waits for the same handover.
func TestPostgresControlPlaneLeaseFirstAcquisitionWaitsForTheHandover(t *testing.T) {
	ctx, pool := handoverTestPool(t)
	previous := activeLeaseForHandover(t, ctx, pool)
	writer, commit := openFencedWrite(t, ctx, pool, previous, 1)
	terminateLeaseBackend(t, ctx, pool, previous)

	started := make(chan *controlPlaneLease, 1)
	failed := make(chan error, 1)
	go func() {
		lease, err := awaitControlPlaneLease(ctx, pool, "127.0.0.1:0", time.Second, slog.New(slog.NewTextHandler(io.Discard, nil)))
		if err != nil {
			failed <- err
			return
		}
		started <- lease
	}()
	leasetest.AwaitWaiterBlockedBy(t, pool, writer)
	select {
	case lease := <-started:
		lease.Close()
		t.Fatal("the Server started while the previous holder's write was open")
	case err := <-failed:
		t.Fatalf("startup failed during the handover: %v", err)
	default:
	}
	commit()
	select {
	case lease := <-started:
		defer lease.Close()
		assertHandoverOutcome(t, ctx, pool, previous, lease)
	case err := <-failed:
		t.Fatalf("startup after the handover: %v", err)
	}
}

// A handover that outlasts database.lockTimeout stops startup like a failed
// startup lease step (FAIL-PER-4): the Server releases the lease and exits.
func TestPostgresControlPlaneLeaseHandoverTimeoutStopsStartup(t *testing.T) {
	ctx, pool := handoverTestPool(t)
	previous := activeLeaseForHandover(t, ctx, pool)
	_, commit := openFencedWrite(t, ctx, pool, previous, 1)
	defer commit()
	terminateLeaseBackend(t, ctx, pool, previous)

	config := pool.Config()
	config.ConnConfig.RuntimeParams["lock_timeout"] = "50"
	bounded, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer bounded.Close()
	lease, err := awaitControlPlaneLease(ctx, bounded, "127.0.0.1:0", time.Second, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if lease != nil {
		lease.Close()
	}
	if err == nil || persistencepostgres.Classify(err) != persistencepostgres.ClassTimeout {
		t.Fatalf("startup beyond the handover's lock timeout = %v, want a definite timeout", err)
	}
	next, err := openControlPlaneLease(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	defer next.Close()
	if acquired, err := next.tryAcquire(ctx); err != nil || !acquired {
		t.Fatalf("the stopped Server kept the lease: (%v, %v)", acquired, err)
	}
}

func handoverTestPool(t *testing.T) (context.Context, *pgxpool.Pool) {
	t.Helper()
	ctx, stop := context.WithTimeout(t.Context(), 30*time.Second)
	t.Cleanup(stop)
	pool := isolatedAppPool(t, ctx)
	if _, err := pool.Exec(ctx, `CREATE TABLE handover_writes (id integer PRIMARY KEY)`); err != nil {
		t.Fatal(err)
	}
	return ctx, pool
}

// activeLeaseForHandover is the previous active Server's lease: acquired and
// past its own handover, as awaitControlPlaneLease returns it.
func activeLeaseForHandover(t *testing.T, ctx context.Context, pool *pgxpool.Pool) *controlPlaneLease {
	t.Helper()
	lease, err := openControlPlaneLease(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(lease.Close)
	if acquired, err := lease.tryAcquire(ctx); err != nil || !acquired {
		t.Fatalf("the previous Server did not acquire the lease: (%v, %v)", acquired, err)
	}
	if err := lease.awaitHandover(ctx); err != nil {
		t.Fatal(err)
	}
	return lease
}

// openFencedWrite opens a lease-fenced write of the lease's holder that has
// written row id and waits for its COMMIT. It returns the writer's backend
// and the commit, which the test also runs at its end.
func openFencedWrite(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, lease *controlPlaneLease, id int,
) (int32, func()) {
	t.Helper()
	tx, err := persistencepostgres.BeginTx(ctx, pool, pgx.TxOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tx.Rollback(context.WithoutCancel(ctx)) })
	if err := lease.server.Prove(ctx, tx); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.Exec(ctx, `INSERT INTO handover_writes (id) VALUES ($1)`, id); err != nil {
		t.Fatal(err)
	}
	var writer int32
	if err := tx.QueryRow(ctx, `SELECT pg_backend_pid()`).Scan(&writer); err != nil {
		t.Fatal(err)
	}
	committed := false
	return writer, func() {
		t.Helper()
		if committed {
			return
		}
		committed = true
		if err := tx.Commit(ctx); err != nil {
			t.Fatal(err)
		}
	}
}

// terminateLeaseBackend ends the lease session from the server side, as
// pg_terminate_backend or a failure of that one connection does, and waits
// until PostgreSQL released the lease.
func terminateLeaseBackend(t *testing.T, ctx context.Context, pool *pgxpool.Pool, lease *controlPlaneLease) {
	t.Helper()
	var terminated bool
	if err := pool.QueryRow(ctx, `SELECT pg_terminate_backend($1, 30000)`, int32(lease.conn.PgConn().PID())).
		Scan(&terminated); err != nil || !terminated {
		t.Fatalf("terminate the lease session = (%v, %v)", terminated, err)
	}
}

func awaitStandbyReadiness(t *testing.T, ctx context.Context, client *http.Client, address string) {
	t.Helper()
	for {
		response, err := client.Get("http://" + address + "/readyz")
		if err == nil {
			_ = response.Body.Close()
			if response.StatusCode != http.StatusServiceUnavailable {
				t.Fatalf("standby readiness = %d, want 503", response.StatusCode)
			}
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("standby readiness listener did not start: %v", err)
		case <-time.After(20 * time.Millisecond):
		}
	}
}

// assertHandoverOutcome checks what the new holder finds once it returned
// from awaitControlPlaneLease: the previous holder's fenced write committed,
// a write the previous holder starts now commits nothing, and the new
// holder's own fenced writes commit.
func assertHandoverOutcome(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, previous, next *controlPlaneLease,
) {
	t.Helper()
	var count int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM handover_writes WHERE id = 1`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("the previous holder's write before the handover = (%d, %v), want committed", count, err)
	}
	write := func(lease *controlPlaneLease, id int) error {
		return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			if err := lease.server.Prove(ctx, tx); err != nil {
				return err
			}
			_, err := tx.Exec(ctx, `INSERT INTO handover_writes (id) VALUES ($1)`, id)
			return err
		})
	}
	if err := write(previous, 2); !errors.Is(err, persistencepostgres.ErrLeaseLost) {
		t.Fatalf("a write of the previous holder after the handover = %v", err)
	}
	if err := write(next, 3); err != nil {
		t.Fatalf("a write of the new holder: %v", err)
	}
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM handover_writes WHERE id = 2`).Scan(&count); err != nil || count != 0 {
		t.Fatalf("the previous holder's late write = (%d, %v), want none", count, err)
	}
}
