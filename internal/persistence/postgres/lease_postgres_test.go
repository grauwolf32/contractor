package postgres

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestPostgresServerLeaseProvesOnlyItsOwnLiveSession pins the proof of a
// lease-fenced write (PER-R34): it holds while the identified session holds
// the lease, and fails once that session is gone, while another session
// holds the lease, and for a backend that reuses the session's process ID.
func TestPostgresServerLeaseProvesOnlyItsOwnLiveSession(t *testing.T) {
	ctx, pool := leaseTestPool(t)
	key := randomLeaseKey(t)
	holder := connectLeaseSession(t, ctx, pool)
	if _, err := NewServerLease(ctx, holder, key); !errors.Is(err, ErrLeaseLost) {
		t.Fatalf("a session without the lease was identified as its holder: %v", err)
	}
	takeLease(t, ctx, holder, key)
	lease, err := NewServerLease(ctx, holder, key)
	if err != nil || !lease.Valid() {
		t.Fatalf("lease holder = (%+v, %v)", lease, err)
	}
	if err := proveIn(ctx, pool, lease); err != nil {
		t.Fatalf("the holder's proof failed: %v", err)
	}
	reused := lease
	reused.started = lease.started.Add(-time.Second)
	if err := proveIn(ctx, pool, reused); !errors.Is(err, ErrLeaseLost) {
		t.Fatalf("a backend reusing the holder's process ID proved the lease: %v", err)
	}
	if err := proveIn(ctx, pool, ServerLease{}); err == nil || errors.Is(err, ErrLeaseLost) {
		t.Fatalf("an unconfigured lease proof = %v, want a configuration error", err)
	}

	terminateLeaseSession(t, ctx, pool, holder)
	if err := proveIn(ctx, pool, lease); !errors.Is(err, ErrLeaseLost) {
		t.Fatalf("the proof of a terminated lease session = %v", err)
	}
	successor := connectLeaseSession(t, ctx, pool)
	takeLease(t, ctx, successor, key)
	next, err := NewServerLease(ctx, successor, key)
	if err != nil {
		t.Fatal(err)
	}
	if err := proveIn(ctx, pool, lease); !errors.Is(err, ErrLeaseLost) {
		t.Fatalf("the previous holder proved a lease another session holds: %v", err)
	}
	if err := proveIn(ctx, pool, next); err != nil {
		t.Fatalf("the successor's proof failed: %v", err)
	}
}

// TestPostgresLeaseHandoverWaitsForEveryFencedWriteInFlight pins the
// handover (PER-R34): the successor's wait ends only once the previous
// holder's fenced transaction ended, whatever its outcome, and a fenced
// write the previous holder starts after the handover commits nothing.
func TestPostgresLeaseHandoverWaitsForEveryFencedWriteInFlight(t *testing.T) {
	ctx, pool := leaseTestPool(t)
	if _, err := pool.Exec(ctx, `CREATE TABLE fenced_writes (id integer PRIMARY KEY)`); err != nil {
		t.Fatal(err)
	}
	key := randomLeaseKey(t)
	previous := connectLeaseSession(t, ctx, pool)
	takeLease(t, ctx, previous, key)
	lease, err := NewServerLease(ctx, previous, key)
	if err != nil {
		t.Fatal(err)
	}
	for _, outcome := range []string{"commit", "rollback"} {
		t.Run(outcome, func(t *testing.T) {
			inFlight, err := BeginTx(ctx, pool, pgx.TxOptions{})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = inFlight.Rollback(context.WithoutCancel(ctx)) }()
			if err := lease.Prove(ctx, inFlight); err != nil {
				t.Fatal(err)
			}
			id := 1
			if outcome == "rollback" {
				id = 2
			}
			if _, err := inFlight.Exec(ctx, `INSERT INTO fenced_writes (id) VALUES ($1)`, id); err != nil {
				t.Fatal(err)
			}
			var writer int32
			if err := inFlight.QueryRow(ctx, `SELECT pg_backend_pid()`).Scan(&writer); err != nil {
				t.Fatal(err)
			}
			handover := make(chan error, 1)
			go func() { handover <- AwaitLeaseHandover(ctx, pool, lease) }()
			awaitAdvisoryWaiterBlockedBy(t, ctx, pool, writer)
			select {
			case err := <-handover:
				t.Fatalf("the handover ended while a fenced write was open: %v", err)
			default:
			}
			if outcome == "commit" {
				err = inFlight.Commit(ctx)
			} else {
				err = inFlight.Rollback(ctx)
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := <-handover; err != nil {
				t.Fatalf("handover after the fenced write ended: %v", err)
			}
		})
	}
	var written []int32
	rows, err := pool.Query(ctx, `SELECT id FROM fenced_writes ORDER BY id`)
	if err != nil {
		t.Fatal(err)
	}
	if written, err = pgx.CollectRows(rows, pgx.RowTo[int32]); err != nil || len(written) != 1 || written[0] != 1 {
		t.Fatalf("fenced writes after the handovers = (%v, %v), want the committed one", written, err)
	}

	terminateLeaseSession(t, ctx, pool, previous)
	successor := connectLeaseSession(t, ctx, pool)
	takeLease(t, ctx, successor, key)
	next, err := NewServerLease(ctx, successor, key)
	if err != nil {
		t.Fatal(err)
	}
	if err := AwaitLeaseHandover(ctx, pool, next); err != nil {
		t.Fatal(err)
	}
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := lease.Prove(ctx, tx); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `INSERT INTO fenced_writes (id) VALUES (3)`)
		return err
	})
	if !errors.Is(err, ErrLeaseLost) {
		t.Fatalf("a write of the previous holder after the handover = %v", err)
	}
	if err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := next.Prove(ctx, tx); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `INSERT INTO fenced_writes (id) VALUES (4)`)
		return err
	}); err != nil {
		t.Fatalf("a write of the successor: %v", err)
	}
	var count int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM fenced_writes WHERE id = 3`).Scan(&count); err != nil || count != 0 {
		t.Fatalf("the previous holder's late write committed: (%d, %v)", count, err)
	}
}

// TestPostgresLeaseHandoverEndsAtTheLockTimeout pins the bound of the
// handover: database.lockTimeout ends a wait for a fenced write that stays
// open as a definite timeout, and a caller cancelled before or during the
// wait keeps its cancellation.
func TestPostgresLeaseHandoverEndsAtTheLockTimeout(t *testing.T) {
	ctx, pool := leaseTestPool(t)
	key := randomLeaseKey(t)
	holder := connectLeaseSession(t, ctx, pool)
	takeLease(t, ctx, holder, key)
	lease, err := NewServerLease(ctx, holder, key)
	if err != nil {
		t.Fatal(err)
	}
	inFlight, err := BeginTx(ctx, pool, pgx.TxOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = inFlight.Rollback(context.WithoutCancel(ctx)) }()
	if err := lease.Prove(ctx, inFlight); err != nil {
		t.Fatal(err)
	}

	config := pool.Config()
	config.ConnConfig.RuntimeParams["lock_timeout"] = "50"
	bounded, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer bounded.Close()
	err = AwaitLeaseHandover(ctx, bounded, lease)
	if err == nil || Classify(err) != ClassTimeout {
		t.Fatalf("a handover beyond the lock timeout = (%v, class %v), want a definite timeout", err, Classify(err))
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if err := AwaitLeaseHandover(cancelled, pool, lease); !errors.Is(err, context.Canceled) {
		t.Fatalf("a cancelled handover = %v", err)
	}
	var writer int32
	if err := inFlight.QueryRow(ctx, `SELECT pg_backend_pid()`).Scan(&writer); err != nil {
		t.Fatal(err)
	}
	waiting, stopWaiting := context.WithCancel(ctx)
	handover := make(chan error, 1)
	go func() { handover <- AwaitLeaseHandover(waiting, pool, lease) }()
	awaitAdvisoryWaiterBlockedBy(t, ctx, pool, writer)
	stopWaiting()
	if err := <-handover; !errors.Is(err, context.Canceled) {
		t.Fatalf("a handover cancelled while it waits = %v, want the caller's cancellation", err)
	}
	if err := AwaitLeaseHandover(ctx, pool, ServerLease{}); err == nil {
		t.Fatal("a handover without a lease was accepted")
	}

	if err := inFlight.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if err := AwaitLeaseHandover(ctx, bounded, lease); err != nil {
		t.Fatalf("handover once the fenced write ended: %v", err)
	}
}

func leaseTestPool(t *testing.T) (context.Context, *pgxpool.Pool) {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	t.Cleanup(cancel)
	pool, _ := isolatedPools(t, ctx, url)
	return ctx, pool
}

// randomLeaseKey keeps the test's lease apart from every other lease of the
// shared test database, including a Server's.
func randomLeaseKey(t *testing.T) int64 {
	t.Helper()
	var random [8]byte
	if _, err := rand.Read(random[:]); err != nil {
		t.Fatal(err)
	}
	return int64(binary.BigEndian.Uint64(random[:]))
}

func connectLeaseSession(t *testing.T, ctx context.Context, pool *pgxpool.Pool) *pgx.Conn {
	t.Helper()
	conn, err := pgx.ConnectConfig(ctx, pool.Config().ConnConfig.Copy())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close(context.Background()) })
	return conn
}

func takeLease(t *testing.T, ctx context.Context, conn *pgx.Conn, key int64) {
	t.Helper()
	var acquired bool
	if err := conn.QueryRow(ctx, `SELECT pg_try_advisory_lock($1)`, key).Scan(&acquired); err != nil || !acquired {
		t.Fatalf("take the test lease = (%v, %v)", acquired, err)
	}
}

// terminateLeaseSession ends conn's backend from the server side and waits
// until PostgreSQL released its locks.
func terminateLeaseSession(t *testing.T, ctx context.Context, pool *pgxpool.Pool, conn *pgx.Conn) {
	t.Helper()
	var terminated bool
	if err := pool.QueryRow(ctx, `SELECT pg_terminate_backend($1, 30000)`, int32(conn.PgConn().PID())).
		Scan(&terminated); err != nil || !terminated {
		t.Fatalf("terminate the lease session = (%v, %v)", terminated, err)
	}
}

func proveIn(ctx context.Context, pool *pgxpool.Pool, lease ServerLease) error {
	return InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error { return lease.Prove(ctx, tx) })
}

// awaitAdvisoryWaiterBlockedBy waits until a session waits for an advisory
// lock that the backend blocker holds.
func awaitAdvisoryWaiterBlockedBy(t *testing.T, ctx context.Context, pool *pgxpool.Pool, blocker int32) {
	t.Helper()
	for {
		var waiting bool
		if err := pool.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1 FROM pg_locks
    WHERE locktype = 'advisory' AND NOT granted AND $1::integer = ANY (pg_blocking_pids(pid)))`, blocker).
			Scan(&waiting); err != nil {
			t.Fatal(err)
		}
		if waiting {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatal("no session waited for the fenced write's advisory lock")
		case <-time.After(10 * time.Millisecond):
		}
	}
}
