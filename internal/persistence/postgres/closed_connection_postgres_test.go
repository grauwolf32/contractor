package postgres

import (
	"context"
	"errors"
	"io"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The server has sent a successful COMMIT reply before the connection wrapper
// withholds it. This proves the ambiguity against PostgreSQL, without guessing
// when a disconnect will land relative to the server's commit.
func TestClosedConnectionAfterLostCommitReplyIsUnavailable(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	pool, observer := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := observer.Exec(ctx, `CREATE TABLE lost_commit_probe (id integer PRIMARY KEY)`); err != nil {
		t.Fatal(err)
	}
	config := pool.Config()
	config.MaxConns = 1
	var armed, forwarded, dropped atomic.Bool
	dial := config.ConnConfig.DialFunc
	config.ConnConfig.DialFunc = func(ctx context.Context, network, address string) (net.Conn, error) {
		conn, err := dial(ctx, network, address)
		if err != nil {
			return nil, err
		}
		return &lostCommitReplyConn{Conn: conn, armed: &armed, forwarded: &forwarded, dropped: &dropped}, nil
	}
	losingPool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(losingPool.Close)
	attempts := 0
	effects := 0
	err = InTxWithRetry(ctx, losingPool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		attempts++
		if !AfterCommit(tx, func(context.Context) { effects++ }, nil) {
			t.Fatal("transaction does not support commit effects")
		}
		if _, err := tx.Exec(ctx, `INSERT INTO lost_commit_probe (id) VALUES (1)`); err != nil {
			return err
		}
		// No SQL follows this point until InTx sends COMMIT. All responses to
		// the INSERT have been consumed before the wrapper is armed.
		armed.Store(true)
		return nil
	})
	if !forwarded.Load() || !dropped.Load() {
		t.Fatalf("COMMIT fault was not exercised: forwarded=%t dropped=%t", forwarded.Load(), dropped.Load())
	}
	if !errors.Is(err, pgconn.ErrConnClosed) {
		t.Fatalf("lost COMMIT reply error = %v, want pgconn.ErrConnClosed", err)
	}
	// This is deliberately a test-only observation: pgx reports safe-to-retry
	// although the separate session below proves that COMMIT reached the server.
	if !pgconn.SafeToRetry(err) {
		t.Fatalf("expected pgx closed-connection retry hint, got %v", err)
	}
	if got := Classify(err); got != ClassUnavailable || !IsTransientFailure(err) {
		t.Errorf("lost COMMIT reply: class=%d transient=%t, want unavailable/transient", got, IsTransientFailure(err))
	}
	if attempts != 1 {
		t.Errorf("ambiguous transaction replayed %d times, want one attempt", attempts)
	}
	if effects != 0 {
		t.Errorf("ran %d commit effects after an ambiguous response", effects)
	}
	var count int
	if err := observer.QueryRow(ctx, `SELECT count(*) FROM lost_commit_probe WHERE id = 1`).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("committed rows = %d, want 1", count)
	}
}

type lostCommitReplyConn struct {
	net.Conn
	armed, forwarded, dropped *atomic.Bool
}

func (c *lostCommitReplyConn) Write(p []byte) (int, error) {
	n, err := c.Conn.Write(p)
	if c.armed.Load() && n > 0 {
		c.forwarded.Store(true)
	}
	return n, err
}

func (c *lostCommitReplyConn) Read(p []byte) (int, error) {
	n, err := c.Conn.Read(p)
	if n > 0 && c.armed.Load() && c.dropped.CompareAndSwap(false, true) {
		_ = c.Conn.Close()
		return 0, io.EOF
	}
	return n, err
}

func TestClosedConnectionAfterBackendTerminationIsUnavailable(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	pool, killer := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	var terminated, closed error
	err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		var stopped bool
		if err := killer.QueryRow(ctx, `SELECT pg_terminate_backend($1, 10000)`, tx.Conn().PgConn().PID()).Scan(&stopped); err != nil {
			return err
		}
		if !stopped {
			t.Fatal("backend was not terminated")
		}
		_, terminated = tx.Exec(ctx, `SELECT 1`)
		_, closed = tx.Exec(ctx, `SELECT 1`)
		// Exercise the COMMIT failure too, even though the caller already
		// observed failed statements on this same transaction.
		return nil
	})
	// The backend may close before the next write. Either its FATAL reply
	// or a socket failure must have the same unavailable/transient outcome.
	for name, failure := range map[string]error{"terminated statement": terminated, "closed statement": closed, "commit": err} {
		if got := Classify(failure); got != ClassUnavailable || !IsTransientFailure(failure) {
			t.Errorf("%s: class=%d transient=%t, want unavailable/transient (%v)", name, got, IsTransientFailure(failure), failure)
		}
	}
	if !errors.Is(closed, pgconn.ErrConnClosed) || !errors.Is(err, pgconn.ErrConnClosed) {
		t.Fatalf("closed statement=%v commit=%v, want pgconn.ErrConnClosed", closed, err)
	}
}
