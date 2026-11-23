// Package leasetest holds the advisory lease that lease-fenced writes prove
// (PER-R34) for other packages' tests. Each Session takes the lease under a
// random key of its own, so tests that share a database neither contend with
// one another nor with a Server's lease.
package leasetest

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Session is a dedicated connection that holds a test lease, as a Server's
// lease session holds the Server's lease.
type Session struct {
	Lease  persistencepostgres.ServerLease
	key    int64
	config *pgx.ConnConfig
	conn   *pgx.Conn
}

// Hold opens a lease session on the database of pool and takes a lease
// under a new random key. The session closes when the test ends.
func Hold(t testing.TB, pool *pgxpool.Pool) *Session {
	t.Helper()
	var random [8]byte
	if _, err := rand.Read(random[:]); err != nil {
		t.Fatal(err)
	}
	key := int64(binary.BigEndian.Uint64(random[:]))
	return acquire(t, pool.Config().ConnConfig.Copy(), key)
}

// Successor opens another lease session that takes the lease of s, as the
// next active Server does once the lease is free. It fails the test while s
// or another session still holds that lease.
func (s *Session) Successor(t testing.TB) *Session {
	t.Helper()
	return acquire(t, s.config.Copy(), s.key)
}

// Lose ends the lease session from the server side, as a terminated or
// failed lease connection does, and waits until PostgreSQL has released the
// lease. The client connection is not told.
func (s *Session) Lose(t testing.TB) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	admin, err := pgx.ConnectConfig(ctx, s.config.Copy())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = admin.Close(context.Background()) }()
	var terminated bool
	if err := admin.QueryRow(ctx, `SELECT pg_terminate_backend($1, 30000)`, int32(s.conn.PgConn().PID())).
		Scan(&terminated); err != nil {
		t.Fatal(err)
	}
	if !terminated {
		t.Fatal("the lease session did not end")
	}
}

// AwaitWaiterBlockedBy waits until a session of the database of pool waits
// for an advisory lock that the backend blocker holds, as a new active
// Server's lease handover waits for a lease-fenced write that is still open.
// It fails the test when no session does so within half a minute.
func AwaitWaiterBlockedBy(t testing.TB, pool *pgxpool.Pool, blocker int32) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
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
			t.Fatal("no session waited for the advisory lock of the open write")
		case <-time.After(10 * time.Millisecond):
		}
	}
}

func acquire(t testing.TB, config *pgx.ConnConfig, key int64) *Session {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	conn, err := pgx.ConnectConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close(context.Background()) })
	var acquired bool
	if err := conn.QueryRow(ctx, `SELECT pg_try_advisory_lock($1)`, key).Scan(&acquired); err != nil {
		t.Fatal(err)
	}
	if !acquired {
		t.Fatal("the test lease is held by another session")
	}
	lease, err := persistencepostgres.NewServerLease(ctx, conn, key)
	if err != nil {
		t.Fatal(err)
	}
	return &Session{Lease: lease, key: key, config: config.Copy(), conn: conn}
}
