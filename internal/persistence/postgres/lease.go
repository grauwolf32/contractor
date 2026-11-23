package postgres

import (
	"context"
	"errors"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// ErrLeaseLost reports a lease-fenced write that could not prove that this
// Server's lease session still holds the advisory lease (PER-R34): the
// session ended, or another Server holds the lease. The write's transaction
// does not commit.
var ErrLeaseLost = errors.New("this Server no longer holds the Control Plane lease")

// ServerLease identifies the session that holds the advisory lease of the
// active Server (PER-R2): the lease key, and the backend process ID and
// session start time of the lease session. A lease-fenced write proves it in
// its own transaction with Prove; the next active Server waits for every such
// write in flight with AwaitLeaseHandover before its startup reads durable
// state (PER-R34).
type ServerLease struct {
	key     int64
	pid     int32
	started time.Time
}

// NewServerLease identifies conn, the dedicated session that has just taken
// the session advisory lease of key, and checks that the lease is granted to
// it. A session that does not hold the lease has no ServerLease.
func NewServerLease(ctx context.Context, conn *pgx.Conn, key int64) (ServerLease, error) {
	if conn == nil {
		return ServerLease{}, errors.New("Control Plane lease session is not configured")
	}
	lease := ServerLease{key: key}
	if err := conn.QueryRow(ctx, `
SELECT pid, backend_start FROM pg_stat_activity WHERE pid = pg_backend_pid()`).
		Scan(&lease.pid, &lease.started); err != nil {
		return ServerLease{}, WrapError("identify Control Plane lease session", err)
	}
	held, err := lease.heldOn(ctx, conn)
	if err != nil {
		return ServerLease{}, err
	}
	if !held {
		return ServerLease{}, ErrLeaseLost
	}
	return lease, nil
}

// Valid reports whether the lease identifies a lease session.
func (lease ServerLease) Valid() bool {
	return lease.pid > 0 && !lease.started.IsZero()
}

// Prove fences the write of tx by the lease (PER-R34). It takes the shared
// transaction lock of the lease handover, which waits only while a new
// active Server is waiting for the writes of its predecessor, and then checks
// that the lease session still holds the lease. tx keeps the shared lock
// until it ends, so the next active Server reads durable state only after tx
// committed or rolled back; a write that starts after that handover finds
// the lease elsewhere and fails with ErrLeaseLost. Call it in the
// transaction that writes, before its write.
func (lease ServerLease) Prove(ctx context.Context, tx pgx.Tx) error {
	if !lease.Valid() || tx == nil {
		return errors.New("Control Plane lease fence is not configured")
	}
	if err := LockAdvisoryShared(ctx, tx, lease.handoverKey()); err != nil {
		return WrapError("lock Control Plane lease handover", err)
	}
	held, err := lease.heldOn(ctx, tx)
	if err != nil {
		return err
	}
	if !held {
		return ErrLeaseLost
	}
	return nil
}

// AwaitLeaseHandover waits until every lease-fenced write of an earlier
// holder of the lease has ended (PER-R34). The active Server calls it once,
// after it acquired lease and before its startup reads durable state: the
// exclusive handover lock is granted only once no transaction holds the
// shared lock of Prove, and a write that asks after this point waits for it
// and then finds the lease held by this Server. The working pool's budgets
// bound the wait: database.lockTimeout the lock wait, database.queryTimeout
// the statement.
func AwaitLeaseHandover(ctx context.Context, pool *pgxpool.Pool, lease ServerLease) error {
	if pool == nil || !lease.Valid() {
		return errors.New("Control Plane lease handover is not configured")
	}
	return InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := LockAdvisory(ctx, tx, lease.handoverKey()); err != nil {
			return WrapError("await Control Plane lease handover", err)
		}
		return nil
	})
}

// handoverKey is the transaction advisory key that orders lease-fenced
// writes against the handover of the lease of the same key.
func (lease ServerLease) handoverKey() AdvisoryKey {
	return NewAdvisoryKey("control-plane-lease-handover", strconv.FormatInt(lease.key, 10))
}

func (lease ServerLease) heldOn(ctx context.Context, db DBTX) (bool, error) {
	var held bool
	if err := db.QueryRow(ctx, leaseHeldSQL, lease.key, lease.pid, lease.started).Scan(&held); err != nil {
		return false, WrapError("prove Control Plane lease", err)
	}
	return held, nil
}
