package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The lock is scoped by PostgreSQL to its database. A dedicated session owns
// it for the entire active Server lifetime; a pooled or transaction-scoped
// connection would release ownership between operations.
const controlPlaneLeaseKey int64 = 0x636f6e7472616374
const controlPlaneLeasePoll = time.Second

// controlPlaneLeaseTolerance bounds how long a single liveness probe may run
// before the lease is declared lost. A probe that merely stalls (no bytes for
// a short while) completes late instead of being cancelled, so a transient
// client-to-database stall does not cancel the Server. It is kept well below
// the server-side keepalive bound (tcp_user_timeout 30 s) so a genuinely
// disconnected active Server stops before PostgreSQL releases the advisory
// lock and a standby can acquire it.
const controlPlaneLeaseTolerance = 10 * time.Second

var errControlPlaneLeaseLost = errors.New("Control Plane lease session lost")

type controlPlaneLease struct {
	pool      *pgxpool.Pool
	poll      time.Duration
	tolerance time.Duration
	conn      *pgx.Conn
	mu        sync.Mutex
	private   net.Listener
	lost      bool
	close     sync.Once
}

// PostgreSQL keeps a session advisory lock until it notices the session is
// gone. When the active Server's host vanishes without closing the connection,
// server-side keepalives bound that to about half a minute instead of the
// operating system's two-hour default, so a standby can take over. They are
// set after connecting because session poolers reject unknown startup
// parameters.
const controlPlaneLeaseKeepalives = `
SELECT set_config('tcp_keepalives_idle', '10', false),
       set_config('tcp_keepalives_interval', '5', false),
       set_config('tcp_keepalives_count', '3', false),
       set_config('tcp_user_timeout', '30000', false)`

func openControlPlaneLease(ctx context.Context, pool *pgxpool.Pool) (*controlPlaneLease, error) {
	conn, err := dialControlPlaneLeaseSession(ctx, pool)
	if err != nil {
		return nil, err
	}
	return &controlPlaneLease{
		pool: pool, poll: controlPlaneLeasePoll, tolerance: controlPlaneLeaseTolerance, conn: conn,
	}, nil
}

func dialControlPlaneLeaseSession(ctx context.Context, pool *pgxpool.Pool) (*pgx.Conn, error) {
	conn, err := pgx.ConnectConfig(ctx, pool.Config().ConnConfig.Copy())
	if err != nil {
		return nil, fmt.Errorf("connect Control Plane lease session: %w", err)
	}
	if _, err := conn.Exec(ctx, controlPlaneLeaseKeepalives); err != nil {
		_ = conn.Close(context.Background())
		return nil, fmt.Errorf("configure Control Plane lease session keepalives: %w", err)
	}
	return conn, nil
}

// reconnect replaces the lease session connection. It is only safe before the
// advisory lock is held: a session advisory lock is bound to the exact backend
// session, so a reconnected session never inherits a lock the old one held.
// The standby acquisition loop uses it to survive a transient acquisition
// stall that drops its not-yet-locking probe connection.
func (lease *controlPlaneLease) reconnect(ctx context.Context) error {
	_ = lease.conn.Close(context.Background())
	// Bound the reconnect so a sustained stall does not block the polling loop;
	// a failed reconnect leaves the closed connection in place and the next
	// poll retries.
	dialCtx, cancel := context.WithTimeout(ctx, lease.tolerance)
	defer cancel()
	conn, err := dialControlPlaneLeaseSession(dialCtx, lease.pool)
	if err != nil {
		return err
	}
	lease.conn = conn
	return nil
}

func (lease *controlPlaneLease) tryAcquire(ctx context.Context) (bool, error) {
	var acquired bool
	err := lease.conn.QueryRow(ctx, `SELECT pg_try_advisory_lock($1)`, controlPlaneLeaseKey).Scan(&acquired)
	if err != nil {
		return false, fmt.Errorf("acquire Control Plane lease: %w", err)
	}
	return acquired, nil
}

// holderPID reports the session holding this database's lease. pg_locks lists
// locks of every database in the cluster, and advisory locks are scoped per
// database, so the same key may also be held for another database.
func (lease *controlPlaneLease) holderPID(ctx context.Context) int32 {
	var pid int32
	_ = lease.conn.QueryRow(ctx, `
SELECT pid FROM pg_locks
WHERE locktype='advisory' AND granted AND objsubid=1
  AND database = (SELECT oid FROM pg_database WHERE datname = current_database())
  AND classid::bigint = ($1::bigint >> 32)
  AND objid::bigint = ($1::bigint & 4294967295)
ORDER BY pid LIMIT 1`, controlPlaneLeaseKey).Scan(&pid)
	return pid
}

func (lease *controlPlaneLease) attachPrivate(listener net.Listener) error {
	lease.mu.Lock()
	defer lease.mu.Unlock()
	if lease.lost {
		_ = listener.Close()
		return errors.New("Control Plane lease was lost before private listener startup")
	}
	lease.private = listener
	return nil
}

func (lease *controlPlaneLease) watch(ctx context.Context, cancel context.CancelCauseFunc) <-chan struct{} {
	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(lease.poll)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
			// A single probe may run up to the tolerance window. A transient
			// stall then resolves late instead of cancelling the probe and
			// dropping the lock-holding session, so it does not stop the
			// Server. A genuinely lost session fails within the window, which
			// stays below the server-side keepalive bound, so the Server stops
			// before PostgreSQL releases the lock to a standby. The lock is
			// bound to this exact session, so a failed probe is unrecoverable:
			// the session is declared lost rather than reconnected.
			pingCtx, stopPing := context.WithTimeout(ctx, lease.tolerance)
			err := lease.conn.Ping(pingCtx)
			stopPing()
			if err == nil || ctx.Err() != nil {
				continue
			}
			lease.mu.Lock()
			lease.lost = true
			if lease.private != nil {
				_ = lease.private.Close()
			}
			lease.mu.Unlock()
			cancel(errors.Join(errControlPlaneLeaseLost, err))
			return
		}
	}()
	return done
}

func (lease *controlPlaneLease) Close() {
	lease.close.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_ = lease.conn.Close(ctx)
	})
}

func awaitControlPlaneLease(ctx context.Context, pool *pgxpool.Pool, publicAddress string, shutdownTimeout time.Duration, logger *slog.Logger) (*controlPlaneLease, error) {
	lease, err := openControlPlaneLease(ctx, pool)
	if err != nil {
		return nil, err
	}
	owned := false
	defer func() {
		if !owned {
			lease.Close()
		}
	}()
	acquired, err := lease.tryAcquire(ctx)
	if err != nil {
		return nil, err
	}
	if acquired {
		owned = true
		return lease, nil
	}
	logger.Warn("Control Plane lease is held by another Server; standing by", "holder_pid", lease.holderPID(ctx))
	listener, err := net.Listen("tcp", publicAddress)
	if err != nil {
		return nil, fmt.Errorf("listen for standby readiness on %q: %w", publicAddress, err)
	}
	standbyCtx, stopStandby := context.WithCancel(ctx)
	ready := newProcessHandler(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, "{\"status\":\"unavailable\"}\n")
	}))
	done := make(chan error, 1)
	go func() { done <- serveHandler(standbyCtx, listener, shutdownTimeout, logger, ready) }()
	stop := func() error {
		stopStandby()
		return <-done
	}
	ticker := time.NewTicker(lease.poll)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil, errors.Join(ctx.Err(), stop())
		case serveErr := <-done:
			stopStandby()
			if serveErr == nil {
				serveErr = errors.New("listener exited before lease acquisition")
			}
			return nil, fmt.Errorf("standby readiness listener stopped: %w", serveErr)
		case <-ticker.C:
			acquired, err = lease.acquireOrReconnect(ctx, logger)
			if err != nil {
				return nil, errors.Join(err, stop())
			}
			if acquired {
				if err := stop(); err != nil {
					return nil, err
				}
				owned = true
				logger.Info("Control Plane lease acquired after standby")
				return lease, nil
			}
		}
	}
}

// acquireOrReconnect attempts one bounded standby acquisition. A transient
// stall can cancel the probe and drop this not-yet-locking connection; because
// the standby holds no lock, it reconnects and reports no acquisition rather
// than failing, so a single acquisition timeout never exits the process. It
// returns an error only when the outer context is done.
func (lease *controlPlaneLease) acquireOrReconnect(ctx context.Context, logger *slog.Logger) (bool, error) {
	pollCtx, cancel := context.WithTimeout(ctx, lease.tolerance)
	acquired, err := lease.tryAcquire(pollCtx)
	cancel()
	if err == nil {
		return acquired, nil
	}
	if ctx.Err() != nil {
		return false, ctx.Err()
	}
	logger.Warn("Control Plane lease acquisition attempt failed; retrying", "error", err)
	if reconnectErr := lease.reconnect(ctx); reconnectErr != nil && ctx.Err() == nil {
		logger.Warn("Control Plane lease session reconnect failed; retrying", "error", reconnectErr)
	}
	return false, nil
}
