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

	publicapi "github.com/grauwolf32/contractor/internal/httpapi/public"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
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
// the server-side keepalive bound (about half a minute, PER-R6) so a genuinely
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
	// held is set once this session wins the advisory lock; reconnect is only
	// valid while it is false.
	held bool
	// server identifies this session as the lease holder once it holds the
	// lock; lease-fenced writes prove it (PER-R34).
	server persistencepostgres.ServerLease
	close  sync.Once
}

// PostgreSQL keeps a session advisory lock until it notices the session is
// gone. When the active Server's host vanishes without closing the connection,
// server-side keepalives bound that to about half a minute instead of the
// operating system's two-hour default, so a standby can take over. PostgreSQL
// applies tcp_user_timeout only where its operating system has
// TCP_USER_TIMEOUT (Linux) and reads it back as 0 elsewhere (PER-R6). They are
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

// dialControlPlaneLeaseSession copies the working pool's connection settings,
// so the session inherits the configured database budgets (PER-R27):
// database.connectTimeout bounds each connection attempt, the pool's budget
// tracer bounds each statement by database.queryTimeout, and the startup
// parameters carry the server-side statement, lock and idle timeouts. Every
// budget only shortens ctx; none extends an earlier caller deadline. Startup
// calls it under the process context, so these budgets alone bound it there;
// a standby reconnect also bounds it by the tolerance window (PER-R28).
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

// tryAcquire takes the advisory lock without waiting. Once this session holds
// it, the session is identified as the lease holder that lease-fenced writes
// prove (PER-R34); a session that holds the lock but cannot be identified is
// an error, and the caller closes it, which releases the lock.
func (lease *controlPlaneLease) tryAcquire(ctx context.Context) (bool, error) {
	var acquired bool
	err := lease.conn.QueryRow(ctx, `SELECT pg_try_advisory_lock($1)`, controlPlaneLeaseKey).Scan(&acquired)
	if err != nil {
		return false, fmt.Errorf("acquire Control Plane lease: %w", err)
	}
	if !acquired {
		return false, nil
	}
	lease.held = true
	server, err := persistencepostgres.NewServerLease(ctx, lease.conn, controlPlaneLeaseKey)
	if err != nil {
		return false, fmt.Errorf("identify Control Plane lease session: %w", err)
	}
	lease.server = server
	return true, nil
}

// awaitHandover waits, once this Server holds the lease, until every
// lease-fenced write of the previous holder has ended (PER-R34). The working
// pool's database.lockTimeout bounds the wait and database.queryTimeout the
// statement; a failure stops startup like a failed startup lease step
// (PER-R27).
func (lease *controlPlaneLease) awaitHandover(ctx context.Context) error {
	if err := persistencepostgres.AwaitLeaseHandover(ctx, lease.pool, lease.server); err != nil {
		return fmt.Errorf("await Control Plane lease handover: %w", err)
	}
	return nil
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
		// Closing the session alone releases the lock only when PostgreSQL
		// ends its backend, which happens after the client disconnects, so a
		// standby polling right after a normal exit could still find it held.
		// Unlocking first hands the lease over before Close returns; on a lost
		// session the unlock fails and release falls back to the backend exit.
		if lease.held {
			_, _ = lease.conn.Exec(ctx, `SELECT pg_advisory_unlock($1)`, controlPlaneLeaseKey)
		}
		_ = lease.conn.Close(ctx)
	})
}

// newStandbyHandler serves a standby Server's listener (PER-R3): public
// health, 503 readiness and, for every /v1 path, the retryable
// server_standby error envelope instead of a plain-text 404 that a client
// would read as a missing resource.
func newStandbyHandler() http.Handler {
	return newProcessHandler(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, "{\"status\":\"unavailable\"}\n")
	}), publicapi.NewStandbyHandler())
}

// awaitControlPlaneLease takes the startup lease steps under the process
// context: connecting the session, configuring its keepalives, the first
// acquisition and, when the lease is held elsewhere, the holder lookup. The
// inherited database budgets bound each of them, not the tolerance window; a
// failed step other than the holder lookup stops startup (PER-R27). Once
// standing by, each acquisition probe runs within the tolerance window
// (PER-R28).
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
		if err := lease.awaitHandover(ctx); err != nil {
			return nil, err
		}
		owned = true
		return lease, nil
	}
	logger.Warn("Control Plane lease is held by another Server; standing by", "holder_pid", lease.holderPID(ctx))
	listener, err := net.Listen("tcp", publicAddress)
	if err != nil {
		return nil, fmt.Errorf("listen for standby readiness on %q: %w", publicAddress, err)
	}
	standbyCtx, stopStandby := context.WithCancel(ctx)
	ready := newStandbyHandler()
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
			return nil, standbyListenerStopped(ctx, serveErr)
		case <-ticker.C:
			acquired, err = lease.acquireOrReconnect(ctx, logger)
			if err != nil {
				return nil, errors.Join(err, stop())
			}
			if acquired {
				// The standby keeps answering 503 while the previous
				// holder's lease-fenced writes end.
				if err := lease.awaitHandover(ctx); err != nil {
					return nil, errors.Join(err, stop())
				}
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
// standbyListenerStopped is how standing by ends when the readiness listener
// stopped with serveErr. Shutdown cancels ctx, which stops the listener too,
// so the standby loop may find the listener's end ready together with
// ctx.Done and select either: once ctx is done the standby reports its
// cancellation, and the process exits without an error (FAIL-PER-2). Only a
// listener that stopped while ctx is live stops the Server with an error.
func standbyListenerStopped(ctx context.Context, serveErr error) error {
	if err := ctx.Err(); err != nil {
		return errors.Join(err, serveErr)
	}
	if serveErr == nil {
		serveErr = errors.New("listener exited before lease acquisition")
	}
	return fmt.Errorf("standby readiness listener stopped: %w", serveErr)
}

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
	if lease.held {
		// The session holds the lock, which a reconnect would drop
		// unnoticed; the caller closes it instead.
		return false, err
	}
	logger.Warn("Control Plane lease acquisition attempt failed; retrying", "error", err)
	if reconnectErr := lease.reconnect(ctx); reconnectErr != nil && ctx.Err() == nil {
		logger.Warn("Control Plane lease session reconnect failed; retrying", "error", reconnectErr)
	}
	return false, nil
}
