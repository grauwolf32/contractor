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

var errControlPlaneLeaseLost = errors.New("Control Plane lease session lost")

type controlPlaneLease struct {
	conn    *pgx.Conn
	mu      sync.Mutex
	private net.Listener
	lost    bool
	close   sync.Once
}

func openControlPlaneLease(ctx context.Context, pool *pgxpool.Pool) (*controlPlaneLease, error) {
	conn, err := pgx.ConnectConfig(ctx, pool.Config().ConnConfig.Copy())
	if err != nil {
		return nil, fmt.Errorf("connect Control Plane lease session: %w", err)
	}
	return &controlPlaneLease{conn: conn}, nil
}

func (lease *controlPlaneLease) tryAcquire(ctx context.Context) (bool, error) {
	var acquired bool
	err := lease.conn.QueryRow(ctx, `SELECT pg_try_advisory_lock($1)`, controlPlaneLeaseKey).Scan(&acquired)
	if err != nil {
		return false, fmt.Errorf("acquire Control Plane lease: %w", err)
	}
	return acquired, nil
}

func (lease *controlPlaneLease) holderPID(ctx context.Context) int32 {
	var pid int32
	_ = lease.conn.QueryRow(ctx, `
SELECT pid FROM pg_locks
WHERE locktype='advisory' AND granted AND objsubid=1
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
		ticker := time.NewTicker(controlPlaneLeasePoll)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
			pingCtx, stopPing := context.WithTimeout(ctx, controlPlaneLeasePoll)
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
	go func() { done <- ServeHandler(standbyCtx, listener, shutdownTimeout, logger, ready) }()
	stop := func() error {
		stopStandby()
		return <-done
	}
	ticker := time.NewTicker(controlPlaneLeasePoll)
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
			pollCtx, cancel := context.WithTimeout(ctx, controlPlaneLeasePoll)
			acquired, err = lease.tryAcquire(pollCtx)
			cancel()
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
