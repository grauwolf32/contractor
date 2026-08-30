package runstore

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

const runEventNotificationChannel = "contractor_run_events_v1"

// PostgresRunEventListener turns commit-bound PostgreSQL notifications into
// wake-up hints. Consumers must always catch up from workflow_run_events;
// LISTEN/NOTIFY is deliberately not the replay authority.
type PostgresRunEventListener struct {
	pool *pgxpool.Pool
}

func NewPostgresRunEventListener(pool *pgxpool.Pool) (*PostgresRunEventListener, error) {
	if pool == nil {
		return nil, errors.New("PostgreSQL Run event listener requires a pool")
	}
	// LISTEN owns one connection for the process lifetime. Fail startup instead
	// of silently starving every repository when an operator configured a
	// single-connection pool through the pgx connection string.
	if pool.Config().MaxConns < 2 {
		return nil, errors.New("PostgreSQL Run event listener requires a pool with at least two connections")
	}
	return &PostgresRunEventListener{pool: pool}, nil
}

func (l *PostgresRunEventListener) Listen(ctx context.Context, consume func(string)) error {
	if consume == nil {
		return errors.New("PostgreSQL Run event listener requires a consumer")
	}
	connection, err := l.pool.Acquire(ctx)
	if err != nil {
		return fmt.Errorf("acquire PostgreSQL Run event listener: %w", err)
	}
	defer connection.Release()
	if _, err := connection.Exec(ctx, "LISTEN "+runEventNotificationChannel); err != nil {
		return fmt.Errorf("listen for committed WorkflowRun events: %w", err)
	}
	defer func() {
		cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		_, _ = connection.Exec(cleanupCtx, "UNLISTEN "+runEventNotificationChannel)
	}()
	for {
		notification, err := connection.Conn().WaitForNotification(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return fmt.Errorf("wait for committed WorkflowRun event: %w", err)
		}
		if notification.Channel != runEventNotificationChannel ||
			validateOpaque("notification runID", notification.Payload) != nil {
			continue
		}
		consume(notification.Payload)
	}
}
