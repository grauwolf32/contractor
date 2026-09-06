// Package postgres owns the shared pgx pool, transaction, and migration
// boundaries used by Contractor's durable repositories.
package postgres

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

var ErrInvalidDatabaseConfiguration = errors.New("invalid PostgreSQL database configuration")

// PoolOptions contains process-level connection settings. Zero values retain
// pgx connection-count defaults. ConnectTimeout defaults to 5 seconds;
// database-operation budgets use DefaultBudgets rather than unbounded waits.
type PoolOptions struct {
	MaxConnections int32
	MinConnections int32
	ConnectTimeout time.Duration
	Budgets        Budgets
	Logger         *slog.Logger
}

// OpenPool parses databaseURL without logging it, applies bounded pool
// settings, and verifies a connection before returning.
func OpenPool(ctx context.Context, databaseURL string, options PoolOptions) (*pgxpool.Pool, error) {
	if strings.TrimSpace(databaseURL) == "" {
		return nil, fmt.Errorf("database URL is required")
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, ErrInvalidDatabaseConfiguration
	}
	if options.MaxConnections < 0 || options.MinConnections < 0 ||
		options.MaxConnections > 0 && options.MinConnections > options.MaxConnections {
		return nil, fmt.Errorf("invalid PostgreSQL pool connection limits")
	}
	if options.MaxConnections > 0 {
		config.MaxConns = options.MaxConnections
	}
	if options.MinConnections > 0 {
		config.MinConns = options.MinConnections
	}
	if config.MinConns > config.MaxConns {
		return nil, fmt.Errorf("invalid PostgreSQL pool connection limits")
	}
	connectTimeout := options.ConnectTimeout
	if connectTimeout == 0 {
		connectTimeout = 5 * time.Second
	}
	if connectTimeout < 0 {
		return nil, fmt.Errorf("PostgreSQL connect timeout must be positive")
	}
	config.ConnConfig.ConnectTimeout = connectTimeout
	budgets, err := options.Budgets.normalized()
	if err != nil {
		return nil, err
	}
	config.ConnConfig.RuntimeParams["statement_timeout"] = timeoutSetting(budgets.StatementTimeout)
	config.ConnConfig.RuntimeParams["lock_timeout"] = timeoutSetting(budgets.LockTimeout)
	config.ConnConfig.RuntimeParams["idle_in_transaction_session_timeout"] = timeoutSetting(budgets.IdleTransactionTimeout)
	config.ConnConfig.Tracer = &budgetTracer{budgets: budgets, logger: options.Logger}

	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, WrapError("create PostgreSQL pool", err)
	}
	pingCtx, cancel := context.WithTimeout(ctx, budgets.QueryTimeout)
	defer cancel()
	if err := pool.Ping(pingCtx); err != nil {
		pool.Close()
		return nil, WrapError("ping PostgreSQL", err)
	}
	return pool, nil
}

// SQLState returns a PostgreSQL SQLSTATE without exposing server diagnostics.
func SQLState(err error) string {
	var pgError *pgconn.PgError
	if !errors.As(err, &pgError) {
		return ""
	}
	return pgError.Code
}
