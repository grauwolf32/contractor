// Package postgres owns the shared pgx pool, transaction, and migration
// boundaries used by Contractor's durable repositories.
package postgres

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

var ErrInvalidDatabaseConfiguration = errors.New("invalid PostgreSQL database configuration")

// PoolOptions contains process-level connection settings. Zero values retain
// pgx defaults except for ConnectTimeout, whose zero value means 5 seconds.
type PoolOptions struct {
	MaxConnections int32
	MinConnections int32
	ConnectTimeout time.Duration
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
	connectTimeout := options.ConnectTimeout
	if connectTimeout == 0 {
		connectTimeout = 5 * time.Second
	}
	if connectTimeout < 0 {
		return nil, fmt.Errorf("PostgreSQL connect timeout must be positive")
	}
	config.ConnConfig.ConnectTimeout = connectTimeout

	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("create PostgreSQL pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping PostgreSQL: %w", err)
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
