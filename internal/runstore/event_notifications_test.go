package runstore

import (
	"context"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresRunEventListenerRequiresCapacityBeyondItsDedicatedConnection(t *testing.T) {
	if _, err := NewPostgresRunEventListener(nil); err == nil {
		t.Fatal("nil PostgreSQL pool was accepted")
	}
	config, err := pgxpool.ParseConfig("postgres://postgres:postgres@127.0.0.1/postgres")
	if err != nil {
		t.Fatal(err)
	}
	config.MaxConns = 1
	pool, err := pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	if _, err := NewPostgresRunEventListener(pool); err == nil {
		t.Fatal("single-connection PostgreSQL pool was accepted for a dedicated listener")
	}
}
