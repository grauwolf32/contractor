package streamline

import (
	"context"
	"testing"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	postgrestest "github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/jackc/pgx/v5/pgxpool"
)

func isolatedStreamlinePool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{})
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}
