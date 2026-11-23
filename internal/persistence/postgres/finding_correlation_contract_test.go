package postgres

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPostgresFindingGroupHash(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Vectors []struct {
			ProjectID string
			Algorithm string
			Value     string
			GroupKey  string
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "public", "finding-correlation-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, v := range fixture.Vectors {
		var key string
		if err := pool.QueryRow(ctx, "SELECT contractor_finding_group_key($1,$2,$3)", v.ProjectID, v.Algorithm, v.Value).Scan(&key); err != nil || key != v.GroupKey {
			t.Fatalf("SQL group key=%s want=%s err=%v", key, v.GroupKey, err)
		}
	}
}
