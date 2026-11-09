package postgres

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestSharedPostgresRuntimeSnapshotCredentialOrderAcrossCollations(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	var fixture struct {
		Cases []struct {
			Name  string   `json:"name"`
			IDs   []string `json:"ids"`
			Valid bool     `json:"valid"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "runtime-snapshot-credential-sets.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, locale := range []string{"C", "en-US"} {
		t.Run(locale, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
			defer cancel()
			pool := snapshotOrderPool(t, ctx, dsn, locale)
			if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
				t.Fatal(err)
			}
			if _, err := pool.Exec(ctx, `INSERT INTO workflow_runs(run_id,owner_id,workflow_name,workflow_version,workflow_schema_version,workflow_snapshot,parameters,runtime_labels,runtime_config_snapshot,scheduler_lane_class,state,state_reason_code)
VALUES ('before','owner','workflow','1','v1','{}','{}','{}',$1::jsonb,'model','initializing','before')`, builtInRunRuntimeSnapshot); err != nil {
				t.Fatal(err)
			}
			var before string
			if err := pool.QueryRow(ctx, `SELECT runtime_config_snapshot::text FROM workflow_runs WHERE run_id='before'`).Scan(&before); err != nil {
				t.Fatal(err)
			}
			candidate := func(list string, ids []string) []byte {
				t.Helper()
				var snapshot map[string]any
				if err := json.Unmarshal([]byte(builtInRunRuntimeSnapshot), &snapshot); err != nil {
					t.Fatal(err)
				}
				snapshot[list] = ids
				data, err := json.Marshal(snapshot)
				if err != nil {
					t.Fatal(err)
				}
				return data
			}
			for _, c := range fixture.Cases {
				for _, list := range []string{"llmCredentialIds", "runtimeCredentialIds"} {
					var accepted bool
					if err := pool.QueryRow(ctx, `SELECT contractor_valid_run_runtime_snapshot($1::jsonb, '{}'::text[])`, candidate(list, c.IDs)).Scan(&accepted); err != nil {
						t.Fatal(err)
					}
					if accepted != c.Valid {
						t.Fatalf("%s/%s accepted=%t, want %t", list, c.Name, accepted, c.Valid)
					}
				}
			}
			if _, err := pool.Exec(ctx, `INSERT INTO workflow_runs(run_id,owner_id,workflow_name,workflow_version,workflow_schema_version,workflow_snapshot,parameters,runtime_labels,runtime_config_snapshot,scheduler_lane_class,state,state_reason_code)
VALUES ('after','owner','workflow','1','v1','{}','{}','{}',$1::jsonb,'model','initializing','after')`, candidate("runtimeCredentialIds", []string{"a-1", "a_2"})); err != nil {
				t.Fatal(err)
			}
			var after string
			if err := pool.QueryRow(ctx, `SELECT runtime_config_snapshot::text FROM workflow_runs WHERE run_id='before'`).Scan(&after); err != nil {
				t.Fatal(err)
			}
			if before != after {
				t.Fatal("credential validation rewrote the existing snapshot")
			}
			if err := VerifyMigrations(ctx, pool); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// A separate database exercises the installed function's actual default
// collation. An expression-level COLLATE cannot change a jsonb parameter.
func snapshotOrderPool(t *testing.T, ctx context.Context, dsn, locale string) *pgxpool.Pool {
	t.Helper()
	config, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgx.ConnectConfig(ctx, config.ConnConfig.Copy())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = admin.Close(context.Background()) })
	name := pgx.Identifier{"credential_order_" + strings.ToLower(rand.Text())}.Sanitize()
	options := ` TEMPLATE template0 ENCODING 'UTF8' LOCALE_PROVIDER libc LC_COLLATE 'C' LC_CTYPE 'C'`
	if locale == "en-US" {
		options = ` TEMPLATE template0 ENCODING 'UTF8' LOCALE_PROVIDER icu ICU_LOCALE 'en-US'`
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		if _, err := admin.Exec(cleanup, "DROP DATABASE IF EXISTS "+name+" WITH (FORCE)"); err != nil {
			t.Error(err)
		}
	})
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+name+options); err != nil {
		t.Fatal(err)
	}
	config.ConnConfig.Database = strings.Trim(name, `"`)
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	return pool
}
