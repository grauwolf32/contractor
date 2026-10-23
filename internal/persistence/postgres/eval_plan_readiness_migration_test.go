package postgres

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// insertFrozenEvalPlan stores an experiment of the evaluation Project with
// its frozen plan and expected members as the schema before the readiness
// migration held them: one member row per plan member for a native plan, one
// for a registration.
func insertFrozenEvalPlan(t *testing.T, tx pgx.Tx, id, mode, kind string, document, setup []byte) {
	t.Helper()
	ctx := t.Context()
	members := 1
	if kind == "playground.plan/v1" {
		var plan struct {
			Members []json.RawMessage `json:"members"`
		}
		if err := json.Unmarshal(document, &plan); err != nil {
			t.Fatal(err)
		}
		members = len(plan.Members)
	}
	if _, err := tx.Exec(ctx, `INSERT INTO eval_experiments(experiment_id,owner_id,project_id,portable_id,control_mode,name,state,expected_count,max_in_flight,wall_ms)
VALUES($1,'owner','evaluation',$1,$2,'Retained plan','ready',$3,1,1000)`, id, mode, members); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.Exec(ctx, `INSERT INTO eval_frozen_plans(experiment_id,document_kind,document,setup,plan_sha256)
VALUES($1,$2,$3,$4,'sha256:'||encode(sha256($3),'hex'))`, id, kind, document, setup); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.Exec(ctx, `INSERT INTO eval_members(experiment_id,member_id,pair_id,ordinal,suite_id,case_id,sample,variant_id,
case_sha256,binding_sha256,eligibility,execution_kind,recipe,submission_key)
SELECT $1, encode(sha256(convert_to($1||'/'||n,'UTF8')),'hex'), encode(sha256(convert_to($1||'/'||n,'UTF8')),'hex'), n-1,
       'suite','case-'||n,1,'a','sha256:'||repeat('a',64),'sha256:'||repeat('b',64),'eligible','run','{}','key-'||$1||'-'||n
  FROM generate_series(1,$2::integer) AS n`, id, members); err != nil {
		t.Fatal(err)
	}
}

// Migration eval_plan_readiness backfills the readiness of every retained
// native plan with exactly what evaldomain.NativeReadiness, and so the
// experiment view before the migration, derives from it (the shared fixture),
// leaves registrations without one and changes no other plan or member data.
// Afterwards the plan row is immutable again and the readiness is required
// exactly for native plans.
func TestSharedPostgresEvalPlanReadinessMigrationBackfillsNativePlans(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	installMigrationPrefix(t, ctx, pool, versionBefore(t, "eval_plan_readiness"))
	var fixtures struct {
		Cases []struct {
			Name       string                `json:"name"`
			Plan       json.RawMessage       `json:"plan"`
			Comparison evaldomain.Comparison `json:"comparison"`
			Readiness  json.RawMessage       `json:"readiness"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "evals", "readiness-cases.json"), &fixtures); err != nil {
		t.Fatal(err)
	}
	registration := contractstest.ReadFile(t, "api", "testdata", "evals", "valid", "registration.json")
	if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES('evaluation','owner','evaluation','Evaluation','create','sha256:'||repeat('a',64))`); err != nil {
		t.Fatal(err)
	}
	err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for i, c := range fixtures.Cases {
			setup, err := json.Marshal(map[string]any{"comparison": c.Comparison})
			if err != nil {
				return err
			}
			insertFrozenEvalPlan(t, tx, fmt.Sprintf("native-%d", i), "server", "playground.plan/v1", c.Plan, setup)
		}
		insertFrozenEvalPlan(t, tx, "external", "external", "ExternalRegistration", registration, []byte(`{}`))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	const retained = `SELECT string_agg(concat_ws('|', experiment_id, document_kind, encode(document,'hex'), document_sha256,
    plan_sha256, encode(setup,'hex'), created_at), ',' ORDER BY experiment_id) FROM eval_frozen_plans`
	const retainedMembers = `SELECT string_agg(concat_ws('|', experiment_id, member_id, pair_id, ordinal, eligibility, encode(recipe,'hex'),
    submission_key), ',' ORDER BY experiment_id, member_id) FROM eval_members`
	var plansBefore, membersBefore string
	if err = pool.QueryRow(ctx, retained).Scan(&plansBefore); err != nil {
		t.Fatal(err)
	}
	if err = pool.QueryRow(ctx, retainedMembers).Scan(&membersBefore); err != nil {
		t.Fatal(err)
	}

	if _, err = ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}

	for i, c := range fixtures.Cases {
		var stored []byte
		if err = pool.QueryRow(ctx, `SELECT readiness FROM eval_frozen_plans WHERE experiment_id=$1`, fmt.Sprintf("native-%d", i)).Scan(&stored); err != nil {
			t.Fatal(err)
		}
		var got, want evaldomain.Readiness
		if err = strictjson.Decode(stored, &got); err != nil {
			t.Fatalf("%s: backfilled readiness %s: %v", c.Name, stored, err)
		}
		if err = strictjson.Decode(c.Readiness, &want); err != nil {
			t.Fatal(err)
		}
		plan, err := evaldomain.Freeze("playground.plan/v1", c.Plan)
		if err != nil {
			t.Fatal(err)
		}
		derived, err := evaldomain.NativeReadiness(plan, c.Comparison)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(derived, want) {
			t.Fatalf("%s: backfilled readiness\n got %+v\nwant %+v", c.Name, got, want)
		}
		if err = evaldomain.Validate("Readiness", stored); err != nil {
			t.Fatalf("%s: backfilled readiness is not schema-valid: %v", c.Name, err)
		}
	}
	var external []byte
	if err = pool.QueryRow(ctx, `SELECT readiness FROM eval_frozen_plans WHERE experiment_id='external'`).Scan(&external); err != nil || external != nil {
		t.Fatalf("registration readiness = %q, %v", external, err)
	}
	var plansAfter, membersAfter string
	if err = pool.QueryRow(ctx, retained).Scan(&plansAfter); err != nil {
		t.Fatal(err)
	}
	if err = pool.QueryRow(ctx, retainedMembers).Scan(&membersAfter); err != nil {
		t.Fatal(err)
	}
	if plansAfter != plansBefore || membersAfter != membersBefore {
		t.Fatal("the migration changed retained plan or member data")
	}

	violation := func(err error, message string) bool {
		var pgErr *pgconn.PgError
		return errors.As(err, &pgErr) && pgErr.Code == SQLStateCheckViolation && strings.Contains(pgErr.Message+pgErr.ConstraintName, message)
	}
	if _, err = pool.Exec(ctx, `UPDATE eval_frozen_plans SET readiness=NULL WHERE experiment_id='native-0'`); !violation(err, "immutable") {
		t.Fatalf("readiness changed after the migration: %v", err)
	}
	if _, err = pool.Exec(ctx, `INSERT INTO eval_experiments(experiment_id,owner_id,project_id,portable_id,control_mode,name,state,expected_count,max_in_flight,wall_ms)
VALUES('late','owner','evaluation','late','server','Late plan','ready',1,1,1000)`); err != nil {
		t.Fatal(err)
	}
	if _, err = pool.Exec(ctx, `INSERT INTO eval_frozen_plans(experiment_id,document_kind,document,setup,plan_sha256)
SELECT 'late',document_kind,document,setup,plan_sha256 FROM eval_frozen_plans WHERE experiment_id='native-0'`); !violation(err, "eval_frozen_plans_readiness_kind_check") {
		t.Fatalf("a native plan without readiness was stored: %v", err)
	}
}

// versionBefore returns the version of the embedded migration that precedes
// the one named suffix, so a test installs the schema its migration upgrades
// whatever number the migration takes.
func versionBefore(t *testing.T, suffix string) int64 {
	t.Helper()
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	for i, item := range available {
		if strings.TrimSuffix(item.name[strings.IndexByte(item.name, '_')+1:], ".sql") == suffix {
			if i == 0 {
				t.Fatalf("migration %s has no predecessor", item.name)
			}
			return available[i-1].version
		}
	}
	t.Fatalf("no embedded migration is named %s", suffix)
	return 0
}
