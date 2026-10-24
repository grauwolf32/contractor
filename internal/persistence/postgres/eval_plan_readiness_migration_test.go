package postgres

import (
	"crypto/sha256"
	"encoding/hex"
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
	previous, _ := embeddedMigration(t, "eval_plan_readiness")
	installMigrationPrefix(t, ctx, pool, previous)
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
			// A NUL in a free setup string must not stop the backfill.
			setup, err := json.Marshal(map[string]any{"comparison": c.Comparison, "note": "setup\x00note"})
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

// embeddedMigration returns the embedded migration named suffix and the
// version of the one before it, so a test installs the schema the migration
// upgrades whatever number the migration takes.
func embeddedMigration(t *testing.T, suffix string) (previous int64, contents string) {
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
			return available[i-1].version, string(item.contents)
		}
	}
	t.Fatalf("no embedded migration is named %s", suffix)
	return 0, ""
}

// ceilingPlan returns a native plan document at the native member ceiling:
// the conformance plan with 1,000 members over both arms, a tenth of them
// unsupported and a tenth blocked, and 1,024 pinned dimensions per arm. A
// disjoint plan pins different dimensions in each arm, the most readiness pins
// a plan can have; otherwise both arms pin the same dimensions with equal,
// different, null, NUL-carrying and operator-supplied values.
func ceilingPlan(t *testing.T, base []byte, id string, disjoint bool) []byte {
	t.Helper()
	var plan map[string]any
	if err := json.Unmarshal(base, &plan); err != nil {
		t.Fatal(err)
	}
	template := plan["members"].([]any)[0].(map[string]any)
	members := make([]any, 0, evaldomain.MaxNativeMembers)
	order := make([]any, 0, evaldomain.MaxNativeMembers)
	for n := range evaldomain.MaxNativeMembers {
		member := map[string]any{}
		for k, v := range template {
			member[k] = v
		}
		digest := sha256.Sum256(fmt.Appendf(nil, "%s/%d", id, n))
		member["member_id"] = hex.EncodeToString(digest[:])
		member["variant_id"] = []string{"a", "b"}[n%2]
		member["case_id"] = fmt.Sprintf("case-%d", n/200)
		member["sample"] = n/2%100 + 1
		member["eligibility"], member["reason"] = "eligible", nil
		switch n % 10 {
		case 2:
			member["eligibility"], member["reason"] = "unsupported", "Missing supported input"
		case 7:
			member["eligibility"], member["reason"] = "blocked", "Required pin is unavailable"
		}
		members = append(members, member)
		order = append(order, member["member_id"])
	}
	pins := map[string]map[string]any{"a": {}, "b": {}}
	for k := range 1024 {
		value := fmt.Sprintf("value-%d", k)
		if k%5 == 0 {
			value = fmt.Sprintf("value\x00%d", k)
		}
		origin := "observed"
		if k%7 == 0 {
			origin = "operator_supplied"
		}
		if disjoint {
			pins["a"][fmt.Sprintf("dimension-a-%04d", k)] = map[string]any{"value": value, "origin": origin}
			pins["b"][fmt.Sprintf("dimension-b-%04d", k)] = map[string]any{"value": value, "origin": origin}
			continue
		}
		other := any(value)
		switch k % 3 {
		case 1:
			other = value + "-other"
		case 2:
			other = nil
		}
		dimension := fmt.Sprintf("dimension-%04d", k)
		pins["a"][dimension] = map[string]any{"value": value, "origin": origin}
		pins["b"][dimension] = map[string]any{"value": other, "origin": "observed"}
	}
	plan["members"], plan["execution_order"], plan["pins"] = members, order, pins
	document, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	return document
}

// At the native member ceiling with the most pins a readiness can hold, the
// backfill derives what evaldomain.NativeReadiness derives and parses each
// plan's document and setup exactly once: the statement plan converts each
// stored column a single time, and every pin and member lookup reads that
// parsed value instead of parsing the document again (AGENTS 3.3). The count
// is structural, not a timing.
func TestSharedPostgresEvalPlanReadinessMigrationParsesEachPlanOnceAtTheNativeCeiling(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, migration := embeddedMigration(t, "eval_plan_readiness")
	installMigrationPrefix(t, ctx, pool, previous)
	if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES('evaluation','owner','evaluation','Evaluation','create','sha256:'||repeat('a',64))`); err != nil {
		t.Fatal(err)
	}
	base := contractstest.ReadFile(t, "api", "testdata", "evals", "valid", "portable-plan.json")
	comparison := evaldomain.Comparison{Baseline: "a", Candidate: "b", RequiredEqual: []string{"dimension-0003", "dimension-a-0001", "dimension-b-0002", "scorers"}}
	setup, err := json.Marshal(map[string]any{"comparison": comparison, "note": "setup\x00note"})
	if err != nil {
		t.Fatal(err)
	}
	plans := map[string][]byte{}
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for i := range 4 {
			id := fmt.Sprintf("ceiling-%d", i)
			plans[id] = ceilingPlan(t, base, id, i%2 == 0)
			insertFrozenEvalPlan(t, tx, id, "server", "playground.plan/v1", plans[id], setup)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	update := strings.Index(migration, "\nUPDATE eval_frozen_plans")
	end := strings.Index(migration[update:], ";\n")
	if update < 0 || end < 0 {
		t.Fatal("the readiness migration has no backfill UPDATE")
	}
	var explained []string
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, migration[:update]); err != nil {
			return err
		}
		rows, err := tx.Query(ctx, "EXPLAIN (VERBOSE, COSTS OFF)"+migration[update:update+end])
		if err != nil {
			return err
		}
		explained, err = pgx.CollectRows(rows, pgx.RowTo[string])
		if err != nil {
			return err
		}
		return errors.New("roll back the explained prefix")
	})
	if len(explained) == 0 {
		t.Fatalf("explain the backfill: %v", err)
	}
	shape := strings.Join(explained, "\n")
	if parses := strings.Count(shape, "convert_from(") + strings.Count(shape, "eval_plan_readiness_json("); parses != 2 {
		t.Fatalf("the backfill converts stored documents %d times, want once for the document and once for the setup:\n%s", parses, shape)
	}

	// The widest schema-valid readiness, as the backfill writes it, stays
	// within the column bound (LIM-EVM-19).
	widest := evaldomain.Readiness{Pins: make([]evaldomain.ReadinessPin, 2048), Arms: make([]evaldomain.ReadinessArm, 2)}
	for i := range widest.Pins {
		widest.Pins[i] = evaldomain.ReadinessPin{Dimension: fmt.Sprintf("d%0127d", i), BaselineOrigin: "producer-supplied", CandidateOrigin: "producer-supplied", Status: "unavailable"}
	}
	for i := range widest.Arms {
		widest.Arms[i] = evaldomain.ReadinessArm{VariantID: strings.Repeat(string(rune('a'+i)), 128), Expected: 10000, Eligible: 10000, Unsupported: 10000, Blocked: 10000}
	}
	encoded, err := json.Marshal(widest)
	if err != nil {
		t.Fatal(err)
	}
	var backfilled int
	if err = pool.QueryRow(ctx, `SELECT octet_length(convert_to(convert_from($1::bytea,'UTF8')::jsonb::text,'UTF8'))`, encoded).Scan(&backfilled); err != nil {
		t.Fatal(err)
	}
	if len(encoded) != 543169 || backfilled != 563670 || backfilled > 1<<20 {
		t.Fatalf("the widest readiness encodes to %d bytes and backfills to %d", len(encoded), backfilled)
	}

	if _, err = ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	for id, document := range plans {
		var stored []byte
		if err = pool.QueryRow(ctx, `SELECT readiness FROM eval_frozen_plans WHERE experiment_id=$1`, id).Scan(&stored); err != nil {
			t.Fatal(err)
		}
		var got evaldomain.Readiness
		if err = strictjson.Decode(stored, &got); err != nil {
			t.Fatalf("%s: backfilled readiness: %v", id, err)
		}
		derived, err := evaldomain.NativeReadiness(evaldomain.Restore("playground.plan/v1", document, ""), comparison)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, derived) {
			t.Fatalf("%s: backfilled readiness differs from NativeReadiness", id)
		}
		if err = evaldomain.Validate("Readiness", stored); err != nil {
			t.Fatalf("%s: backfilled readiness is not schema-valid: %v", id, err)
		}
		pins := map[bool]int{true: 2048, false: 1024}[id == "ceiling-0" || id == "ceiling-2"]
		if len(got.Pins) != pins || got.Arms[0].Expected != 500 || got.Arms[0].Unsupported != 100 || got.Arms[1].Blocked != 100 {
			t.Fatalf("%s: %d pins, arms %+v", id, len(got.Pins), got.Arms)
		}
		t.Logf("%s: %d document bytes, %d pins, %d readiness bytes", id, len(document), len(got.Pins), len(stored))
	}
}
