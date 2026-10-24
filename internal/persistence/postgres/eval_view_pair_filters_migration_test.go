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
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// Migration eval_view_pair_filters backfills the member columns of every
// retained view pair row from the row's own document and the frozen ordinals
// of its two members, in every experiment and generation, also when a free
// string of the document holds a NUL escape or the escaped text of one, and
// changes no other column. Afterwards the row is immutable again and every
// column but the execution state is required.
func TestSharedPostgresEvalViewPairFiltersMigrationBackfillsMemberColumns(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "eval_view_pair_filters")
	installMigrationPrefix(t, ctx, pool, previous)
	var trace struct {
		Items []evaldomain.Pair `json:"items"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "evals", "valid", "pair-page.json"), &trace); err != nil {
		t.Fatal(err)
	}
	pairs := trace.Items
	nul, escaped := "a NUL \x00 here", `an escaped \u0000 text`
	pairs[0].A.Member.Reason = &nul
	pairs[1].B.Member.Reason = &escaped
	pairs[2].A.Execution = nil
	pairs[3].B.Conflicting = true
	pairs[3].B.Member.Eligibility = "blocked"
	registration := contractstest.ReadFile(t, "api", "testdata", "evals", "valid", "registration.json")
	if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES('evaluation','owner','evaluation','Evaluation','create','sha256:'||repeat('a',64))`); err != nil {
		t.Fatal(err)
	}
	// Both experiments hold the same members at different ordinals, and the
	// first one two generations of its view.
	ordinal := map[string]func(i int) int{
		"first":  func(i int) int { return 2*len(pairs) - 1 - i },
		"second": func(i int) int { return (i + 3) % (2 * len(pairs)) },
	}
	generations := map[string][]int64{"first": {1, 2}, "second": {1}}
	err := InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, id := range []string{"first", "second"} {
			if _, err := tx.Exec(ctx, `INSERT INTO eval_experiments(experiment_id,owner_id,project_id,portable_id,control_mode,name,state,expected_count,max_in_flight,wall_ms)
VALUES($1,'owner','evaluation',$1,'external','Retained view','finished',$2,1,1000)`, id, 2*len(pairs)); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `INSERT INTO eval_frozen_plans(experiment_id,document_kind,document,setup,plan_sha256)
VALUES($1,'ExternalRegistration',$2,'{}','sha256:'||encode(sha256($2),'hex'))`, id, registration); err != nil {
				return err
			}
			for i, pair := range pairs {
				for j, side := range []evaldomain.MemberView{pair.A, pair.B} {
					if _, err := tx.Exec(ctx, `INSERT INTO eval_members(experiment_id,member_id,pair_id,ordinal,suite_id,case_id,sample,variant_id,
case_sha256,binding_sha256,eligibility,execution_kind,recipe,submission_key)
VALUES($1,$2,$3,$4,$5,$6,$7,$8,'sha256:'||repeat('a',64),'sha256:'||repeat('b',64),$9,'run','{}',$1||'-'||$2)`,
						id, side.Member.ID, pair.ID, ordinal[id](2*i+j), pair.SuiteID, pair.CaseID, pair.Sample, side.Member.VariantID, side.Member.Eligibility); err != nil {
						return err
					}
				}
			}
			for _, generation := range generations[id] {
				for i, pair := range pairs {
					document, err := json.Marshal(pair)
					if err != nil {
						return err
					}
					if _, err = tx.Exec(ctx, `INSERT INTO eval_view_pairs(experiment_id,generation,ordinal,pair_id,suite_id,document,regression,unresolved,
tokens_a,tokens_b,duration_a,duration_b,complete_a,complete_b)
VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,NULL,$10,20,$11,$12)`, id, generation, i, pair.ID, pair.SuiteID, document, i%2 == 0, i%3 == 0,
						float64(10*i), float64(1000+i), i != 1, i != 2); err != nil {
						return err
					}
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	const retained = `SELECT string_agg(concat_ws('|', experiment_id, generation, ordinal, pair_id, suite_id, encode(document,'hex'),
    regression, unresolved, tokens_a, tokens_b, duration_a, duration_b, complete_a, complete_b), ',' ORDER BY experiment_id, generation, ordinal)
FROM eval_view_pairs`
	var before string
	if err = pool.QueryRow(ctx, retained).Scan(&before); err != nil {
		t.Fatal(err)
	}

	if _, err = ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}

	var after string
	if err = pool.QueryRow(ctx, retained).Scan(&after); err != nil || after != before {
		t.Fatalf("the migration changed retained pair columns: %v", err)
	}
	type sides struct {
		Ordinals    [2]int
		Variants    [2]string
		Eligibility [2]string
		States      [2]*string
		Assessments [2]string
		Conflicting [2]bool
	}
	rows, err := pool.Query(ctx, `SELECT experiment_id, generation, ordinal, member_ordinal_a, member_ordinal_b, variant_a, variant_b,
    eligibility_a, eligibility_b, execution_state_a, execution_state_b, assessment_a, assessment_b, conflicting_a, conflicting_b
FROM eval_view_pairs ORDER BY experiment_id, generation, ordinal`)
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for rows.Next() {
		var id string
		var generation int64
		var pairOrdinal int
		var got sides
		if err = rows.Scan(&id, &generation, &pairOrdinal, &got.Ordinals[0], &got.Ordinals[1], &got.Variants[0], &got.Variants[1],
			&got.Eligibility[0], &got.Eligibility[1], &got.States[0], &got.States[1], &got.Assessments[0], &got.Assessments[1],
			&got.Conflicting[0], &got.Conflicting[1]); err != nil {
			t.Fatal(err)
		}
		pair := pairs[pairOrdinal]
		var want sides
		for j, side := range []evaldomain.MemberView{pair.A, pair.B} {
			want.Ordinals[j] = ordinal[id](2*pairOrdinal + j)
			want.Variants[j], want.Eligibility[j] = side.Member.VariantID, side.Member.Eligibility
			want.Assessments[j], want.Conflicting[j] = side.Assessment, side.Conflicting
			if side.Execution != nil {
				state := side.Execution.State
				want.States[j] = &state
			}
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s generation %d pair %d backfilled %+v, want %+v", id, generation, pairOrdinal, got, want)
		}
		checked++
	}
	if err = rows.Err(); err != nil {
		t.Fatal(err)
	}
	if checked != 3*len(pairs) {
		t.Fatalf("checked %d pair rows, want %d", checked, 3*len(pairs))
	}

	var function *string
	if err = pool.QueryRow(ctx, `SELECT to_regprocedure('eval_view_pair_sides(bytea)')::text`).Scan(&function); err != nil || function != nil {
		t.Fatalf("the backfill function remains: %v %v", function, err)
	}
	stateOf := func(err error, code, message string) bool {
		var pgErr *pgconn.PgError
		return errors.As(err, &pgErr) && pgErr.Code == code && strings.Contains(pgErr.Message+pgErr.ColumnName, message)
	}
	if _, err = pool.Exec(ctx, `UPDATE eval_view_pairs SET variant_a='b' WHERE experiment_id='first'`); !stateOf(err, SQLStateCheckViolation, "immutable") {
		t.Fatalf("a pair row changed after the migration: %v", err)
	}
	columns := []string{"member_ordinal_a", "member_ordinal_b", "variant_a", "variant_b", "eligibility_a", "eligibility_b",
		"assessment_a", "assessment_b", "conflicting_a", "conflicting_b"}
	copyRow := func(generation int, null string) error {
		values := make([]string, len(columns))
		for i, column := range columns {
			values[i] = column
			if column == null {
				values[i] = "NULL"
			}
		}
		_, err := pool.Exec(ctx, fmt.Sprintf(`INSERT INTO eval_view_pairs(experiment_id,generation,ordinal,pair_id,suite_id,document,regression,unresolved,
complete_a,complete_b,execution_state_a,execution_state_b,%s)
SELECT experiment_id,%d,ordinal,pair_id,suite_id,document,regression,unresolved,complete_a,complete_b,NULL,NULL,%s
FROM eval_view_pairs WHERE experiment_id='second' AND generation=1 AND ordinal=0`, strings.Join(columns, ","), generation, strings.Join(values, ",")))
		return err
	}
	for i, column := range columns {
		var pgErr *pgconn.PgError
		if err = copyRow(10+i, column); Classify(err) != ClassInvalid || !errors.As(err, &pgErr) || pgErr.ColumnName != column {
			t.Fatalf("a pair row without %s was stored: %v", column, err)
		}
	}
	if err = copyRow(9, ""); err != nil {
		t.Fatalf("a pair row without execution states was rejected: %v", err)
	}
}
