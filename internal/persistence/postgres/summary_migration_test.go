package postgres

import (
	"encoding/json"
	"os"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestPostgresSummaryMigrationConstrainsVariantsAndOrdinalRoots(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	installMigrationPrefix(t, ctx, pool, 15)
	result, err := ApplyMigrations(ctx, pool, MigrationBudgets{})
	if err != nil || !slices.Contains(result.AppliedVersions, 16) {
		t.Fatalf("summary migration: %+v %v", result, err)
	}
	// LIKE copies the actual migrated CHECK definitions, without unrelated
	// FK/lifecycle triggers. The ordinal predicate comes from the real index.
	if _, err := pool.Exec(ctx, "CREATE TABLE summary_fixture (LIKE stage_executions INCLUDING CONSTRAINTS INCLUDING DEFAULTS)"); err != nil {
		t.Fatal(err)
	}
	var predicate string
	if err := pool.QueryRow(ctx, "SELECT pg_get_expr(indpred, indrelid) FROM pg_index WHERE indexrelid='stage_execution_escalation_ordinal_unique'::regclass").Scan(&predicate); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "CREATE UNIQUE INDEX summary_fixture_ordinal ON summary_fixture(run_id,stage_name,execution_config_variant,escalation_ordinal) WHERE "+predicate); err != nil {
		t.Fatal(err)
	}
	insert := func(id, variant string, ordinal *int, previous string, context string) error {
		_, err := pool.Exec(ctx, `INSERT INTO summary_fixture(stage_execution_id,run_id,stage_name,attempt,previous_execution_id,
stage_spec_schema_version,stage_spec_snapshot,stage_context_schema_version,stage_context_snapshot,
state,state_reason_code,execution_config_variant,escalation_ordinal)
VALUES($1,'run','copy',2,$2,'contractor/v1alpha1','{}','contractor/v1alpha1',$3::jsonb,'preparing','created',$4,$5)`,
			id, previous, context, variant, ordinal)
		return err
	}
	one := 1
	saved := `{"parameters":{},"artifacts":{},"continuation":{"sourceStageExecutionId":"source"}}`
	for _, tc := range []struct {
		name, variant, context string
		ordinal                *int
	}{
		{"base-with-continuation", "base", saved, nil},
		{"failed-with-continuation", "failed_escalation", saved, &one},
		{"interrupted-with-continuation", "interrupted_escalation", saved, &one},
		{"summary-without-context", "summary_escalation", `{}`, &one},
		{"summary-without-source", "summary_escalation", `{"continuation":{}}`, &one},
		{"summary-without-ordinal", "summary_escalation", saved, nil},
	} {
		if err := insert(tc.name, tc.variant, tc.ordinal, "source", tc.context); Classify(err) != ClassInvalid {
			t.Fatalf("%s invalid shape accepted: %v", tc.name, err)
		}
	}
	if err := insert("target", "summary_escalation", &one, "source", saved); err != nil {
		t.Fatal(err)
	}
	if err := insert("retry", "summary_escalation", &one, "target", saved); err != nil {
		t.Fatalf("retry consumed ordinal twice: %v", err)
	}
	if err := insert("duplicate", "summary_escalation", &one, "source", saved); Classify(err) != ClassDuplicate {
		t.Fatalf("second root accepted: %v", err)
	}
}

func TestSharedPostgresStageSummaryContracts(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	data := contractstest.ReadFile(t, "api", "testdata", "stage-summary-cases.json")
	var cases []struct {
		Name   string
		Valid  bool
		Result json.RawMessage
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			var valid bool
			if err := pool.QueryRow(ctx, "SELECT contractor_valid_stage_result($1::jsonb)", tc.Result).Scan(&valid); err != nil {
				t.Fatal(err)
			}
			if valid != tc.Valid {
				t.Fatalf("SQL validity=%v want %v", valid, tc.Valid)
			}
		})
	}
}
