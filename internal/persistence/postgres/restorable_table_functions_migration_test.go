package postgres

import (
	"context"
	"encoding/json"
	"maps"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// tableFunctionAttributesSQL describes every function of the current schema
// by everything a caller sees except its language and body: arguments,
// result, volatility, strictness, parallel safety, security, leakproofness
// and planner estimates.
const tableFunctionAttributesSQL = `
SELECT oid::regprocedure::text,
       concat_ws(' ', pg_get_function_arguments(oid), pg_get_function_result(oid), provolatile, proisstrict,
                 proparallel, prosecdef, proleakproof, procost, prorows)
FROM pg_proc
WHERE pronamespace = current_schema()::regnamespace`

func tableFunctionAttributes(t *testing.T, ctx context.Context, pool *pgxpool.Pool) map[string]string {
	t.Helper()
	rows, err := pool.Query(ctx, tableFunctionAttributesSQL)
	if err != nil {
		t.Fatal(err)
	}
	attributes := map[string]string{}
	var signature, described string
	if _, err := pgx.ForEachRow(rows, []any{&signature, &described}, func() error {
		attributes[signature] = described
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return attributes
}

// runtimeSnapshotCase is an argument pair of contractor_valid_run_runtime_snapshot.
type runtimeSnapshotCase struct {
	name      string
	candidate *string
	labels    *string
}

// runtimeSnapshotCases covers each rule of the Run snapshot validator on both
// sides of its boundary, and inputs whose JSON types the validator's
// functions refuse.
func runtimeSnapshotCases(t *testing.T) []runtimeSnapshotCase {
	t.Helper()
	pinned := func(label string, explicit bool) map[string]any {
		return map[string]any{
			"label": label, "explicit": explicit, "bindingRevision": json.Number("1"),
			"config": map[string]any{"name": "config", "version": "1.0", "digest": "sha256:" + strings.Repeat("a", 64)},
		}
	}
	valid := func() map[string]any {
		return map[string]any{
			"default":              pinned("default", false),
			"labels":               []any{pinned("gpu", true), pinned("tpu", true)},
			"llmCredentialIds":     []any{"llm-a", "llm-b"},
			"runtimeCredentialIds": []any{"runtime-a"},
		}
	}
	text := func(value string) *string { return &value }
	encode := func(value any) *string {
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		return text(string(encoded))
	}
	labels := text("{gpu,tpu}")
	cases := []runtimeSnapshotCase{
		{"valid", encode(valid()), labels},
		{"valid without labels", encode(map[string]any{
			"default": pinned("default", false), "labels": []any{}, "llmCredentialIds": []any{}, "runtimeCredentialIds": []any{},
		}), text("{}")},
		{"null candidate", nil, labels},
		{"null labels", encode(valid()), nil},
		{"array", text(`[]`), labels},
		{"string", text(`"default"`), labels},
		{"json null", text(`null`), labels},
		{"empty object", text(`{}`), labels},
		{"fewer labels than the Run", encode(valid()), text("{gpu,tpu,vpu}")},
		{"more labels than the Run", encode(valid()), text("{gpu}")},
		{"labels in another order", encode(valid()), text("{tpu,gpu}")},
		{"two-dimensional labels", encode(valid()), text("{{gpu,tpu}}")},
		{"labels from index zero", encode(valid()), text("[0:1]={gpu,tpu}")},
	}
	mutate := func(name string, change func(map[string]any)) {
		candidate := valid()
		change(candidate)
		cases = append(cases, runtimeSnapshotCase{name, encode(candidate), labels})
	}
	for _, key := range []string{"default", "labels", "llmCredentialIds", "runtimeCredentialIds"} {
		mutate("without "+key, func(c map[string]any) { delete(c, key) })
		mutate(key+" is a string", func(c map[string]any) { c[key] = "x" })
		mutate(key+" is an object", func(c map[string]any) { c[key] = map[string]any{"x": 1} })
		mutate(key+" is null", func(c map[string]any) { c[key] = nil })
	}
	mutate("extra member", func(c map[string]any) { c["extra"] = true })
	for name, change := range map[string]func(map[string]any){
		"explicit":                     func(p map[string]any) { p["explicit"] = !p["explicit"].(bool) },
		"label":                        func(p map[string]any) { p["label"] = "other" },
		"revision zero":                func(p map[string]any) { p["bindingRevision"] = json.Number("0") },
		"fractional revision":          func(p map[string]any) { p["bindingRevision"] = json.Number("1.5") },
		"revision 2^64":                func(p map[string]any) { p["bindingRevision"] = json.Number("18446744073709551616") },
		"revision 2^64-1":              func(p map[string]any) { p["bindingRevision"] = json.Number("18446744073709551615") },
		"revision text":                func(p map[string]any) { p["bindingRevision"] = "1" },
		"extra pinned member":          func(p map[string]any) { p["extra"] = 1 },
		"config string":                func(p map[string]any) { p["config"] = "config" },
		"config array":                 func(p map[string]any) { p["config"] = []any{} },
		"config extra member":          func(p map[string]any) { p["config"].(map[string]any)["extra"] = 1 },
		"config name":                  func(p map[string]any) { p["config"].(map[string]any)["name"] = "Config" },
		"config name of 64 bytes":      func(p map[string]any) { p["config"].(map[string]any)["name"] = "c" + strings.Repeat("a", 63) },
		"config version of 129 bytes":  func(p map[string]any) { p["config"].(map[string]any)["version"] = "1" + strings.Repeat("a", 128) },
		"config version of 128 bytes":  func(p map[string]any) { p["config"].(map[string]any)["version"] = "1" + strings.Repeat("a", 127) },
		"config digest":                func(p map[string]any) { p["config"].(map[string]any)["digest"] = "sha256:" + strings.Repeat("A", 64) },
		"config digest number":         func(p map[string]any) { p["config"].(map[string]any)["digest"] = 1 },
		"pinned config replaced by []": nil,
	} {
		for _, target := range []string{"default", "first label", "second label"} {
			mutate(target+": "+name, func(c map[string]any) {
				set := func(value any) { c["default"] = value }
				item, _ := c["default"].(map[string]any)
				if target != "default" {
					index := map[string]int{"first label": 0, "second label": 1}[target]
					set = func(value any) { c["labels"].([]any)[index] = value }
					item = c["labels"].([]any)[index].(map[string]any)
				}
				if change == nil {
					set([]any{})
					return
				}
				change(item)
			})
		}
	}
	for _, list := range []string{"llmCredentialIds", "runtimeCredentialIds"} {
		for name, ids := range map[string][]any{
			"empty":              {},
			"json null":          {nil},
			"number":             {1},
			"object":             {map[string]any{"id": "a"}},
			"empty string":       {""},
			"capital":            {"Llm"},
			"duplicate":          {"a", "a"},
			"descending":         {"b", "a"},
			"ascending":          {"a", "b", "c"},
			"128 bytes":          {"a" + strings.Repeat("b", 127)},
			"129 bytes":          {"a" + strings.Repeat("b", 128)},
			"null after valid":   {"a", nil},
			"descending at end":  {"a", "c", "b"},
			"digits and dashes":  {"a-1", "a_2"},
			"prefix sorts first": {"ab", "abc"},
		} {
			mutate(list+": "+name, func(c map[string]any) { c[list] = ids })
		}
	}
	return cases
}

func runtimeSnapshotVerdicts(t *testing.T, ctx context.Context, pool *pgxpool.Pool, cases []runtimeSnapshotCase) map[string]*bool {
	t.Helper()
	verdicts := make(map[string]*bool, len(cases))
	for _, item := range cases {
		var verdict *bool
		if err := pool.QueryRow(ctx, `SELECT contractor_valid_run_runtime_snapshot($1::jsonb, $2::text[])`,
			item.candidate, item.labels).Scan(&verdict); err != nil {
			t.Fatalf("%s: %v", item.name, err)
		}
		verdicts[item.name] = verdict
	}
	return verdicts
}

// TestPostgresRestorableTableFunctionsMigrationKeepsEveryVerdict upgrades a
// schema to the migration that gives table functions SQL-standard bodies
// (PER-R46). Every function keeps what callers see of it, and the Run
// snapshot validator, rewritten from PL/pgSQL, gives the verdict its
// predecessor gave for every case, including inputs whose JSON types made the
// predecessor's functions fail into its exception handler.
func TestPostgresRestorableTableFunctionsMigrationKeepsEveryVerdict(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, databaseURL)
	previous, _ := embeddedMigration(t, "restorable_table_functions")
	installMigrationPrefix(t, ctx, pool, previous)
	cases := runtimeSnapshotCases(t)
	attributes := tableFunctionAttributes(t, ctx, pool)
	verdicts := runtimeSnapshotVerdicts(t, ctx, pool, cases)
	valid := 0
	for _, verdict := range verdicts {
		if verdict != nil && *verdict {
			valid++
		}
	}
	if len(verdicts) != len(cases) || valid < 8 || valid == len(cases) {
		t.Fatalf("%d cases with %d distinct names, %d valid; the corpus is broken", len(cases), len(verdicts), valid)
	}
	if result := applyMigrationsThrough(t, ctx, pool, "restorable_table_functions"); len(result.AppliedVersions) != 1 || result.AppliedVersions[0] != previous+1 {
		t.Fatalf("migration=%+v", result)
	}
	if after := tableFunctionAttributes(t, ctx, pool); !maps.Equal(after, attributes) {
		for signature, described := range attributes {
			if after[signature] != described {
				t.Errorf("%s: %q, before %q", signature, after[signature], described)
			}
		}
		t.Fatalf("the migration changed what callers see of a function")
	}
	after := runtimeSnapshotVerdicts(t, ctx, pool, cases)
	show := func(verdict *bool) string {
		if verdict == nil {
			return "NULL"
		}
		if *verdict {
			return "true"
		}
		return "false"
	}
	for _, item := range cases {
		if show(after[item.name]) != show(verdicts[item.name]) {
			t.Errorf("%s: %s, before %s", item.name, show(after[item.name]), show(verdicts[item.name]))
		}
	}
	var language string
	if err := pool.QueryRow(ctx, `SELECT l.lanname FROM pg_proc p JOIN pg_language l ON l.oid = p.prolang
WHERE p.oid = 'contractor_valid_run_runtime_snapshot'::regproc`).Scan(&language); err != nil || language != "sql" {
		t.Fatalf("the Run snapshot validator is written in %q, %v", language, err)
	}
}
