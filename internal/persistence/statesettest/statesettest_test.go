package statesettest

import (
	"context"
	"maps"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/jackc/pgx/v5/pgxpool"
)

// fixtureDDL creates parity_fixture: a state column whose shape CHECK sets
// closed_at in exactly the closed state, and a hold_state column that reuses
// the word "open" but is not declared, so it must never be compared.
const fixtureDDL = `
CREATE TABLE parity_fixture (
    id text PRIMARY KEY,
    state text NOT NULL CHECK (state IN ('open', 'held', 'closed')),
    hold_state text NOT NULL CHECK (hold_state IN ('open', 'archived')),
    closed_at timestamptz,
    CONSTRAINT parity_fixture_shape CHECK (
        (state IN ('open', 'held') AND closed_at IS NULL)
        OR (state = 'closed' AND closed_at IS NOT NULL)
    ),
    CONSTRAINT parity_fixture_hold_shape CHECK (state <> 'closed' OR hold_state = 'archived')
);
CREATE INDEX parity_fixture_open_idx ON parity_fixture (id) WHERE state IN ('open', 'held');
CREATE INDEX parity_fixture_unclosed_idx ON parity_fixture (id) WHERE state <> 'closed';
CREATE INDEX parity_fixture_closed_idx ON parity_fixture (closed_at) WHERE closed_at IS NOT NULL;
CREATE INDEX parity_fixture_archived_idx ON parity_fixture (id) WHERE hold_state = 'archived'`

// fixtureCatalog is the catalog PostgreSQL reports for fixtureDDL;
// TestPostgresReadReportsTheCatalogForm holds it to the real server.
func fixtureCatalog() Catalog {
	return Catalog{
		Checks: map[string]string{
			"parity_fixture_hold_shape":       `CHECK (((state <> 'closed'::text) OR (hold_state = 'archived'::text)))`,
			"parity_fixture_hold_state_check": `CHECK ((hold_state = ANY (ARRAY['open'::text, 'archived'::text])))`,
			"parity_fixture_shape": `CHECK ((((state = ANY (ARRAY['open'::text, 'held'::text])) AND (closed_at IS NULL)) OR ` +
				`((state = 'closed'::text) AND (closed_at IS NOT NULL))))`,
			"parity_fixture_state_check": `CHECK ((state = ANY (ARRAY['open'::text, 'held'::text, 'closed'::text])))`,
		},
		Predicates: map[string]string{
			"parity_fixture_archived_idx": `(hold_state = 'archived'::text)`,
			"parity_fixture_closed_idx":   `(closed_at IS NOT NULL)`,
			"parity_fixture_open_idx":     `(state = ANY (ARRAY['open'::text, 'held'::text]))`,
			"parity_fixture_unclosed_idx": `(state <> 'closed'::text)`,
		},
	}
}

// fixtureTable declares parity_fixture with the owning set values.
func fixtureTable(values []string) Table {
	return Table{
		Name: "parity_fixture",
		Columns: []Column{{
			Name: "state", Check: "parity_fixture_state_check", Values: values,
			Subsets: map[string][]string{
				"unclosed": {"open", "held"},
				"closed":   {"closed"},
			},
			Partitions: [][]string{{"unclosed", "closed"}},
			Markers:    []Marker{{Column: "closed_at", Check: "parity_fixture_shape", Subset: "closed"}},
		}},
		Indexes: map[string]map[string]string{
			"parity_fixture_archived_idx": nil,
			"parity_fixture_closed_idx":   {"state": "closed"},
			"parity_fixture_open_idx":     {"state": "unclosed"},
			"parity_fixture_unclosed_idx": {"state": "unclosed"},
		},
	}
}

// driftCase changes the declaration or the catalog of the fixture and names
// the problems Compare must report.
type driftCase struct {
	name   string
	change func(t *testing.T, table *Table, catalog *Catalog)
	want   []string
}

func driftCases() []driftCase {
	return []driftCase{
		{
			name: "a constant added to the owning type escapes no handwritten list",
			change: func(t *testing.T, table *Table, _ *Catalog) {
				table.Columns[0].Values = mustDeclaredValues(t, "added")
			},
			want: []string{
				`parity_fixture.state: "paused" lies in 0 of the subsets [unclosed closed], which must split the owning Go set exactly; classify it`,
				`parity_fixture.state: the owning Go set declares [paused], which CHECK parity_fixture_state_check does not list`,
				`parity_fixture.state: marker closed_at: CHECK parity_fixture_shape admits no row in [paused]`,
				// The inequality form admits the new state at once.
				`parity_fixture.state: partial index parity_fixture_unclosed_idx admits [held open paused], want subset "unclosed" [held open]: extra [paused]`,
			},
		},
		{
			name: "a value only the schema lists",
			change: func(_ *testing.T, _ *Table, catalog *Catalog) {
				catalog.Checks["parity_fixture_state_check"] =
					`CHECK ((state = ANY (ARRAY['open'::text, 'held'::text, 'closed'::text, 'archived'::text])))`
			},
			want: []string{
				`parity_fixture.state: CHECK parity_fixture_state_check lists [archived], which the owning Go set does not declare`,
			},
		},
		{
			name: "a value only a shape CHECK compares",
			change: func(_ *testing.T, _ *Table, catalog *Catalog) {
				catalog.Checks["parity_fixture_hold_shape"] = `CHECK (((state <> 'archived'::text) OR (hold_state = 'archived'::text)))`
			},
			want: []string{
				`parity_fixture.state: CHECK parity_fixture_hold_shape compares it with "archived", which the owning Go set does not declare`,
			},
		},
		{
			name: "a predicate that misses a member of its subset",
			change: func(_ *testing.T, _ *Table, catalog *Catalog) {
				catalog.Predicates["parity_fixture_open_idx"] = `(state = 'open'::text)`
			},
			want: []string{
				`parity_fixture.state: partial index parity_fixture_open_idx admits [open], want subset "unclosed" [held open]: missing [held] (WHERE (state = 'open'::text))`,
			},
		},
		{
			name: "a predicate that admits a member outside its subset",
			change: func(_ *testing.T, _ *Table, catalog *Catalog) {
				catalog.Predicates["parity_fixture_open_idx"] = `(state = ANY (ARRAY['open'::text, 'held'::text, 'closed'::text]))`
			},
			want: []string{
				`parity_fixture.state: partial index parity_fixture_open_idx admits [closed held open], want subset "unclosed" [held open]: extra [closed]`,
			},
		},
		{
			name: "a predicate that lists a value only the schema knows",
			change: func(_ *testing.T, _ *Table, catalog *Catalog) {
				catalog.Predicates["parity_fixture_open_idx"] = `(state = ANY (ARRAY['open'::text, 'held'::text, 'archived'::text]))`
			},
			want: []string{
				`parity_fixture.state: partial index parity_fixture_open_idx compares it with "archived", which the owning Go set does not declare`,
			},
		},
		{
			name: "an undeclared state-dependent predicate",
			change: func(_ *testing.T, _ *Table, catalog *Catalog) {
				catalog.Predicates["parity_fixture_held_idx"] = `(state = 'held'::text)`
			},
			want: []string{
				`parity_fixture: partial index parity_fixture_held_idx (WHERE (state = 'held'::text)) is not declared; declare the subset of each state column it admits, or nil when it restricts none`,
			},
		},
		{
			name: "a predicate that restricts a column it declares no subset for",
			change: func(_ *testing.T, table *Table, _ *Catalog) {
				table.Indexes["parity_fixture_open_idx"] = nil
			},
			want: []string{
				`parity_fixture.state: partial index parity_fixture_open_idx admits [held open] but declares no subset`,
			},
		},
		{
			name: "a declared subset whose predicate no longer restricts the column",
			change: func(_ *testing.T, _ *Table, catalog *Catalog) {
				catalog.Predicates["parity_fixture_open_idx"] = `(hold_state = 'open'::text)`
			},
			want: []string{
				`parity_fixture.state: partial index parity_fixture_open_idx declares subset "unclosed" but does not restrict the column`,
			},
		},
		{
			name: "a declared partial index the schema lacks",
			change: func(_ *testing.T, _ *Table, catalog *Catalog) {
				delete(catalog.Predicates, "parity_fixture_unclosed_idx")
			},
			want: []string{`parity_fixture: declared partial index parity_fixture_unclosed_idx is missing or has no predicate`},
		},
		{
			name: "a restriction in a form the check cannot read",
			change: func(_ *testing.T, _ *Table, catalog *Catalog) {
				catalog.Predicates["parity_fixture_open_idx"] = `(lower(state) = 'open'::text)`
			},
			want: []string{
				`parity_fixture.state: partial index parity_fixture_open_idx: the conjunct "lower(state) = 'open'::text" restricts the column in a form this check cannot read`,
			},
		},
		{
			name: "a marker set outside its subset",
			change: func(_ *testing.T, _ *Table, catalog *Catalog) {
				catalog.Checks["parity_fixture_shape"] = `CHECK ((((state = 'open'::text) AND (closed_at IS NULL)) OR ` +
					`((state = ANY (ARRAY['held'::text, 'closed'::text])) AND (closed_at IS NOT NULL))))`
			},
			want: []string{
				`parity_fixture.state: marker closed_at: CHECK parity_fixture_shape sets closed_at in [closed held], want subset "closed" [closed]: extra [held]`,
			},
		},
		{
			name: "a handwritten subset that lists an undeclared value",
			change: func(_ *testing.T, table *Table, _ *Catalog) {
				table.Columns[0].Subsets["closed"] = []string{"closed", "archived"}
			},
			want: []string{
				`parity_fixture.state: subset "closed" lists "archived", which the owning Go set does not declare`,
				`parity_fixture.state: marker closed_at: CHECK parity_fixture_shape sets closed_at in [closed], want subset "closed" [archived closed]: missing [archived]`,
				`parity_fixture.state: partial index parity_fixture_closed_idx admits [closed], want subset "closed" [archived closed]: missing [archived]`,
			},
		},
		{
			name: "a missing enumerating CHECK",
			change: func(_ *testing.T, _ *Table, catalog *Catalog) {
				delete(catalog.Checks, "parity_fixture_state_check")
			},
			want: []string{`parity_fixture.state: CHECK parity_fixture_state_check is missing`},
		},
	}
}

// TestCompareReportsDriftWithItsOwner proves the checker can fail: the
// fixture passes, and each drift case reports its problems naming the
// table.column or index concerned. The undeclared hold_state column reuses
// "open" and "archived" and is never compared.
func TestCompareReportsDriftWithItsOwner(t *testing.T) {
	t.Parallel()
	if problems := Compare(fixtureTable(mustDeclaredValues(t, "declared")), fixtureCatalog()); len(problems) > 0 {
		t.Fatalf("the fixture reports problems:\n%s", strings.Join(problems, "\n"))
	}
	for _, test := range driftCases() {
		t.Run(test.name, func(t *testing.T) {
			table, catalog := fixtureTable(mustDeclaredValues(t, "declared")), fixtureCatalog()
			test.change(t, &table, &catalog)
			requireProblems(t, Compare(table, catalog), test.want)
		})
	}
}

func requireProblems(t *testing.T, problems, want []string) {
	t.Helper()
	for _, expected := range want {
		if !slices.ContainsFunc(problems, func(problem string) bool { return strings.HasPrefix(problem, expected) }) {
			t.Errorf("no problem starts with\n  %s\nproblems:\n  %s", expected, strings.Join(problems, "\n  "))
		}
	}
	if len(problems) != len(want) {
		t.Errorf("got %d problems, want %d:\n  %s", len(problems), len(want), strings.Join(problems, "\n  "))
	}
}

// TestPostgresReadReportsTheCatalogForm creates the fixture on a real
// server: Read returns exactly fixtureCatalog, so the drift cases test the
// text PostgreSQL prints, and drift made with DDL is reported too.
func TestPostgresReadReportsTheCatalogForm(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedFixturePool(t, ctx)
	if _, err := pool.Exec(ctx, fixtureDDL); err != nil {
		t.Fatal(err)
	}
	catalog, err := Read(ctx, pool, "parity_fixture")
	if err != nil {
		t.Fatal(err)
	}
	want := fixtureCatalog()
	if !maps.Equal(catalog.Checks, want.Checks) || !maps.Equal(catalog.Predicates, want.Predicates) {
		t.Fatalf("Read = %#v\nwant %#v", catalog, want)
	}
	if _, err := Read(ctx, pool, "parity_fixture_missing"); err == nil {
		t.Error("Read of a missing table succeeded")
	}

	for _, statement := range []string{
		`CREATE INDEX parity_fixture_held_idx ON parity_fixture (id) WHERE state = 'held'`,
		`ALTER TABLE parity_fixture DROP CONSTRAINT parity_fixture_state_check,
		 ADD CONSTRAINT parity_fixture_state_check CHECK (state IN ('open', 'held', 'closed', 'archived'))`,
	} {
		if _, err := pool.Exec(ctx, statement); err != nil {
			t.Fatal(err)
		}
	}
	catalog, err = Read(ctx, pool, "parity_fixture")
	if err != nil {
		t.Fatal(err)
	}
	requireProblems(t, Compare(fixtureTable(mustDeclaredValues(t, "declared")), catalog), []string{
		`parity_fixture.state: CHECK parity_fixture_state_check lists [archived], which the owning Go set does not declare`,
		`parity_fixture: partial index parity_fixture_held_idx (WHERE (state = 'held'::text)) is not declared`,
	})
}

// TestDeclaredValuesReadsTheWholeDeclaration reads every constant of the
// type, in either form, and nothing else: not another type's constant with
// the same word, an untyped constant, a variable or a test file's constant.
// A declaration it cannot read whole fails instead of returning part of it:
// a non-literal or implicitly repeated value, an untyped expression of the
// type, an alias of the type, or a constant or alias inside a function.
func TestDeclaredValuesReadsTheWholeDeclaration(t *testing.T) {
	t.Parallel()
	if got := mustDeclaredValues(t, "declared"); !slices.Equal(got, []string{"closed", "held", "open"}) {
		t.Errorf("declared = %v", got)
	}
	if got := mustDeclaredValues(t, "added"); !slices.Equal(got, []string{"closed", "held", "open", "paused"}) {
		t.Errorf("added = %v", got)
	}
	for fixture, want := range map[string]string{
		"computed":   "constant Open of type State is not a string literal",
		"implicit":   "constant Held repeats a State value implicitly",
		"expression": "constant Open uses State in an expression the declaration check cannot read",
		"alias":      "parked is an alias of State; declare its constants with State itself",
		"local":      "function-local constant of type State; declare it at package level",
		"localtype":  "function-local type parked shadows or aliases State",
		"duplicate":  "two constants of type State share a value",
		"untyped":    "declares no type State",
		"missing":    "no such file or directory",
	} {
		if values, err := declaredValues(filepath.Join("testdata", fixture), "State"); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: (%v, %v), want an error containing %q", fixture, values, err, want)
		}
	}
	if values, err := declaredValues(filepath.Join("testdata", "declared"), "Missing"); err == nil {
		t.Errorf("a type the package does not declare reads %v", values)
	}
}

func mustDeclaredValues(t *testing.T, fixture string) []string {
	t.Helper()
	values, err := declaredValues(filepath.Join("testdata", fixture), "State")
	if err != nil {
		t.Fatal(err)
	}
	return values
}

// isolatedFixturePool connects to a new schema of the test database, dropped
// when the test ends. It needs no migrations: the tests create their tables.
func isolatedFixturePool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{})
	return pool
}
