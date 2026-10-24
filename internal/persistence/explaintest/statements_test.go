package explaintest

import (
	"slices"
	"strings"
	"testing"
)

func TestSplitStatementsEndsStatementsWherePsqlDoes(t *testing.T) {
	for _, test := range []struct {
		name, source string
		// want lists each statement's words joined by spaces.
		want []string
		// values lists the values of the last statement.
		values []string
	}{
		{name: "plain", source: "SELECT 1; COMMIT;\nvacuum t", want: []string{"select", "commit", "vacuum t"}},
		{name: "empty statements", source: ";; SELECT 1;;", want: []string{"select"}},
		{name: "line comment", source: "SELECT 1; -- COMMIT;\nSELECT 2", want: []string{"select", "select"}},
		{name: "nested block comment", source: "SELECT 1 /* a /* COMMIT; */ b; */; END", want: []string{"select", "end"}},
		{name: "plain string", source: "SELECT 'a;''b'; COMMIT", want: []string{"select", "commit"}},
		{name: "escape string", source: `SELECT E'it\'s; COMMIT', e'\\'; COMMIT`, want: []string{"select", "commit"}},
		{name: "quoted identifier", source: `SELECT "a;"";b" FROM t`, want: []string{"select from t"}, values: []string{`a;";b`}},
		{name: "dollar body", source: "CREATE FUNCTION f() RETURNS void LANGUAGE plpgsql AS $body$ BEGIN COMMIT; END; $body$; SELECT 1",
			want: []string{"create function f returns void language plpgsql as", "select"}},
		{name: "dollar value", source: "SELECT set_config($$statement_timeout$$, '0', true)",
			want: []string{"select set_config true"}, values: []string{"statement_timeout", "0"}},
		{name: "parentheses", source: "CREATE RULE r AS ON INSERT TO t DO ALSO (NOTIFY a; NOTIFY b); SELECT 1",
			want: []string{"create rule r as on insert to t do also notify a notify b", "select"}},
		{name: "begin atomic", source: "CREATE FUNCTION f(a int) RETURNS int LANGUAGE sql BEGIN ATOMIC SELECT a; SELECT CASE WHEN a > 0 THEN 1 END; END; COMMIT",
			want: []string{"create function f a int returns int language sql begin atomic select a select case when a then end end", "commit"}},
		{name: "begin atomic or replace procedure", source: "create or replace procedure p() begin atomic insert into t values (1); end; savepoint s",
			want: []string{"create or replace procedure p begin atomic insert into t values end", "savepoint s"}},
		{name: "case outside a routine", source: "UPDATE t SET a = CASE WHEN b THEN 1 ELSE 2\nEND; BEGIN", want: []string{"update t set a case when b then else end", "begin"}},
		{name: "transaction block", source: "BEGIN; SELECT 1; END;", want: []string{"begin", "select", "end"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			statements, err := SplitStatements(test.source)
			if err != nil {
				t.Fatal(err)
			}
			var got []string
			for _, statement := range statements {
				got = append(got, strings.Join(statement.Words, " "))
			}
			if !slices.Equal(got, test.want) {
				t.Fatalf("SplitStatements(%q) = %q, want %q", test.source, got, test.want)
			}
			if test.values != nil && !slices.Equal(statements[len(statements)-1].Values, test.values) {
				t.Fatalf("values of the last statement = %q, want %q", statements[len(statements)-1].Values, test.values)
			}
		})
	}
	for _, source := range []string{"SELECT 'a", `SELECT E'a\'`, "SELECT /* a /* b */", "SELECT $x$ a", `SELECT "a`, "SELECT {"} {
		if statements, err := SplitStatements(source); err == nil {
			t.Errorf("SplitStatements(%q) = %v, want an error", source, statements)
		}
	}
}
