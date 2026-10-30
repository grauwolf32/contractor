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

func TestReadBodyFindsWritesLocksReadsCallsAndDynamicSQL(t *testing.T) {
	for _, test := range []struct {
		name, source                string
		writes, locks, reads, calls []Name
		dynamic                     bool
	}{
		{name: "plain writes", source: "BEGIN INSERT INTO a VALUES (1); UPDATE b SET x = 1; DELETE FROM c; MERGE INTO d USING e ON true WHEN MATCHED THEN DELETE; TRUNCATE TABLE f, g; LOCK h IN SHARE MODE; END",
			writes: []Name{{Object: "a"}, {Object: "b"}, {Object: "c"}, {Object: "d"}, {Object: "f"}, {Object: "g"}, {Object: "h"}},
			reads:  []Name{{Object: "e"}}, calls: []Name{{Object: "values"}}},
		{name: "quoted and folded names", source: `INSERT INTO "Log" VALUES (1); update ONLY Runs SET x = 1; DELETE FROM ONLY "a""b"`,
			writes: []Name{{Object: "Log"}, {Object: "runs"}, {Object: `a"b`}}, calls: []Name{{Object: "values"}}},
		{name: "qualified names", source: `DELETE FROM public.runs; INSERT INTO "My Schema"."Log" (note) SELECT s.f(1); UPDATE ONLY x.y SET a = 1`,
			writes: []Name{{Schema: "public", Object: "runs"}, {Schema: "My Schema", Object: "Log"}, {Schema: "x", Object: "y"}},
			calls:  []Name{{Schema: "My Schema", Object: "Log"}, {Schema: "s", Object: "f"}}},
		{name: "comment marker in a constant", source: "BEGIN RAISE NOTICE 'a -- b'; DELETE FROM log; END",
			writes: []Name{{Object: "log"}}},
		{name: "constants and comments are not SQL", source: "BEGIN IF TG_OP = 'DELETE' THEN RAISE NOTICE $n$DELETE FROM a -- it's$n$; END IF; " +
			"RAISE NOTICE E'UPDATE b SET \\' x'; -- INSERT INTO c\n /* DELETE FROM d /* nested */ */ PERFORM f(OLD.id); END",
			calls: []Name{{Object: "f"}}},
		{name: "updates that name no table", source: "INSERT INTO a VALUES (1) ON CONFLICT (k) DO UPDATE SET v = 1; SELECT 1 FROM b FOR UPDATE OF b SKIP LOCKED; SELECT 1 FOR UPDATE NOWAIT; SELECT 1 FOR NO KEY UPDATE",
			writes: []Name{{Object: "a"}}, locks: []Name{{Object: "b"}}, reads: []Name{{Object: "b"}},
			calls: []Name{{Object: "values"}, {Object: "conflict"}}},
		{name: "reads and joins", source: "SELECT x INTO y FROM a AS p, public.b JOIN c q ON q.id = p.id LEFT JOIN LATERAL jsonb_each(p.doc) AS e ON true " +
			"WHERE EXISTS (SELECT 1 FROM d) AND EXTRACT(EPOCH FROM NEW.at) > 0; DELETE FROM f USING g WHERE f.id = g.id; " +
			"FOR row IN SELECT * FROM h LOOP END LOOP; RAISE EXCEPTION 'no' USING ERRCODE = 'P0001'",
			writes: []Name{{Object: "f"}},
			reads: []Name{{Object: "a"}, {Schema: "public", Object: "b"}, {Object: "c"}, {Object: "d"}, {Schema: "new", Object: "at"},
				{Object: "g"}, {Object: "h"}, {Object: "errcode"}},
			calls: []Name{{Object: "jsonb_each"}, {Object: "exists"}, {Object: "extract"}}},
		{name: "row locks", source: "PERFORM 1 FROM a AS p JOIN b ON b.id = p.id WHERE p.id = 1 FOR NO KEY UPDATE OF p NOWAIT; " +
			"SELECT 1 FROM c, d WHERE EXISTS (SELECT 1 FROM e FOR UPDATE) FOR UPDATE; SELECT 1 FROM f FOR SHARE; SELECT 1 FROM g FOR KEY SHARE",
			locks: []Name{{Object: "a"}, {Object: "e"}, {Object: "c"}, {Object: "d"}},
			reads: []Name{{Object: "a"}, {Object: "b"}, {Object: "c"}, {Object: "d"}, {Object: "e"}, {Object: "f"}, {Object: "g"}},
			calls: []Name{{Object: "exists"}}},
		{name: "dynamic SQL", source: "BEGIN EXECUTE format('DELETE FROM %I', t); END",
			calls: []Name{{Object: "format"}}, dynamic: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			body, err := ReadBody(test.source)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(body.Writes, test.writes) || !slices.Equal(body.Locks, test.locks) ||
				!slices.Equal(body.Reads, test.reads) || !slices.Equal(body.Calls, test.calls) || body.Dynamic != test.dynamic {
				t.Fatalf("ReadBody(%q) = %+v, want writes %v, locks %v, reads %v, calls %v, dynamic %t",
					test.source, body, test.writes, test.locks, test.reads, test.calls, test.dynamic)
			}
		})
	}
	for _, source := range []string{"DELETE FROM 'a", "BEGIN $x$ DELETE FROM a; END", `UPDATE "a SET x = 1`} {
		if body, err := ReadBody(source); err == nil {
			t.Errorf("ReadBody(%q) = %+v, want an error", source, body)
		}
	}
}
