package explaintest

import (
	"slices"
	"strings"
	"testing"
)

// guardFixture is a column-scoped guard of x that a change of x rejects
// unless the stored state is open.
const guardFixture = `
BEGIN
    IF NEW.x IS DISTINCT FROM OLD.x AND OLD.state <> 'open'
    THEN
        RAISE EXCEPTION 'x is immutable';
    END IF;
    RETURN NEW;
END;
`

const guardDefinition = `CREATE TRIGGER guard BEFORE UPDATE OF x ON public.t FOR EACH ROW EXECUTE FUNCTION f()`

// withCondition adds condition to the IF of guardFixture.
func withCondition(condition string) string {
	return strings.Replace(guardFixture, "\n    THEN", "\n        "+condition+"\n    THEN", 1)
}

// withStatement adds statement before the RETURN of guardFixture.
func withStatement(statement string) string {
	return strings.Replace(guardFixture, "    RETURN NEW;", "    "+statement+"\n    RETURN NEW;", 1)
}

func TestCheckBeforeUpdateTriggerReadsEveryReferenceToNewAndOld(t *testing.T) {
	columns := set([]string{"x", "y", "state"})
	for _, test := range []struct {
		name, language, source, definition string
		updateOf, qualifiers               []string
		// want is a part of a problem expected, or empty for none.
		want string
	}{
		{name: "guard with a qualifier"},
		{name: "reversed comparison of an unlisted column", source: withCondition("OR OLD.y IS DISTINCT FROM NEW.y"),
			want: "reads [NEW.y OLD.y], which it neither lists nor declares"},
		{name: "<> comparison", source: withCondition("OR NEW.y <> OLD.y"),
			want: "reads [NEW.y OLD.y], which it neither lists nor declares"},
		{name: "!= comparison with casts", source: withCondition("OR NEW.y::text != OLD.y::text"),
			want: "reads [NEW.y OLD.y], which it neither lists nor declares"},
		{name: "comparison inside a function call", source: withCondition("OR coalesce(NEW.y, 0) <> coalesce(OLD.y, 0)"),
			want: "reads [NEW.y OLD.y], which it neither lists nor declares"},
		{name: "read of an unlisted NEW column", source: withCondition("AND NEW.y IS NULL"),
			want: "reads [NEW.y], which it neither lists nor declares"},
		{name: "quoted identifier", source: withCondition(`OR NEW."y" IS DISTINCT FROM OLD."y"`),
			want: "reads [NEW.y OLD.y], which it neither lists nor declares"},
		{name: "quoted identifier of another case", source: withCondition(`OR NEW."X" IS DISTINCT FROM OLD."X"`),
			want: "reads [NEW.X OLD.X], which are no columns"},
		{name: "upper case and spaces around the dot", source: withCondition("OR new . Y IS DISTINCT FROM Old . y"),
			want: "reads [NEW.y OLD.y], which it neither lists nor declares"},
		{name: "comments and string constants", source: withStatement(
			"-- NEW.y IS DISTINCT FROM OLD.y\n    /* NEW.y /* nested */ := 1; */\n" +
				"    RAISE NOTICE 'NEW.y = %', E'NEW.y \\' := 1', $q$ NEW.y := 1 $q$;")},
		{name: "assignment with =", source: withStatement("NEW.y = upper(OLD.state);"),
			want: "assigns NEW.y"},
		{name: "assignment with :=", source: withStatement("NEW.y := 1;"),
			want: "assigns NEW.y"},
		{name: "assignment of an array element", source: withStatement("NEW.y[1] := 0;"),
			want: "assigns NEW.y"},
		{name: "assignment through INTO", source: withStatement("SELECT 1, 2 INTO STRICT NEW.x, NEW.y;"),
			want: "assigns NEW.y"},
		{name: "assignment of the whole row", source: withStatement("NEW := OLD;"),
			want: "assigns NEW"},
		{name: "comparison of the whole row", source: withCondition("OR NEW IS DISTINCT FROM OLD"),
			want: "uses the whole NEW row"},
		{name: "every column of a row", source: withStatement("PERFORM to_jsonb(OLD.*);"),
			want: "uses OLD. without a column name"},
		{name: "unterminated string constant", source: withStatement("RAISE EXCEPTION 'x;"),
			want: "cannot read: unterminated string constant"},
		{name: "unterminated dollar quote", source: withStatement("PERFORM $q$ x;"),
			want: "cannot read: unterminated dollar-quoted string"},
		{name: "function in another language", language: "sql", want: "runs a sql function"},
		{name: "listed column it never reads", updateOf: []string{"x", "y"},
			want: "fires on UPDATE OF [x y] but never reads NEW of [y]"},
		{name: "listed column read only through OLD", source: withCondition("AND OLD.y IS NOT NULL"),
			updateOf: []string{"x", "y"}, want: "never reads NEW of [y]"},
		{name: "compared qualifier", source: withCondition("OR NEW.state IS DISTINCT FROM OLD.state"),
			want: "compares its qualifier state between NEW and OLD"},
		{name: "qualifier it never reads", qualifiers: []string{"state", "y"},
			want: "declares the qualifier y, which it never reads"},
		{name: "listed column declared as a qualifier", qualifiers: []string{"state", "x"},
			want: "declares its listed column x as a qualifier"},
		{name: "comparison of two columns", source: withCondition("OR NEW.x IS NOT DISTINCT FROM OLD.state"),
			want: "compares NEW.x with OLD.state"},
		{name: "WHEN condition", source: "BEGIN RAISE EXCEPTION 'x is immutable'; END;", qualifiers: []string{},
			definition: `CREATE TRIGGER guard BEFORE UPDATE OF x ON public.t FOR EACH ROW ` +
				`WHEN ((new.x IS DISTINCT FROM old.x)) EXECUTE FUNCTION f()`},
		{name: "WHEN condition reading an unlisted column", source: "BEGIN RAISE EXCEPTION 'x is immutable'; END;",
			qualifiers: []string{}, definition: `CREATE TRIGGER guard BEFORE UPDATE OF x ON public.t FOR EACH ROW ` +
				`WHEN (((new.x IS DISTINCT FROM old.x) OR (new.y IS DISTINCT FROM old.y))) EXECUTE FUNCTION f()`,
			want: "reads [NEW.y OLD.y], which it neither lists nor declares"},
		{name: "UPDATE branch of an INSERT OR UPDATE guard", qualifiers: []string{}, source: `
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF NEW.x IS DISTINCT FROM OLD.x THEN
            RAISE EXCEPTION 'x is immutable';
        END IF;
        RETURN NEW;
    END IF;
    IF NEW.y IS NULL THEN
        RAISE EXCEPTION 'y is required';
    END IF;
    RETURN NEW;
END;`},
		{name: "UPDATE branch that falls through", qualifiers: []string{}, source: `
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF NEW.x IS DISTINCT FROM OLD.x THEN
            RAISE EXCEPTION 'x is immutable';
        END IF;
    END IF;
    IF NEW.y IS NULL THEN
        RAISE EXCEPTION 'y is required';
    END IF;
    RETURN NEW;
END;`, want: "reads [NEW.y], which it neither lists nor declares"},
		{name: "INSERT branch first", qualifiers: []string{}, source: `
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.y IS NULL THEN
            RAISE EXCEPTION 'y is required';
        END IF;
        RETURN NEW;
    END IF;
    IF NEW.x IS DISTINCT FROM OLD.x THEN
        RAISE EXCEPTION 'x is immutable';
    END IF;
    RETURN NEW;
END;`},
		{name: "UPDATE branch after declarations", qualifiers: []string{}, source: `
DECLARE found text; begun boolean := true;
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF NEW.x IS DISTINCT FROM OLD.x THEN
            RAISE EXCEPTION 'x is immutable';
        END IF;
        RETURN NEW;
    END IF;
    SELECT state INTO found FROM t WHERE y = NEW.y;
    RETURN NEW;
END;`},
		{name: "declaration that reads an unlisted column", qualifiers: []string{}, source: `
DECLARE missing boolean := NEW.y IS NULL;
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF NEW.x IS DISTINCT FROM OLD.x THEN
            RAISE EXCEPTION 'x is immutable';
        END IF;
        RETURN NEW;
    END IF;
    RETURN NEW;
END;`, want: "reads [NEW.y], which it neither lists nor declares"},
		{name: "declarations without a body", qualifiers: []string{},
			source: "DECLARE missing boolean := NEW.y IS NULL;", want: "reads [NEW.y], which it neither lists nor declares"},
		{name: "unscoped trigger that assigns", updateOf: []string{}, qualifiers: []string{},
			source: "BEGIN NEW.y := now(); RETURN NEW; END;", want: "assigns NEW.y"},
		{name: "unscoped trigger reading whole rows", updateOf: []string{}, qualifiers: []string{},
			source: "BEGIN IF NEW IS DISTINCT FROM OLD THEN RAISE EXCEPTION 'immutable'; END IF; RETURN NEW; END;"},
		{name: "unscoped trigger with qualifiers", updateOf: []string{},
			source: "BEGIN RETURN NEW; END;", want: "declares qualifiers but fires on every UPDATE"},
		{name: "insert into a log table", source: withStatement("INSERT INTO log (value) SELECT NEW.x;")},
	} {
		t.Run(test.name, func(t *testing.T) {
			trigger := guardTrigger{name: "guard", language: "plpgsql", source: guardFixture,
				definition: guardDefinition, updateOf: []string{"x"}}
			qualifiers := []string{"state"}
			if test.language != "" {
				trigger.language = test.language
			}
			if test.source != "" {
				trigger.source = test.source
			}
			if test.definition != "" {
				trigger.definition = test.definition
			}
			if test.updateOf != nil {
				trigger.updateOf = test.updateOf
			}
			if test.qualifiers != nil {
				qualifiers = test.qualifiers
			}
			problems := checkBeforeUpdateTrigger(trigger, columns, qualifiers)
			switch {
			case test.want == "" && len(problems) != 0:
				t.Errorf("problems = %q, want none", problems)
			case test.want != "" && !slices.ContainsFunc(problems, func(problem string) bool {
				return strings.Contains(problem, test.want)
			}):
				t.Errorf("problems = %q, want one containing %q", problems, test.want)
			}
		})
	}
}

func TestTokenizeSplitsAsPostgreSQLDoes(t *testing.T) {
	tokens, err := tokenize(`NEW."a""b" := E'x\'y' || $$z$$ || $1 ||1..2; x=-1 /* a /* b */ c */ -- d`)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, current := range tokens {
		got = append(got, current.text)
	}
	want := []string{"new", ".", `a"b`, ":=", "x'y", "||", "z", "||", "$1", "||", "1", "..", "2", ";", "x", "=", "-", "1"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Errorf("tokens = %q, want %q", got, want)
	}
	for _, source := range []string{`'open`, `"open`, `""`, `$tag$ open`, `/* open`, `$ x`, "\\"} {
		if _, err := tokenize(source); err == nil {
			t.Errorf("tokenize(%q) succeeded", source)
		}
	}
}
