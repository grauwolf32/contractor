package evalstore

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// Trigger event bits of pg_trigger.tgtype.
const (
	triggerInsert = 1 << 2
	triggerDelete = 1 << 3
	triggerUpdate = 1 << 4
)

var deleteActions = map[string]string{"a": "NO ACTION", "r": "RESTRICT", "c": "CASCADE", "n": "SET NULL", "d": "SET DEFAULT"}

// purgeForeignKeysSQL lists every foreign key whose referenced table is in
// the current schema: the referencing table, the referenced table and the
// referenced row's delete action.
const purgeForeignKeysSQL = `
SELECT child.relname::text, parent.relname::text, c.confdeltype::text
FROM pg_constraint c
JOIN pg_class child ON child.oid = c.conrelid
JOIN pg_class parent ON parent.oid = c.confrelid
WHERE c.contype = 'f' AND parent.relnamespace = current_schema()::regnamespace
ORDER BY 2, 1, c.conname`

// purgeTriggersSQL lists every user trigger of the current schema's tables
// with its events and the source of its function.
const purgeTriggersSQL = `
SELECT c.relname::text, t.tgname::text, t.tgtype::integer, p.proname::text
FROM pg_trigger t
JOIN pg_class c ON c.oid = t.tgrelid
JOIN pg_proc p ON p.oid = t.tgfoid
WHERE NOT t.tgisinternal AND c.relnamespace = current_schema()::regnamespace
ORDER BY 1, 2`

// purgeFunctionsSQL lists the source of every function of the current
// schema, overloads joined.
var purgeFunctionsSQL = `
SELECT proname::text, string_agg(` + explaintest.RoutineBodySQL("") + `, E'\n')
FROM pg_proc
WHERE pronamespace = current_schema()::regnamespace
GROUP BY proname`

// purgeRulesSQL lists every rewrite rule of the current schema's tables
// other than a view's _RETURN rule: each one changes what a write of its
// table does.
const purgeRulesSQL = `
SELECT c.relname::text, r.rulename::text
FROM pg_rewrite r
JOIN pg_class c ON c.oid = r.ev_class
WHERE c.relnamespace = current_schema()::regnamespace AND r.rulename <> '_RETURN' AND r.ev_type <> '1'
ORDER BY 1, 2`

// purgeColumnsSQL lists every column of the current schema's tables in
// column order.
const purgeColumnsSQL = `
SELECT table_name::text, array_agg(column_name::text ORDER BY ordinal_position)
FROM information_schema.columns
WHERE table_schema = current_schema()
GROUP BY table_name`

// inSchema returns the name of a table or function of the current schema
// as the catalog stores it, or, for a name another schema qualifies, the
// qualified name, which no purge table or schema function has.
func inSchema(name explaintest.Name, schema string) string {
	if name.Schema == "" || name.Schema == schema {
		return name.Object
	}
	return name.Schema + "." + name.Object
}

// writtenTables returns the tables the PL/pgSQL function fn writes, directly
// or through the functions of the schema it calls, sorted; "?" stands for
// dynamic SQL, which may write anything. explaintest.ReadBody reads the
// function's source as PostgreSQL's lexer does, so comments and string
// constants are ignored, and a function that only checks TG_OP = 'DELETE'
// writes nothing.
func writtenTables(functions map[string]string, schema, fn string, seen map[string]bool) ([]string, error) {
	if seen[fn] {
		return nil, nil
	}
	seen[fn] = true
	body, err := explaintest.ReadBody(functions[fn])
	if err != nil {
		return nil, fmt.Errorf("read function %s: %w", fn, err)
	}
	written := map[string]bool{}
	for _, table := range body.Writes {
		written[inSchema(table, schema)] = true
	}
	if body.Dynamic {
		written["?"] = true
	}
	for _, call := range body.Calls {
		if callee := inSchema(call, schema); callee != fn {
			if _, ok := functions[callee]; ok {
				tables, err := writtenTables(functions, schema, callee, seen)
				if err != nil {
					return nil, err
				}
				for _, table := range tables {
					written[table] = true
				}
			}
		}
	}
	return slices.Sorted(maps.Keys(written)), nil
}

// purgeCoverageViolations compares what the purge declares it deletes and
// charges with the schema, and returns every row the purge would change
// without counting it:
//
//   - every table referencing a table the purge deletes from must be emptied
//     by an earlier step, or be a cascading child of eval_experiments that the
//     final unit counts (finalPurgeCascades); a cascading, nulling or blocking
//     child of any other kind is a violation, and so are the children of a
//     table that an uncounted cascade empties;
//   - every trigger that a purge statement fires and that writes rows must be
//     a deletion trigger of its table declared in purgeDeleteTriggers, which
//     the purge charges, and every declared trigger must be one; triggers on
//     the rows a declared trigger writes are followed the same way;
//   - no table the purge deletes from, updates or has a trigger write has a
//     rewrite rule, which would write rows the purge does not count or
//     change what its own statements do;
//   - every table the purge deletes from or counts in its final unit must
//     have its columns measured in purgeRowColumns, every column in schema
//     order, and no other table may be listed there (Eval receipts are
//     measured by pg.DeleteEvalReceiptsWithinBudget and its own test).
//
// The steps, in the order the purge takes them, are experimentPurgePhases,
// the experiment row, projectPurgeTables and the Eval receipts.
func purgeCoverageViolations(ctx context.Context, db pg.DBTX) ([]string, error) {
	steps := append(slices.Clone(experimentPurgePhases), "eval_experiments")
	steps = append(append(steps, projectPurgeTables...), "mutation_receipts")
	order := map[string]int{}
	for i, table := range steps {
		order[table] = i
	}
	removedAt := maps.Clone(order)
	final := map[string]bool{}
	for _, table := range finalPurgeCascades {
		removedAt[table] = order["eval_experiments"]
		final[table] = true
	}

	type foreignKey struct{ child, parent, action string }
	children := map[string][]foreignKey{}
	rows, err := db.Query(ctx, purgeForeignKeysSQL)
	if err != nil {
		return nil, err
	}
	keys, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (foreignKey, error) {
		var key foreignKey
		return key, row.Scan(&key.child, &key.parent, &key.action)
	})
	if err != nil {
		return nil, err
	}
	for _, key := range keys {
		children[key.parent] = append(children[key.parent], key)
	}

	var violations []string
	cascaded := map[string]bool{}
	queue := slices.Sorted(maps.Keys(removedAt))
	for len(queue) > 0 {
		parent := queue[0]
		queue = queue[1:]
		for _, key := range children[parent] {
			if at, ok := order[key.child]; ok && at < removedAt[parent] {
				continue
			}
			if parent == "eval_experiments" && final[key.child] && key.action == "c" {
				cascaded[key.child] = true
				continue
			}
			violations = append(violations, fmt.Sprintf(
				"%s references %s (ON DELETE %s): the purge neither empties it first nor counts it in the final unit",
				key.child, parent, deleteActions[key.action]))
			if _, ok := removedAt[key.child]; !ok && key.action == "c" {
				removedAt[key.child] = removedAt[parent]
				queue = append(queue, key.child)
			}
		}
	}
	for _, table := range finalPurgeCascades {
		if !cascaded[table] {
			violations = append(violations, fmt.Sprintf("the final unit counts %s, which is no cascading child of eval_experiments", table))
		}
		if _, ok := order[table]; ok {
			violations = append(violations, fmt.Sprintf("%s is both a purge step and a cascade of the final unit", table))
		}
	}

	var schema string
	if err := db.QueryRow(ctx, `SELECT current_schema()`).Scan(&schema); err != nil {
		return nil, err
	}
	rows, err = db.Query(ctx, purgeFunctionsSQL)
	if err != nil {
		return nil, err
	}
	functions := map[string]string{}
	var name, source string
	if _, err = pgx.ForEachRow(rows, []any{&name, &source}, func() error {
		functions[name] = source
		return nil
	}); err != nil {
		return nil, err
	}
	type trigger struct {
		name, function string
		events         int
	}
	triggers := map[string][]trigger{}
	rows, err = db.Query(ctx, purgeTriggersSQL)
	if err != nil {
		return nil, err
	}
	var table string
	var tr trigger
	if _, err = pgx.ForEachRow(rows, []any{&table, &tr.name, &tr.events, &tr.function}, func() error {
		triggers[table] = append(triggers[table], tr)
		return nil
	}); err != nil {
		return nil, err
	}
	type event struct {
		table string
		event int
	}
	var fired []event
	for _, table := range slices.Sorted(maps.Keys(removedAt)) {
		fired = append(fired, event{table, triggerDelete})
	}
	// The only row a pass updates is the experiment's projection queue row
	// (invalidatePurgedViewSQL).
	invalidation, err := explaintest.ReadBody(invalidatePurgedViewSQL)
	if err != nil || len(invalidation.Writes) != 1 {
		return nil, fmt.Errorf("read invalidatePurgedViewSQL: writes %v, %v", invalidation.Writes, err)
	}
	fired = append(fired, event{inSchema(invalidation.Writes[0], schema), triggerUpdate})
	declared := map[string]bool{}
	visited := map[event]bool{}
	eventNames := map[int]string{triggerInsert: "inserts into", triggerDelete: "deletes from", triggerUpdate: "updates"}
	for len(fired) > 0 {
		next := fired[0]
		fired = fired[1:]
		if visited[next] {
			continue
		}
		visited[next] = true
		for _, tr := range triggers[next.table] {
			if tr.events&next.event == 0 {
				continue
			}
			written, err := writtenTables(functions, schema, tr.function, map[string]bool{})
			if err != nil {
				return nil, err
			}
			if len(written) == 0 {
				continue
			}
			if d, ok := purgeDeleteTriggers[tr.name]; ok && d.table == next.table && next.event == triggerDelete {
				declared[tr.name] = true
			} else {
				violations = append(violations, fmt.Sprintf(
					"trigger %s on %s writes %v when the purge %s it, and the purge does not count those rows",
					tr.name, next.table, written, eventNames[next.event]))
			}
			for _, table := range written {
				if table != "?" {
					// INSERT ... ON CONFLICT DO UPDATE fires either event.
					fired = append(fired, event{table, triggerInsert}, event{table, triggerUpdate})
				}
			}
		}
	}
	for _, name := range slices.Sorted(maps.Keys(purgeDeleteTriggers)) {
		if !declared[name] {
			violations = append(violations, fmt.Sprintf(
				"the purge counts trigger %s on %s, which is no writing deletion trigger of a purged table",
				name, purgeDeleteTriggers[name].table))
		}
	}
	changed := map[string]bool{}
	for next := range visited {
		changed[next.table] = true
	}
	rows, err = db.Query(ctx, purgeRulesSQL)
	if err != nil {
		return nil, err
	}
	var rule string
	if _, err = pgx.ForEachRow(rows, []any{&table, &rule}, func() error {
		if changed[table] {
			violations = append(violations, fmt.Sprintf(
				"rule %s on %s rewrites the purge's changes to it, and the purge does not count them", rule, table))
		}
		return nil
	}); err != nil {
		return nil, err
	}

	rows, err = db.Query(ctx, purgeColumnsSQL)
	if err != nil {
		return nil, err
	}
	columns := map[string][]string{}
	var names []string
	if _, err = pgx.ForEachRow(rows, []any{&table, &names}, func() error {
		columns[table] = slices.Clone(names)
		return nil
	}); err != nil {
		return nil, err
	}
	charged := map[string]bool{}
	for _, table := range append(steps, finalPurgeCascades...) {
		charged[table] = table != "mutation_receipts"
	}
	delete(charged, "mutation_receipts")
	for _, table := range slices.Sorted(maps.Keys(charged)) {
		if measured, ok := purgeRowColumns[table]; !ok {
			violations = append(violations, fmt.Sprintf("the purge does not measure the rows of %s", table))
		} else if !slices.Equal(measured, columns[table]) {
			violations = append(violations, fmt.Sprintf("%s has columns %v, the purge measures %v", table, columns[table], measured))
		}
	}
	for _, table := range slices.Sorted(maps.Keys(purgeRowColumns)) {
		if !charged[table] {
			violations = append(violations, fmt.Sprintf("the purge measures %s, which it never deletes from", table))
		}
	}
	slices.Sort(violations)
	return violations, nil
}

// The purge's own account of what it deletes and charges matches the schema:
// every cascading, nulling or blocking child of a table it deletes from,
// whatever its name, is emptied first or counted in the final unit; every
// trigger its statements fire that writes rows is charged; and every charged
// table is measured column by column. Each fixture adds to the schema, in a
// transaction that is rolled back, one change that the purge would not
// count, and the guard must name it.
func TestSharedPostgresEvalPurgeCoversCascadesAndTriggers(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	violations, err := purgeCoverageViolations(ctx, pool)
	if err != nil || len(violations) != 0 {
		t.Fatalf("purge coverage of the schema = %q, %v", violations, err)
	}
	logTable := `CREATE TABLE eval_purge_log (note text)`
	for _, fixture := range []struct {
		name string
		ddl  []string
		want []string
	}{
		{"cascading Eval child of members", []string{`CREATE TABLE eval_member_notes (
    experiment_id text NOT NULL, member_id text NOT NULL, note text,
    FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members (experiment_id, member_id) ON DELETE CASCADE)`},
			[]string{"eval_member_notes references eval_members (ON DELETE CASCADE)"}},
		{"cascading child outside the Eval tables", []string{`CREATE TABLE review_eval_links (
    experiment_id text NOT NULL, member_id text NOT NULL,
    FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members (experiment_id, member_id) ON DELETE CASCADE)`},
			[]string{"review_eval_links references eval_members (ON DELETE CASCADE)"}},
		{"child of an uncounted cascade", []string{
			`CREATE TABLE eval_member_notes (experiment_id text NOT NULL, member_id text NOT NULL, note_id text PRIMARY KEY,
    FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members (experiment_id, member_id) ON DELETE CASCADE)`,
			`CREATE TABLE eval_member_note_links (note_id text NOT NULL REFERENCES eval_member_notes ON DELETE CASCADE)`},
			[]string{"eval_member_note_links references eval_member_notes (ON DELETE CASCADE)",
				"eval_member_notes references eval_members (ON DELETE CASCADE)"}},
		{"cascade of the experiment row outside the final unit", []string{
			`CREATE TABLE eval_experiment_labels (experiment_id text NOT NULL REFERENCES eval_experiments ON DELETE CASCADE)`},
			[]string{"eval_experiment_labels references eval_experiments (ON DELETE CASCADE)"}},
		{"child that a delete sets to NULL", []string{
			`CREATE TABLE eval_command_notes (experiment_id text REFERENCES eval_experiments ON DELETE SET NULL)`},
			[]string{"eval_command_notes references eval_experiments (ON DELETE SET NULL)"}},
		{"child that blocks the delete", []string{`CREATE TABLE audit_eval_refs (
    experiment_id text NOT NULL, member_id text NOT NULL,
    FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members (experiment_id, member_id) ON DELETE RESTRICT)`},
			[]string{"audit_eval_refs references eval_members (ON DELETE RESTRICT)"}},
		{"writing deletion trigger", []string{logTable,
			`CREATE FUNCTION eval_purge_fixture() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN INSERT INTO eval_purge_log VALUES (OLD.member_id); RETURN OLD; END $$`,
			`CREATE TRIGGER eval_members_logged AFTER DELETE ON eval_members FOR EACH ROW EXECUTE FUNCTION eval_purge_fixture()`},
			[]string{"trigger eval_members_logged on eval_members writes [eval_purge_log] when the purge deletes from it"}},
		{"deletion trigger writing through a helper", []string{logTable,
			`CREATE FUNCTION eval_purge_fixture_helper(note text) RETURNS void LANGUAGE plpgsql AS $$
BEGIN UPDATE eval_purge_log SET note = note; END $$`,
			`CREATE FUNCTION eval_purge_fixture() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN PERFORM eval_purge_fixture_helper(OLD.command_id); RETURN OLD; END $$`,
			`CREATE TRIGGER eval_commands_logged BEFORE DELETE ON eval_commands FOR EACH ROW EXECUTE FUNCTION eval_purge_fixture()`},
			[]string{"trigger eval_commands_logged on eval_commands writes [eval_purge_log] when the purge deletes from it"}},
		{"deletion trigger writing through a SQL-standard helper", []string{logTable,
			`CREATE FUNCTION eval_purge_fixture_helper(note text) RETURNS void LANGUAGE sql
BEGIN ATOMIC UPDATE eval_purge_log SET note = $1; END`,
			`CREATE FUNCTION eval_purge_fixture() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN PERFORM eval_purge_fixture_helper(OLD.command_id); RETURN OLD; END $$`,
			`CREATE TRIGGER eval_commands_logged BEFORE DELETE ON eval_commands FOR EACH ROW EXECUTE FUNCTION eval_purge_fixture()`},
			[]string{"trigger eval_commands_logged on eval_commands writes [eval_purge_log] when the purge deletes from it"}},
		{"deletion trigger with dynamic SQL", []string{
			`CREATE FUNCTION eval_purge_fixture() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN EXECUTE format('SELECT %L', OLD.experiment_id); RETURN NULL; END $$`,
			`CREATE TRIGGER eval_frozen_plans_logged AFTER DELETE ON eval_frozen_plans FOR EACH STATEMENT EXECUTE FUNCTION eval_purge_fixture()`},
			[]string{"trigger eval_frozen_plans_logged on eval_frozen_plans writes [?] when the purge deletes from it"}},
		{"trigger on the projection queue update", []string{logTable,
			`CREATE FUNCTION eval_purge_fixture() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN INSERT INTO eval_purge_log VALUES (NEW.experiment_id); RETURN NEW; END $$`,
			`CREATE TRIGGER eval_queue_logged AFTER UPDATE ON eval_projection_queue FOR EACH ROW EXECUTE FUNCTION eval_purge_fixture()`},
			[]string{"trigger eval_queue_logged on eval_projection_queue writes [eval_purge_log] when the purge updates it"}},
		{"trigger on the collection rows a counted trigger writes", []string{logTable,
			`CREATE FUNCTION eval_purge_fixture() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN INSERT INTO eval_purge_log VALUES (NEW.project_id); RETURN NEW; END $$`,
			`CREATE TRIGGER eval_collections_logged AFTER INSERT OR UPDATE ON eval_collections FOR EACH ROW EXECUTE FUNCTION eval_purge_fixture()`},
			[]string{"trigger eval_collections_logged on eval_collections writes [eval_purge_log] when the purge inserts into it",
				"trigger eval_collections_logged on eval_collections writes [eval_purge_log] when the purge updates it"}},
		{"deletion trigger writing a quoted table", []string{logTable,
			`CREATE FUNCTION eval_purge_fixture() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN INSERT INTO "eval_purge_log" VALUES (OLD.member_id); RETURN OLD; END $$`,
			`CREATE TRIGGER eval_members_logged AFTER DELETE ON eval_members FOR EACH ROW EXECUTE FUNCTION eval_purge_fixture()`},
			[]string{"trigger eval_members_logged on eval_members writes [eval_purge_log] when the purge deletes from it"}},
		{"deletion trigger writing a schema-qualified table", []string{logTable,
			`CREATE FUNCTION eval_purge_fixture() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN DELETE FROM {schema}.eval_purge_log WHERE note = OLD.member_id; RETURN OLD; END $$`,
			`CREATE TRIGGER eval_members_logged AFTER DELETE ON eval_members FOR EACH ROW EXECUTE FUNCTION eval_purge_fixture()`},
			[]string{"trigger eval_members_logged on eval_members writes [eval_purge_log] when the purge deletes from it"}},
		{"deletion trigger writing after a comment marker in a string constant", []string{logTable,
			`CREATE FUNCTION eval_purge_fixture() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE NOTICE 'a -- b'; DELETE FROM eval_purge_log; RETURN OLD; END $$`,
			`CREATE TRIGGER eval_members_logged AFTER DELETE ON eval_members FOR EACH ROW EXECUTE FUNCTION eval_purge_fixture()`},
			[]string{"trigger eval_members_logged on eval_members writes [eval_purge_log] when the purge deletes from it"}},
		{"deletion trigger with a dollar-quoted constant", []string{logTable,
			`CREATE FUNCTION eval_purge_fixture() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE NOTICE $note$DELETE FROM eval_members -- it's$note$; INSERT INTO eval_purge_log VALUES (OLD.member_id); RETURN OLD; END $$`,
			`CREATE TRIGGER eval_members_logged AFTER DELETE ON eval_members FOR EACH ROW EXECUTE FUNCTION eval_purge_fixture()`},
			[]string{"trigger eval_members_logged on eval_members writes [eval_purge_log] when the purge deletes from it"}},
		{"rule that writes when the purge deletes", []string{logTable,
			`CREATE RULE eval_members_logged AS ON DELETE TO eval_members DO ALSO INSERT INTO eval_purge_log VALUES (OLD.member_id)`},
			[]string{"rule eval_members_logged on eval_members rewrites"}},
		{"declared trigger that no longer writes", []string{
			`DROP TRIGGER eval_datasets_collection ON eval_dataset_revisions`},
			[]string{"the purge counts trigger eval_datasets_collection on eval_dataset_revisions, which is no writing deletion trigger"}},
		{"unmeasured column", []string{`ALTER TABLE eval_members ADD COLUMN note text`},
			[]string{"eval_members has columns"}},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			tx, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = tx.Rollback(ctx) }()
			var schema string
			if err := tx.QueryRow(ctx, `SELECT current_schema()`).Scan(&schema); err != nil {
				t.Fatal(err)
			}
			for _, statement := range fixture.ddl {
				statement = strings.ReplaceAll(statement, "{schema}", pgx.Identifier{schema}.Sanitize())
				if _, err := tx.Exec(ctx, statement); err != nil {
					t.Fatalf("%s: %v", statement, err)
				}
			}
			violations, err := purgeCoverageViolations(ctx, tx)
			if err != nil {
				t.Fatal(err)
			}
			if len(violations) != len(fixture.want) {
				t.Fatalf("violations = %q, want %d naming %q", violations, len(fixture.want), fixture.want)
			}
			for i, want := range fixture.want {
				if !strings.HasPrefix(violations[i], want) {
					t.Errorf("violation %d = %q, want %q", i, violations[i], want)
				}
			}
		})
	}
}
