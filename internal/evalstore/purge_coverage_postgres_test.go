package evalstore

import (
	"context"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strings"
	"testing"

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
const purgeFunctionsSQL = `
SELECT proname::text, string_agg(prosrc, E'\n')
FROM pg_proc
WHERE pronamespace = current_schema()::regnamespace
GROUP BY proname`

// purgeColumnsSQL lists every column of the current schema's tables in
// column order.
const purgeColumnsSQL = `
SELECT table_name::text, array_agg(column_name::text ORDER BY ordinal_position)
FROM information_schema.columns
WHERE table_schema = current_schema()
GROUP BY table_name`

var (
	sqlComment = regexp.MustCompile(`--[^\n]*`)
	sqlLiteral = regexp.MustCompile(`'(?:[^']|'')*'`)
	sqlWrite   = regexp.MustCompile(`(?i)\b(?:INSERT\s+INTO|UPDATE(?:\s+ONLY)?|DELETE\s+FROM(?:\s+ONLY)?|MERGE\s+INTO)\s+([a-z_][a-z0-9_]*)`)
	sqlDynamic = regexp.MustCompile(`(?i)\bEXECUTE\b`)
	sqlCall    = regexp.MustCompile(`(?i)\b([a-z_][a-z0-9_]*)\s*\(`)
	// Words that follow UPDATE without naming a table: ON CONFLICT DO UPDATE
	// SET and the row locks FOR UPDATE [OF | NOWAIT | SKIP LOCKED].
	notTable = map[string]bool{"set": true, "of": true, "nowait": true, "skip": true}
)

// writtenTables returns the tables the PL/pgSQL function fn writes, directly
// or through the functions of the schema it calls, sorted; "?" stands for
// dynamic SQL, which may write anything. Comments and string literals are
// ignored, so a function that only checks TG_OP = 'DELETE' writes nothing.
func writtenTables(functions map[string]string, fn string, seen map[string]bool) []string {
	if seen[fn] {
		return nil
	}
	seen[fn] = true
	source := sqlLiteral.ReplaceAllString(sqlComment.ReplaceAllString(functions[fn], " "), "''")
	written := map[string]bool{}
	for _, match := range sqlWrite.FindAllStringSubmatch(source, -1) {
		if table := strings.ToLower(match[1]); !notTable[table] {
			written[table] = true
		}
	}
	if sqlDynamic.MatchString(source) {
		written["?"] = true
	}
	for _, match := range sqlCall.FindAllStringSubmatch(source, -1) {
		if callee := strings.ToLower(match[1]); callee != fn {
			if _, ok := functions[callee]; ok {
				for _, table := range writtenTables(functions, callee, seen) {
					written[table] = true
				}
			}
		}
	}
	return slices.Sorted(maps.Keys(written))
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
	updated := regexp.MustCompile(`UPDATE (\w+)`).FindStringSubmatch(invalidatePurgedViewSQL)[1]
	fired = append(fired, event{updated, triggerUpdate})
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
			written := writtenTables(functions, tr.function, map[string]bool{})
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
			for _, statement := range fixture.ddl {
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
