package persistence

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// This file applies PER-R23 to the functions that migrations define. A
// trigger's function acts for the owner of the trigger's table, and a
// function that Go SQL calls acts for the calling package; either acts
// through the functions it calls in turn. The check reads their bodies from
// the migrated schema with explaintest.ReadBody: a trigger may write or
// exclusively lock another owner's relation, and a called function may read
// one, only where the [routines] section of the registry lists that access
// with its reason, and a called function writes only the caller's
// relations. The reads that triggers make stay outside the rule.

// routineTrigger is a trigger of the migrated schema and its function.
type routineTrigger struct{ table, name, function string }

// routineBody is what one function's body does with the registry's
// relations and the schema's other functions, overloads merged.
type routineBody struct {
	reads, writes, calls []string
	dynamic              bool
	// unreadable says why the check cannot read the body.
	unreadable string
}

// migrationRoutines are the functions and triggers of a migrated schema.
type migrationRoutines struct {
	bodies   map[string]*routineBody
	triggers []routineTrigger
}

func (r migrationRoutines) names() map[string]bool {
	names := make(map[string]bool, len(r.bodies))
	for name := range r.bodies {
		names[name] = true
	}
	return names
}

// routineEffect is an access to a relation by the body of function.
type routineEffect struct {
	function, relation string
	write              bool
}

// reach returns the accesses that running function makes, through the
// functions it calls, and the functions on the way whose bodies the check
// cannot follow, with why.
func (r migrationRoutines) reach(function string) ([]routineEffect, map[string]string) {
	var effects []routineEffect
	blind := map[string]string{}
	seen := map[string]bool{}
	var visit func(string)
	visit = func(name string) {
		body := r.bodies[name]
		if seen[name] || body == nil {
			return
		}
		seen[name] = true
		switch {
		case body.unreadable != "":
			blind[name] = body.unreadable
		case body.dynamic:
			blind[name] = "it runs EXECUTE, whose SQL the check cannot read"
		}
		for _, relation := range body.reads {
			effects = append(effects, routineEffect{function: name, relation: relation})
		}
		for _, relation := range body.writes {
			effects = append(effects, routineEffect{function: name, relation: relation, write: true})
		}
		for _, callee := range body.calls {
			visit(callee)
		}
	}
	visit(function)
	return effects, blind
}

// routinesFunctionsSQL lists the functions of the current schema with their
// language and source; a SQL function with a BEGIN ATOMIC body keeps no
// source text.
const routinesFunctionsSQL = `
SELECT p.proname::text, l.lanname::text, p.prosrc, p.prosqlbody IS NOT NULL
FROM pg_proc AS p JOIN pg_language AS l ON l.oid = p.prolang
WHERE p.pronamespace = current_schema()::regnamespace
ORDER BY 1`

// routinesTriggersSQL lists the triggers that migrations define on the
// relations of the current schema; internal triggers enforce foreign keys.
const routinesTriggersSQL = `
SELECT c.relname::text, t.tgname::text, p.proname::text, p.pronamespace = c.relnamespace
FROM pg_trigger AS t
JOIN pg_class AS c ON c.oid = t.tgrelid
JOIN pg_proc AS p ON p.oid = t.tgfoid
WHERE NOT t.tgisinternal AND c.relnamespace = current_schema()::regnamespace
ORDER BY 1, 2`

// loadMigrationRoutines reads the functions and triggers of the schema that
// pool's search path selects, keeping of each body the relations of
// relations it reads and writes and the schema's functions it calls.
func loadMigrationRoutines(ctx context.Context, pool *pgxpool.Pool, relations map[string]ownedRelation) (migrationRoutines, error) {
	var schema string
	if err := pool.QueryRow(ctx, `SELECT current_schema()`).Scan(&schema); err != nil {
		return migrationRoutines{}, err
	}
	type functionSource struct{ name, text string }
	var sources []functionSource
	var name, language, text string
	var atomic bool
	rows, err := pool.Query(ctx, routinesFunctionsSQL)
	if err != nil {
		return migrationRoutines{}, err
	}
	routines := migrationRoutines{bodies: map[string]*routineBody{}}
	if _, err := pgx.ForEachRow(rows, []any{&name, &language, &text, &atomic}, func() error {
		body := routines.bodies[name]
		if body == nil {
			body = &routineBody{}
			routines.bodies[name] = body
		}
		switch {
		case language != "plpgsql" && language != "sql":
			body.unreadable = "it is written in " + language
		case atomic:
			body.unreadable = "its BEGIN ATOMIC body keeps no source text"
		default:
			sources = append(sources, functionSource{name: name, text: text})
		}
		return nil
	}); err != nil {
		return migrationRoutines{}, err
	}
	// local returns the object of a name in the current schema, or "".
	local := func(name explaintest.Name) string {
		if name.Schema != "" && name.Schema != schema {
			return ""
		}
		return name.Object
	}
	for _, source := range sources {
		read, err := explaintest.ReadBody(source.text)
		if err != nil {
			return migrationRoutines{}, fmt.Errorf("read the body of %s: %w", source.name, err)
		}
		body := routines.bodies[source.name]
		body.dynamic = body.dynamic || read.Dynamic
		for _, name := range read.Reads {
			if _, ok := relations[local(name)]; ok {
				body.reads = append(body.reads, local(name))
			}
		}
		for _, name := range append(read.Writes, read.Locks...) {
			if _, ok := relations[local(name)]; ok {
				body.writes = append(body.writes, local(name))
			}
		}
		for _, name := range read.Calls {
			if callee := local(name); routines.bodies[callee] != nil && callee != source.name {
				body.calls = append(body.calls, callee)
			}
		}
	}
	var trigger routineTrigger
	var sameSchema bool
	rows, err = pool.Query(ctx, routinesTriggersSQL)
	if err != nil {
		return migrationRoutines{}, err
	}
	if _, err := pgx.ForEachRow(rows, []any{&trigger.table, &trigger.name, &trigger.function, &sameSchema}, func() error {
		if !sameSchema {
			return fmt.Errorf("trigger %s on %s runs %s of another schema", trigger.name, trigger.table, trigger.function)
		}
		routines.triggers = append(routines.triggers, trigger)
		return nil
	}); err != nil {
		return migrationRoutines{}, err
	}
	return routines, nil
}

// checkRoutineOwnership returns one message per access of a trigger or of a
// function that Go SQL calls that the registry does not allow, and per
// [routines] entry that no access needs, sorted.
func checkRoutineOwnership(registry ownershipRegistry, routines migrationRoutines, calls []sqlCall) []string {
	type observed struct{ read, write string } // first place of each kind of access
	seen := map[routineAccess]*observed{}
	problems := map[string]bool{}
	owner := func(relation string) string { return registry.relations[relation].owner }
	note := func(key routineAccess, write bool, via string) {
		found := seen[key]
		if found == nil {
			found = &observed{}
			seen[key] = found
		}
		first := &found.read
		if write {
			first = &found.write
		}
		if *first == "" || via < *first {
			*first = via
		}
	}
	follow := func(via, function string) []routineEffect {
		effects, blind := routines.reach(function)
		for name, why := range blind {
			problems[fmt.Sprintf("%s runs %s, and the check cannot read %s: %s; write its statements in a PL/pgSQL or SQL body",
				via, function, name, why)] = true
		}
		return effects
	}
	for _, trigger := range routines.triggers {
		table, ok := registry.relations[trigger.table]
		if !ok {
			problems[fmt.Sprintf("trigger %s on %s: %s is not listed under [relations] in %s",
				trigger.name, trigger.table, trigger.table, ownershipRegistryFile)] = true
			continue
		}
		via := fmt.Sprintf("trigger %s on %s", trigger.name, trigger.table)
		for _, effect := range follow(via, trigger.function) {
			if effect.write && owner(effect.relation) != table.owner {
				note(routineAccess{pkg: table.owner, function: effect.function, relation: effect.relation}, true, via)
			}
		}
	}
	for _, call := range calls {
		via := call.position + ": " + call.pkg
		effects := follow(via, call.function)
		written := map[routineEffect]bool{}
		for _, effect := range effects {
			if effect.write {
				written[routineEffect{function: effect.function, relation: effect.relation}] = true
			}
		}
		for _, effect := range effects {
			switch {
			case owner(effect.relation) == call.pkg:
			case !effect.write && written[effect]:
				// The write below covers the read of a locked row.
			case effect.write:
				problems[fmt.Sprintf(
					"%s writes %s through %s in %s, which %s owns (PER-R23); a migration function that Go SQL calls writes only the caller's relations, so move the write behind %s",
					via, effect.relation, call.function, effect.function, owner(effect.relation), owner(effect.relation))] = true
			default:
				note(routineAccess{pkg: call.pkg, function: effect.function, relation: effect.relation}, false, via)
			}
		}
	}
	for key, found := range seen {
		entry := registry.routines[key]
		if found.write != "" && (entry == nil || !entry.write) {
			problems[fmt.Sprintf(
				"%s, for %s, writes %s in %s, which %s owns (PER-R23); list the write under [routines] in %s with its reason, or move it behind %s",
				found.write, key.pkg, key.relation, key.function, owner(key.relation), ownershipRegistryFile, owner(key.relation))] = true
		}
		if found.read != "" && entry == nil {
			problems[fmt.Sprintf(
				"%s reads %s through %s, which %s owns (PER-R23); list the read under [routines] in %s with its reason",
				found.read, key.relation, key.function, owner(key.relation), ownershipRegistryFile)] = true
		}
	}
	for key, entry := range registry.routines {
		found := seen[key]
		if found == nil || (entry.write && found.write == "") || (!entry.write && found.read == "") {
			access := "read"
			if entry.write {
				access = "write"
			}
			problems[fmt.Sprintf("%s:%d: [routines] %s %s %s %s matches no access; delete the entry",
				ownershipRegistryFile, entry.line, key.pkg, key.function, access, key.relation)] = true
		}
	}
	return slices.Sorted(maps.Keys(problems))
}

// migratedSchema returns a pool on a new schema of the test database with
// every migration applied, dropped at cleanup. It skips the test only while
// CONTRACTOR_TEST_DATABASE_URL is unset.
func migratedSchema(t *testing.T) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		t.Fatal(err)
	}
	schema := "ownership_" + hex.EncodeToString(random)
	admin, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(admin.Close)
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+schema); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		if _, err := admin.Exec(cleanup, `DROP SCHEMA `+schema+` CASCADE`); err != nil {
			t.Errorf("drop schema %s: %v", schema, err)
		}
	})
	config, err := pgxpool.ParseConfig(url)
	if err != nil {
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}

// TestMigrationRoutinesFollowTableOwnership is the PER-R23 check of the
// migration functions: every trigger write and exclusive lock of another
// owner's relation, and every such read by a function that Go SQL calls, is
// a [routines] entry, every entry still matches one, and no called function
// writes another owner's relation.
func TestMigrationRoutinesFollowTableOwnership(t *testing.T) {
	pool := migratedSchema(t)
	root := moduleRootDir(t)
	registry := loadOwnershipRegistry(t, root)
	routines, err := loadMigrationRoutines(context.Background(), pool, registry.relations)
	if err != nil {
		t.Fatal(err)
	}
	if len(routines.triggers) < 50 || len(routines.bodies) < 50 {
		t.Fatalf("read %d triggers and %d functions; the schema queries are broken", len(routines.triggers), len(routines.bodies))
	}
	scanned, files, err := scanModuleSQL(root, registry.relationNames(), routines.names())
	if err != nil {
		t.Fatal(err)
	}
	if files < 100 || len(scanned.calls) == 0 {
		t.Fatalf("scanned %d Go files and found %d calls of migration functions; the scan is broken", files, len(scanned.calls))
	}
	for _, problem := range checkRoutineOwnership(registry, routines, scanned.calls) {
		t.Error(problem)
	}
}

// probeRoutinesSQL is a probe migration: a runstore trigger that changes
// projectstore's projects, and a function that locks a Project exclusively.
const probeRoutinesSQL = `
CREATE FUNCTION contractor_probe_touch_project() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE projects SET updated_at = clock_timestamp() WHERE project_id = NEW.project_id;
    RETURN NEW;
END;
$$;
CREATE TRIGGER workflow_runs_probe_touch_project AFTER INSERT ON workflow_runs
    FOR EACH ROW EXECUTE FUNCTION contractor_probe_touch_project();
CREATE FUNCTION contractor_probe_lock_project(selected text) RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    PERFORM 1 FROM projects WHERE project_id = selected FOR NO KEY UPDATE;
END;
$$;`

// TestRoutineOwnershipCheckRejectsProbeRoutines is acceptance A2 of
// V372-009's check on PostgreSQL: a probe migration whose trigger writes
// another owner's relation, and a function that Go SQL calls to lock one,
// fail it; a [routines] entry admits the trigger's write but not the call's
// lock.
func TestRoutineOwnershipCheckRejectsProbeRoutines(t *testing.T) {
	pool := migratedSchema(t)
	ctx := context.Background()
	if _, err := pool.Exec(ctx, probeRoutinesSQL); err != nil {
		t.Fatal(err)
	}
	root := moduleRootDir(t)
	registry := loadOwnershipRegistry(t, root)
	routines, err := loadMigrationRoutines(ctx, pool, registry.relations)
	if err != nil {
		t.Fatal(err)
	}
	scanned, _, err := scanModuleSQL(root, registry.relationNames(), routines.names())
	if err != nil {
		t.Fatal(err)
	}
	probe, err := scanPackageSQL("internal/runstore", map[string][]byte{
		"internal/runstore/probe.go": []byte("package runstore\n\nconst lockProject = `SELECT contractor_probe_lock_project($1)`\n"),
	}, registry.relationNames(), routines.names())
	if err != nil {
		t.Fatal(err)
	}
	calls := append(scanned.calls, probe.calls...)
	want := []string{
		"internal/runstore/probe.go:3: internal/runstore writes projects through contractor_probe_lock_project in contractor_probe_lock_project, which internal/projectstore owns (PER-R23)",
		"trigger workflow_runs_probe_touch_project on workflow_runs, for internal/runstore, writes projects in contractor_probe_touch_project, which internal/projectstore owns (PER-R23)",
	}
	check := func(want []string) {
		t.Helper()
		slices.Sort(want)
		got := checkRoutineOwnership(registry, routines, calls)
		matches := len(got) == len(want)
		for index := 0; matches && index < len(want); index++ {
			matches = strings.HasPrefix(got[index], want[index])
		}
		if !matches {
			t.Fatalf("problems =\n%s\nwant prefixes\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
		}
	}
	check(want)
	registry.routines[routineAccess{pkg: "internal/runstore", function: "contractor_probe_touch_project", relation: "projects"}] = &routineEntry{write: true, reason: "probe"}
	registry.routines[routineAccess{pkg: "internal/runstore", function: "contractor_probe_lock_project", relation: "projects"}] = &routineEntry{write: true, reason: "probe"}
	check(append(want[:1:1], ownershipRegistryFile+":0: [routines] internal/runstore contractor_probe_lock_project write projects matches no access"))
}

// TestRoutineOwnershipCheckReportsEachRule checks the rules of
// checkRoutineOwnership on routines given directly.
func TestRoutineOwnershipCheckReportsEachRule(t *testing.T) {
	registry, problems := parseOwnershipRegistry(`
[relations]
runs        internal/runstore       exec
allocations internal/runstore       exec
evals       internal/evalstore      evals
members     internal/evalstore      evals
reports     internal/telemetry      exec

[routines]
internal/runstore  mark_dirty       write evals    Eval projection
internal/telemetry run_members      read  members  metric retention
internal/telemetry run_members      read  evals    stale: the function no longer reads evals
internal/telemetry mark_received    write runs     stale: the function writes allocations
`)
	if len(problems) > 0 {
		t.Fatal(problems)
	}
	routines := migrationRoutines{
		bodies: map[string]*routineBody{
			"run_changed":   {calls: []string{"mark_dirty", "run_members"}},
			"mark_dirty":    {reads: []string{"evals"}, writes: []string{"evals", "members"}},
			"run_members":   {reads: []string{"members", "runs"}},
			"mark_received": {writes: []string{"allocations"}},
			"renew":         {calls: []string{"run_members", "touch_runs"}},
			"touch_runs":    {writes: []string{"runs"}},
			"dynamic":       {dynamic: true},
			"native":        {unreadable: "it is written in c"},
		},
		triggers: []routineTrigger{
			{table: "runs", name: "runs_changed", function: "run_changed"},
			{table: "reports", name: "reports_received", function: "mark_received"},
			{table: "missing", name: "missing_changed", function: "run_changed"},
		},
	}
	calls := []sqlCall{
		{pkg: "internal/telemetry", function: "renew", position: "internal/telemetry/a.go:1"},
		{pkg: "internal/telemetry", function: "dynamic", position: "internal/telemetry/a.go:2"},
		{pkg: "internal/runstore", function: "native", position: "internal/runstore/a.go:3"},
		{pkg: "internal/evalstore", function: "mark_dirty", position: "internal/evalstore/a.go:4"},
	}
	got := checkRoutineOwnership(registry, routines, calls)
	want := []string{
		ownershipRegistryFile + ":12: [routines] internal/telemetry run_members read evals matches no access; delete the entry",
		ownershipRegistryFile + ":13: [routines] internal/telemetry mark_received write runs matches no access; delete the entry",
		"internal/runstore/a.go:3: internal/runstore runs native, and the check cannot read native: it is written in c; write its statements in a PL/pgSQL or SQL body",
		"internal/telemetry/a.go:1: internal/telemetry reads runs through run_members, which internal/runstore owns (PER-R23); list the read under [routines] in " +
			ownershipRegistryFile + " with its reason",
		"internal/telemetry/a.go:1: internal/telemetry writes runs through renew in touch_runs, which internal/runstore owns (PER-R23); a migration function that Go SQL calls writes only the caller's relations, so move the write behind internal/runstore",
		"internal/telemetry/a.go:2: internal/telemetry runs dynamic, and the check cannot read dynamic: it runs EXECUTE, whose SQL the check cannot read; write its statements in a PL/pgSQL or SQL body",
		"trigger missing_changed on missing: missing is not listed under [relations] in " + ownershipRegistryFile,
		"trigger reports_received on reports, for internal/telemetry, writes allocations in mark_received, which internal/runstore owns (PER-R23); list the write under [routines] in " +
			ownershipRegistryFile + " with its reason, or move it behind internal/runstore",
		"trigger runs_changed on runs, for internal/runstore, writes members in mark_dirty, which internal/evalstore owns (PER-R23); list the write under [routines] in " +
			ownershipRegistryFile + " with its reason, or move it behind internal/evalstore",
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("problems =\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
}
