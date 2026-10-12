package persistence

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// ownershipRegistryFile is the table ownership registry of PER-R18,
// relative to the module root; it lives next to this test.
const ownershipRegistryFile = "internal/persistence/table_ownership.txt"

// ownershipRegistry is the parsed table ownership registry: the owner of
// every relation, the reads other packages may make, and the known foreign
// accesses that a task still has to remove.
type ownershipRegistry struct {
	relations map[string]ownedRelation
	reads     map[accessPair]*registryRead
	known     map[accessPair]*knownAccess
}

type ownedRelation struct {
	owner string
	group string // scripts/erd/meta.json group, or "-" when the ERD does not draw it
	line  int
}

// accessPair is a package that is not the owner and a relation it uses.
type accessPair struct{ pkg, relation string }

type registryRead struct {
	reason string
	line   int
}

type knownAccess struct {
	write  bool
	task   string
	reason string
	line   int
}

var (
	registryPackage  = regexp.MustCompile(`^(api|cmd|internal|tools)(/[a-z0-9_]+)*$`)
	registryRelation = regexp.MustCompile(`^[a-z_][a-z0-9_]*$`)
	// registryTask is a task ID as scripts/check_tasks.py accepts it.
	registryTask = regexp.MustCompile(`^(MVP|V[0-9]+)-[0-9]{3}[A-Z]?$`)
)

// parseOwnershipRegistry reads the registry text and returns it with one
// message per malformed or inconsistent line.
func parseOwnershipRegistry(source string) (ownershipRegistry, []string) {
	registry := ownershipRegistry{
		relations: map[string]ownedRelation{},
		reads:     map[accessPair]*registryRead{},
		known:     map[accessPair]*knownAccess{},
	}
	var problems []string
	fail := func(line int, format string, args ...any) {
		problems = append(problems, fmt.Sprintf("%s:%d: %s", ownershipRegistryFile, line, fmt.Sprintf(format, args...)))
	}
	section := ""
	var reason *string
	type pendingPair struct {
		pair accessPair
		line int
	}
	var pairs []pendingPair
	for index, text := range strings.Split(source, "\n") {
		line := index + 1
		trimmed := strings.TrimSpace(text)
		switch {
		case trimmed == "" || strings.HasPrefix(trimmed, "#"):
			reason = nil
			continue
		case text[0] == ' ' || text[0] == '\t':
			if reason == nil {
				fail(line, "an indented line must continue the reason of the entry above it")
				continue
			}
			*reason += " " + trimmed
			continue
		}
		reason = nil
		if strings.HasPrefix(trimmed, "[") {
			if trimmed != "[relations]" && trimmed != "[reads]" && trimmed != "[known]" {
				fail(line, "unknown section %s; use [relations], [reads] or [known]", trimmed)
			}
			section = trimmed
			continue
		}
		fields := strings.Fields(trimmed)
		switch section {
		case "[relations]":
			if len(fields) != 3 {
				fail(line, "a relation line is: <relation> <owner package> <ERD group or ->")
				continue
			}
			name, owner, group := fields[0], fields[1], fields[2]
			if !registryRelation.MatchString(name) || !registryPackage.MatchString(owner) {
				fail(line, "%q is not a relation name with an owner package path", trimmed)
				continue
			}
			if previous, ok := registry.relations[name]; ok {
				fail(line, "relation %s is already listed at line %d", name, previous.line)
				continue
			}
			registry.relations[name] = ownedRelation{owner: owner, group: group, line: line}
		case "[reads]":
			if len(fields) < 3 {
				fail(line, "a read line is: <package> <relation> <reason>")
				continue
			}
			pair := accessPair{pkg: fields[0], relation: fields[1]}
			if previous, ok := registry.reads[pair]; ok {
				fail(line, "the read of %s by %s is already listed at line %d", pair.relation, pair.pkg, previous.line)
				continue
			}
			entry := &registryRead{reason: strings.Join(fields[2:], " "), line: line}
			registry.reads[pair] = entry
			reason = &entry.reason
			pairs = append(pairs, pendingPair{pair: pair, line: line})
		case "[known]":
			if len(fields) < 5 || (fields[1] != "read" && fields[1] != "write") || !registryTask.MatchString(fields[3]) {
				fail(line, "a known access line is: <package> read|write <relation> <task ID> <reason>")
				continue
			}
			pair := accessPair{pkg: fields[0], relation: fields[2]}
			if previous, ok := registry.known[pair]; ok {
				fail(line, "the access of %s by %s is already listed at line %d; a write also covers reads",
					pair.relation, pair.pkg, previous.line)
				continue
			}
			entry := &knownAccess{
				write: fields[1] == "write", task: fields[3], reason: strings.Join(fields[4:], " "), line: line,
			}
			registry.known[pair] = entry
			reason = &entry.reason
			pairs = append(pairs, pendingPair{pair: pair, line: line})
		default:
			fail(line, "entry outside a section")
		}
	}
	for _, pending := range pairs {
		pair := pending.pair
		relation, ok := registry.relations[pair.relation]
		switch {
		case !registryPackage.MatchString(pair.pkg):
			fail(pending.line, "%q is not a package path", pair.pkg)
		case !ok:
			fail(pending.line, "%s is not listed under [relations]", pair.relation)
		case relation.owner == pair.pkg:
			fail(pending.line, "%s owns %s and needs no entry", pair.pkg, pair.relation)
		}
		read, granted := registry.reads[pair]
		if known, ok := registry.known[pair]; ok && granted && read.line == pending.line {
			fail(pending.line, "the read of %s by %s is also a known access at line %d; keep one entry",
				pair.relation, pair.pkg, known.line)
		}
	}
	return registry, problems
}

func (r ownershipRegistry) relationNames() map[string]bool {
	names := make(map[string]bool, len(r.relations))
	for name := range r.relations {
		names[name] = true
	}
	return names
}

// checkTableOwnership returns one message per access that the registry does
// not allow and per registry entry that no access needs, sorted. taskState
// returns "" for an open task and otherwise why the task cannot remove an
// access.
func checkTableOwnership(registry ownershipRegistry, accesses []sqlAccess, taskState func(string) string) []string {
	type observed struct{ read, write string } // first position of each kind of access
	seen := map[accessPair]*observed{}
	for _, access := range accesses {
		relation, ok := registry.relations[access.relation]
		if !ok || relation.owner == access.pkg {
			continue
		}
		pair := accessPair{pkg: access.pkg, relation: access.relation}
		found := seen[pair]
		if found == nil {
			found = &observed{}
			seen[pair] = found
		}
		first := &found.read
		if access.write {
			first = &found.write
		}
		if *first == "" || access.position < *first {
			*first = access.position
		}
	}

	var problems []string
	for pair, found := range seen {
		owner := registry.relations[pair.relation].owner
		known, isKnown := registry.known[pair]
		_, granted := registry.reads[pair]
		if found.write != "" && (!isKnown || !known.write) {
			problems = append(problems, fmt.Sprintf(
				"%s: %s writes %s, which %s owns (PER-R18); move the statement behind %s, or list the access under [known] in %s with the open task that moves it",
				found.write, pair.pkg, pair.relation, owner, owner, ownershipRegistryFile))
		}
		if found.read != "" && !granted && !isKnown {
			problems = append(problems, fmt.Sprintf(
				"%s: %s reads %s, which %s owns (PER-R18); read it through %s, or grant the read with its reason under [reads] in %s",
				found.read, pair.pkg, pair.relation, owner, owner, ownershipRegistryFile))
		}
	}
	for pair, read := range registry.reads {
		if found := seen[pair]; found == nil || found.read == "" {
			problems = append(problems, fmt.Sprintf("%s:%d: [reads] %s %s matches no read; delete the entry",
				ownershipRegistryFile, read.line, pair.pkg, pair.relation))
		}
	}
	for pair, known := range registry.known {
		found := seen[pair]
		switch {
		case found == nil:
			problems = append(problems, fmt.Sprintf("%s:%d: [known] %s %s matches no access; delete the entry",
				ownershipRegistryFile, known.line, pair.pkg, pair.relation))
		case known.write && found.write == "":
			problems = append(problems, fmt.Sprintf(
				"%s:%d: [known] %s no longer writes %s; delete the entry, or make it a read if reads remain",
				ownershipRegistryFile, known.line, pair.pkg, pair.relation))
		}
		if state := taskState(known.task); state != "" {
			problems = append(problems, fmt.Sprintf("%s:%d: [known] %s %s names %s, which %s",
				ownershipRegistryFile, known.line, pair.pkg, pair.relation, known.task, state))
		}
	}
	slices.Sort(problems)
	return problems
}

// openTaskIn returns the taskState of the task files under root/tasks: an
// open task lives in tasks/, a finished one in tasks/done/.
func openTaskIn(root string) func(string) string {
	return func(task string) string {
		name := strings.ToLower(task) + "-*.yml"
		if open, _ := filepath.Glob(filepath.Join(root, "tasks", name)); len(open) == 1 {
			return ""
		}
		if done, _ := filepath.Glob(filepath.Join(root, "tasks", "done", name)); len(done) == 1 {
			return "is finished; remove the access or name the open task that removes it"
		}
		return "has no task file; name the open task that removes the access"
	}
}

func moduleRootDir(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func loadOwnershipRegistry(t *testing.T, root string) ownershipRegistry {
	t.Helper()
	source, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(ownershipRegistryFile)))
	if err != nil {
		t.Fatal(err)
	}
	registry, problems := parseOwnershipRegistry(string(source))
	if len(problems) > 0 {
		t.Fatalf("malformed table ownership registry:\n%s", strings.Join(problems, "\n"))
	}
	if len(registry.relations) == 0 {
		t.Fatalf("%s lists no relations", ownershipRegistryFile)
	}
	return registry
}

// TestSQLAccessFollowsTableOwnership is the PER-R18 fitness check: SQL text
// needs no Go import, so internal/archtest cannot see one package writing
// another package's table. Every foreign write must be a known access with an
// open task, every foreign read a granted read or a known access, and every
// registry entry must still match an access.
func TestSQLAccessFollowsTableOwnership(t *testing.T) {
	root := moduleRootDir(t)
	registry := loadOwnershipRegistry(t, root)
	accesses, files, err := scanModuleSQL(root, registry.relationNames())
	if err != nil {
		t.Fatal(err)
	}
	if files < 100 {
		t.Fatalf("scanned only %d Go files; the repository walk is broken", files)
	}
	for _, problem := range checkTableOwnership(registry, accesses, openTaskIn(root)) {
		t.Error(problem)
	}
}

// TestTableOwnershipMatchesTheSchemaDocuments keeps the ERD and the
// specification on the registry's relations and owners: scripts/erd/meta.json
// draws every relation of the registry with an ERD group in that group, and
// the "Table owners" table of persistence.md names the same owners. The
// PostgreSQL integration test of internal/persistence/postgres compares the
// registry with the migrated schema.
func TestTableOwnershipMatchesTheSchemaDocuments(t *testing.T) {
	root := moduleRootDir(t)
	registry := loadOwnershipRegistry(t, root)

	for _, name := range slices.Sorted(maps.Keys(registry.relations)) {
		relation := registry.relations[name]
		entries, err := os.ReadDir(filepath.Join(root, filepath.FromSlash(relation.owner)))
		if err != nil || !slices.ContainsFunc(entries, func(entry os.DirEntry) bool {
			return strings.HasSuffix(entry.Name(), ".go") && !strings.HasSuffix(entry.Name(), "_test.go")
		}) {
			t.Errorf("%s:%d: owner %s of %s is not a Go package", ownershipRegistryFile, relation.line, relation.owner, name)
		}
	}

	raw, err := os.ReadFile(filepath.Join(root, "scripts", "erd", "meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		Groups []struct {
			ID string `json:"id"`
		} `json:"groups"`
		Tables map[string][]json.RawMessage `json:"tables"`
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	groups := map[string]bool{}
	for _, group := range meta.Groups {
		groups[group.ID] = true
	}
	drawn := map[string]string{}
	for name, entry := range meta.Tables {
		var group string
		if len(entry) == 0 || json.Unmarshal(entry[0], &group) != nil {
			t.Fatalf("scripts/erd/meta.json: table %s has no group", name)
		}
		drawn[name] = group
	}
	for _, problem := range erdGroupProblems(registry, groups, drawn) {
		t.Error(problem)
	}

	spec, err := os.ReadFile(filepath.Join(root, "docs", "spec", "platform", "persistence.md"))
	if err != nil {
		t.Fatal(err)
	}
	for _, problem := range specOwnerProblems(registry, string(spec)) {
		t.Error(problem)
	}
}

// erdGroupProblems compares the registry with the ERD's group of each table
// it draws.
func erdGroupProblems(registry ownershipRegistry, groups map[string]bool, drawn map[string]string) []string {
	var problems []string
	for name, relation := range registry.relations {
		group, ok := drawn[name]
		switch {
		case relation.group == "-" && ok:
			problems = append(problems, fmt.Sprintf("%s:%d: scripts/erd/meta.json draws %s in group %s; give it that group here",
				ownershipRegistryFile, relation.line, name, group))
		case relation.group == "-":
		case !groups[relation.group]:
			problems = append(problems, fmt.Sprintf("%s:%d: %s is not a group of scripts/erd/meta.json",
				ownershipRegistryFile, relation.line, relation.group))
		case !ok:
			problems = append(problems, fmt.Sprintf("%s:%d: scripts/erd/meta.json does not describe %s; add it to group %s",
				ownershipRegistryFile, relation.line, name, relation.group))
		case group != relation.group:
			problems = append(problems, fmt.Sprintf("%s:%d: scripts/erd/meta.json draws %s in group %s, the registry in %s",
				ownershipRegistryFile, relation.line, name, group, relation.group))
		}
	}
	for name := range drawn {
		if _, ok := registry.relations[name]; !ok {
			problems = append(problems, fmt.Sprintf("scripts/erd/meta.json describes %s, which %s does not list",
				name, ownershipRegistryFile))
		}
	}
	slices.Sort(problems)
	return problems
}

// specOwnersHeading is the persistence.md heading of the owner table.
const specOwnersHeading = "### Table owners"

var (
	specOwnerRow  = regexp.MustCompile("^\\| `([^`]+)` \\| (.*) \\|$")
	specCodeSpans = regexp.MustCompile("`([^`]+)`")
)

// specOwnerProblems compares the registry with the owner table under the
// "Table owners" heading of persistence.md: one row per owner package, its
// relations as code spans.
func specOwnerProblems(registry ownershipRegistry, spec string) []string {
	const document = "docs/spec/platform/persistence.md"
	lines := strings.Split(spec, "\n")
	start := slices.Index(lines, specOwnersHeading)
	if start < 0 {
		return []string{fmt.Sprintf("%s has no %q section", document, specOwnersHeading)}
	}
	listed := map[string]string{} // relation -> owner in the specification
	var problems []string
	inTable := false
	for index := start + 1; index < len(lines); index++ {
		line := lines[index]
		if !strings.HasPrefix(line, "|") {
			if inTable || strings.HasPrefix(line, "#") {
				break
			}
			continue
		}
		inTable = true
		row := specOwnerRow.FindStringSubmatch(line)
		if row == nil {
			continue // the header and separator rows
		}
		for _, span := range specCodeSpans.FindAllStringSubmatch(row[2], -1) {
			name := span[1]
			if previous, ok := listed[name]; ok {
				problems = append(problems, fmt.Sprintf("%s:%d: %s is listed for %s and %s", document, index+1, name, previous, row[1]))
			}
			listed[name] = row[1]
		}
	}
	for name, relation := range registry.relations {
		owner, ok := listed[name]
		switch {
		case !ok:
			problems = append(problems, fmt.Sprintf("%s: %q does not list %s for %s", document, specOwnersHeading, name, relation.owner))
		case owner != relation.owner:
			problems = append(problems, fmt.Sprintf("%s: %q lists %s for %s, the registry for %s",
				document, specOwnersHeading, name, owner, relation.owner))
		}
	}
	for name := range listed {
		if _, ok := registry.relations[name]; !ok {
			problems = append(problems, fmt.Sprintf("%s: %q lists %s, which %s does not", document, specOwnersHeading, name, ownershipRegistryFile))
		}
	}
	slices.Sort(problems)
	return problems
}

// testRegistry is a registry for the rule tests: runstore owns the Run and
// Stage tables, projectstore owns projects.
const testRegistry = `
# comment
[relations]
runs        internal/runstore       exec
stages      internal/runstore       exec
projects    internal/projectstore   projects

[reads]
internal/telemetry   runs       resource history scopes allocations
    to the owner's Runs
internal/telemetry   projects   no SQL reads this any more

[known]
internal/scheduler   write  stages    V370-001  admission updates the Stage
internal/scheduler   write  projects  V370-002  only reads remain
internal/gone        read   runs      V370-002  the package is gone
internal/memory      read   runs      V100-001  the task is finished
`

// testSources is SQL of four packages in the shape repository code has.
var testSources = map[string]map[string]string{
	"internal/runstore": {"internal/runstore/store.go": "package runstore\n\n" +
		"const updateRun = `UPDATE runs SET state = $2 WHERE run_id = $1`\n"},
	"internal/scheduler": {"internal/scheduler/admission.go": "package scheduler\n\n" +
		"func admit() {\n\t_ = `UPDATE stages SET admitted_at = now() WHERE stage_id = $1`\n" +
		"\t_ = `SELECT kind FROM projects WHERE project_id = $1`\n}\n"},
	"internal/telemetry": {"internal/telemetry/history.go": "package telemetry\n\n" +
		"const history = `SELECT run.owner_id FROM runs AS run`\n"},
	"internal/memory": {"internal/memory/fence.go": "package memory\n\n" +
		"const fence = `SELECT state FROM runs WHERE run_id = $1`\n"},
}

func scanTestSources(t *testing.T, sources map[string]map[string]string, relations map[string]bool) []sqlAccess {
	t.Helper()
	var accesses []sqlAccess
	for _, pkg := range slices.Sorted(maps.Keys(sources)) {
		files := map[string][]byte{}
		for name, source := range sources[pkg] {
			files[name] = []byte(source)
		}
		found, err := scanPackageSQL(pkg, files, relations)
		if err != nil {
			t.Fatal(err)
		}
		accesses = append(accesses, found...)
	}
	return accesses
}

// TestTableOwnershipCheckRejectsAnUnlistedForeignUpdate is acceptance A1 of
// the ownership check: an UPDATE of another package's table fails until the
// registry lists it as a known access, and that entry fails once the
// statement is gone.
func TestTableOwnershipCheckRejectsAnUnlistedForeignUpdate(t *testing.T) {
	const relations = "[relations]\nruns internal/runstore exec\n"
	update := map[string]map[string]string{
		"internal/runstore": {"internal/runstore/store.go": "package runstore\n\nconst q = `SELECT state FROM runs`\n"},
		"internal/gatewayrecovery": {"internal/gatewayrecovery/waiting.go": "package gatewayrecovery\n\n" +
			"func wait() {\n\t_ = `UPDATE runs SET state='waiting' WHERE run_id=$1`\n}\n"},
	}
	open := func(string) string { return "" }

	registry, problems := parseOwnershipRegistry(relations)
	if len(problems) > 0 {
		t.Fatal(problems)
	}
	got := checkTableOwnership(registry, scanTestSources(t, update, registry.relationNames()), open)
	want := "internal/gatewayrecovery/waiting.go:4: internal/gatewayrecovery writes runs, which internal/runstore owns"
	if len(got) != 1 || !strings.HasPrefix(got[0], want) {
		t.Fatalf("problems = %q, want one starting %q", got, want)
	}

	registry, problems = parseOwnershipRegistry(relations +
		"[known]\ninternal/gatewayrecovery write runs V370-001 waiting moves the Run\n")
	if len(problems) > 0 {
		t.Fatal(problems)
	}
	if got := checkTableOwnership(registry, scanTestSources(t, update, registry.relationNames()), open); len(got) > 0 {
		t.Fatalf("problems with the known access = %q, want none", got)
	}

	delete(update, "internal/gatewayrecovery")
	got = checkTableOwnership(registry, scanTestSources(t, update, registry.relationNames()), open)
	want = ownershipRegistryFile + ":4: [known] internal/gatewayrecovery runs matches no access; delete the entry"
	if len(got) != 1 || got[0] != want {
		t.Fatalf("problems without the UPDATE = %q, want %q", got, want)
	}
}

// TestTableOwnershipCheckReportsEachRule checks every rule on a synthetic
// registry, independently of the repository.
func TestTableOwnershipCheckReportsEachRule(t *testing.T) {
	registry, problems := parseOwnershipRegistry(testRegistry)
	if len(problems) > 0 {
		t.Fatal(problems)
	}
	if got := registry.reads[accessPair{"internal/telemetry", "runs"}].reason; got != "resource history scopes allocations to the owner's Runs" {
		t.Fatalf("continued reason = %q", got)
	}
	accesses := scanTestSources(t, testSources, registry.relationNames())
	finished := func(task string) string {
		if task == "V100-001" {
			return "is finished"
		}
		return ""
	}
	got := checkTableOwnership(registry, accesses, finished)
	want := []string{
		ownershipRegistryFile + ":11: [reads] internal/telemetry projects matches no read; delete the entry",
		ownershipRegistryFile + ":15: [known] internal/scheduler no longer writes projects; delete the entry, or make it a read if reads remain",
		ownershipRegistryFile + ":16: [known] internal/gone runs matches no access; delete the entry",
		ownershipRegistryFile + ":17: [known] internal/memory runs names V100-001, which is finished",
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("problems =\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}

	// A read is not covered by a granted read of another relation, and a
	// write is not covered by a granted read.
	accesses = append(accesses,
		sqlAccess{pkg: "internal/telemetry", relation: "stages", position: "internal/telemetry/a.go:1"},
		sqlAccess{pkg: "internal/telemetry", relation: "runs", write: true, position: "internal/telemetry/a.go:2"},
	)
	got = checkTableOwnership(registry, accesses, finished)
	for _, want := range []string{
		"internal/telemetry/a.go:1: internal/telemetry reads stages, which internal/runstore owns",
		"internal/telemetry/a.go:2: internal/telemetry writes runs, which internal/runstore owns",
	} {
		if !slices.ContainsFunc(got, func(problem string) bool { return strings.HasPrefix(problem, want) }) {
			t.Errorf("no problem starts %q; got:\n%s", want, strings.Join(got, "\n"))
		}
	}
}

func TestTableOwnershipRegistryRejectsMalformedEntries(t *testing.T) {
	_, problems := parseOwnershipRegistry(`
    orphan continuation
[relations]
runs internal/runstore exec
runs internal/runstore exec
stages internal/runstore
Bad internal/runstore exec

[reads]
internal/runstore runs the owner reads its own relation
internal/telemetry missing no such relation
internal/telemetry runs granted twice
internal/telemetry runs granted twice

[known]
internal/telemetry write runs V370-001 also granted
internal/scheduler update runs V370-001 not an access kind
internal/scheduler write runs no-task missing task
internal/scheduler read runs V370-001 first
internal/scheduler write runs V370-001 second

[other]
`)
	want := []string{
		":2: an indented line must continue the reason of the entry above it",
		":5: relation runs is already listed at line 4",
		":6: a relation line is: <relation> <owner package> <ERD group or ->",
		`:7: "Bad internal/runstore exec" is not a relation name with an owner package path`,
		":13: the read of runs by internal/telemetry is already listed at line 12",
		":17: a known access line is: <package> read|write <relation> <task ID> <reason>",
		":18: a known access line is: <package> read|write <relation> <task ID> <reason>",
		":20: the access of runs by internal/scheduler is already listed at line 19; a write also covers reads",
		":22: unknown section [other]; use [relations], [reads] or [known]",
		":10: internal/runstore owns runs and needs no entry",
		":11: missing is not listed under [relations]",
		":12: the read of runs by internal/telemetry is also a known access at line 16; keep one entry",
	}
	for index := range want {
		want[index] = ownershipRegistryFile + want[index]
	}
	if strings.Join(problems, "\n") != strings.Join(want, "\n") {
		t.Fatalf("problems =\n%s\nwant\n%s", strings.Join(problems, "\n"), strings.Join(want, "\n"))
	}
}

func TestTableOwnershipDocumentChecksReportDrift(t *testing.T) {
	registry, problems := parseOwnershipRegistry(`
[relations]
runs        internal/runstore       exec
stages      internal/runstore       audits
projects    internal/projectstore   projects
ledger      internal/persistence/postgres  -
events      internal/runstore       nowhere
`)
	if len(problems) > 0 {
		t.Fatal(problems)
	}
	got := erdGroupProblems(registry, map[string]bool{"exec": true, "audits": true, "projects": true},
		map[string]string{"runs": "exec", "stages": "exec", "ledger": "exec", "extra": "exec"})
	want := []string{
		ownershipRegistryFile + ":4: scripts/erd/meta.json draws stages in group exec, the registry in audits",
		ownershipRegistryFile + ":5: scripts/erd/meta.json does not describe projects; add it to group projects",
		ownershipRegistryFile + ":6: scripts/erd/meta.json draws ledger in group exec; give it that group here",
		ownershipRegistryFile + ":7: nowhere is not a group of scripts/erd/meta.json",
		"scripts/erd/meta.json describes extra, which " + ownershipRegistryFile + " does not list",
	}
	slices.Sort(want)
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("ERD problems =\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}

	spec := "# Persistence\n\n" + specOwnersHeading + "\n\nIntro.\n\n| Owner package | Tables and views |\n| --- | --- |\n" +
		"| `internal/runstore` | `runs`, `stages`, `projects` |\n" +
		"| `internal/persistence/postgres` | `ledger` (the migration ledger), `gone` |\n\n## Requirements\n"
	got = specOwnerProblems(registry, spec)
	want = []string{
		`docs/spec/platform/persistence.md: "### Table owners" does not list events for internal/runstore`,
		`docs/spec/platform/persistence.md: "### Table owners" lists gone, which ` + ownershipRegistryFile + " does not",
		`docs/spec/platform/persistence.md: "### Table owners" lists projects for internal/runstore, the registry for internal/projectstore`,
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("specification problems =\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
}
