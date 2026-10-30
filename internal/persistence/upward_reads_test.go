package persistence

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// This file holds the reads of the table ownership registry to the layer
// stack (PER-R23): a package reads a relation whose owner sits in a higher
// layer only where its entry names the open task that removes the read or an
// accepted upward read of persistence.md. The import check of
// internal/archtest cannot see SQL, so this check reads that package's layer
// table as data rather than keeping a second list.

// layerTableFile holds the Server's layer table, the composite literal of the
// variable layers.
const layerTableFile = "internal/archtest/layers_test.go"

// layerTable is the layer of each exact package entry and "dir/..." pattern
// of the layer table, by index from the bottom.
type layerTable struct {
	names    []string
	exact    map[string]int
	patterns map[string]int
}

// parseLayerTable reads the layers variable of the layer table's source: a
// slice of composite literals with a name and a packages list of string
// literals.
func parseLayerTable(source []byte) (layerTable, error) {
	file, err := parser.ParseFile(token.NewFileSet(), layerTableFile, source, parser.SkipObjectResolution)
	if err != nil {
		return layerTable{}, err
	}
	table := layerTable{exact: map[string]int{}, patterns: map[string]int{}}
	str := func(expr ast.Expr) (string, error) {
		literal, ok := expr.(*ast.BasicLit)
		if !ok || literal.Kind != token.STRING {
			return "", fmt.Errorf("%s: layers holds a value that is not a string literal", layerTableFile)
		}
		return strconv.Unquote(literal.Value)
	}
	for _, decl := range file.Decls {
		general, ok := decl.(*ast.GenDecl)
		if !ok || general.Tok != token.VAR {
			continue
		}
		for _, spec := range general.Specs {
			value := spec.(*ast.ValueSpec)
			if len(value.Names) != 1 || value.Names[0].Name != "layers" || len(value.Values) != 1 {
				continue
			}
			list, ok := value.Values[0].(*ast.CompositeLit)
			if !ok {
				return layerTable{}, fmt.Errorf("%s: layers is not a composite literal", layerTableFile)
			}
			for _, element := range list.Elts {
				layer, ok := element.(*ast.CompositeLit)
				if !ok {
					return layerTable{}, fmt.Errorf("%s: a layer is not a composite literal", layerTableFile)
				}
				index := len(table.names)
				table.names = append(table.names, "")
				for _, field := range layer.Elts {
					pair, ok := field.(*ast.KeyValueExpr)
					if !ok {
						return layerTable{}, fmt.Errorf("%s: a layer field has no name", layerTableFile)
					}
					key, _ := pair.Key.(*ast.Ident)
					switch {
					case key == nil:
						return layerTable{}, fmt.Errorf("%s: a layer field has no name", layerTableFile)
					case key.Name == "name":
						if table.names[index], err = str(pair.Value); err != nil {
							return layerTable{}, err
						}
					case key.Name == "packages":
						packages, ok := pair.Value.(*ast.CompositeLit)
						if !ok {
							return layerTable{}, fmt.Errorf("%s: packages of a layer is not a composite literal", layerTableFile)
						}
						for _, entry := range packages.Elts {
							pkg, err := str(entry)
							if err != nil {
								return layerTable{}, err
							}
							if prefix, ok := strings.CutSuffix(pkg, "/..."); ok {
								table.patterns[prefix] = index
							} else {
								table.exact[pkg] = index
							}
						}
					}
				}
				if table.names[index] == "" {
					return layerTable{}, fmt.Errorf("%s: layer %d has no name", layerTableFile, index)
				}
			}
		}
	}
	if len(table.names) == 0 {
		return layerTable{}, fmt.Errorf("%s defines no layers", layerTableFile)
	}
	return table, nil
}

// layerOf returns the layer of a package as internal/archtest assigns it: an
// exact entry, else the longest "dir/..." pattern that covers it.
func (l layerTable) layerOf(pkg string) (int, bool) {
	if index, ok := l.exact[pkg]; ok {
		return index, true
	}
	best, found := "", false
	for prefix := range l.patterns {
		if (pkg == prefix || strings.HasPrefix(pkg, prefix+"/")) && len(prefix) >= len(best) {
			best, found = prefix, true
		}
	}
	return l.patterns[best], found
}

// acceptedReadsHeading is the persistence.md heading of the accepted upward
// reads.
const acceptedReadsHeading = "### Accepted upward reads"

var (
	acceptedReadRow = regexp.MustCompile("^\\| `(PER-UR-[0-9]+)` \\| ([^|]*) \\| ([^|]*) \\|")
	acceptedReadID  = regexp.MustCompile(`\bPER-UR-[0-9]+\b`)
	// reasonTask is a task ID within a registry reason.
	reasonTask = regexp.MustCompile(`\b(?:MVP|V[0-9]+)-[0-9]{3}[A-Z]?\b`)
)

// acceptedRead is one record of the accepted upward reads: the reader and the
// relations it may read.
type acceptedRead struct {
	reader    string
	relations []string
}

// parseAcceptedReads reads the records of the "Accepted upward reads" table
// of persistence.md: the record ID, the reader as the first code span of its
// cell, and the relations as the code spans of theirs.
func parseAcceptedReads(spec string) (map[string]acceptedRead, []string) {
	const document = "docs/spec/platform/persistence.md"
	lines := strings.Split(spec, "\n")
	start := slices.Index(lines, acceptedReadsHeading)
	if start < 0 {
		return nil, []string{fmt.Sprintf("%s has no %q section", document, acceptedReadsHeading)}
	}
	records := map[string]acceptedRead{}
	var problems []string
	for _, line := range lines[start+1:] {
		if strings.HasPrefix(line, "#") {
			break
		}
		row := acceptedReadRow.FindStringSubmatch(line)
		if row == nil {
			continue
		}
		readers := specCodeSpans.FindStringSubmatch(row[2])
		var relations []string
		for _, span := range specCodeSpans.FindAllStringSubmatch(row[3], -1) {
			relations = append(relations, span[1])
		}
		if _, ok := records[row[1]]; ok || readers == nil || len(relations) == 0 {
			problems = append(problems, fmt.Sprintf("%s: %s is listed twice or names no reader or relation", document, row[1]))
			continue
		}
		records[row[1]] = acceptedRead{reader: readers[1], relations: relations}
	}
	return records, problems
}

// checkUpwardReads returns one message per [reads] or [routines] read whose
// package sits below the relation's owner and whose reason names neither an
// open task nor an accepted read that lists it, per package the layer table
// does not place, and per record that lists a read no such entry names,
// sorted. taskState is as for checkTableOwnership.
func checkUpwardReads(registry ownershipRegistry, layers layerTable, accepted map[string]acceptedRead, taskState func(string) string) []string {
	type read struct {
		pkg, relation, reason, entry string
		line                         int
	}
	var reads []read
	for pair, entry := range registry.reads {
		reads = append(reads, read{pkg: pair.pkg, relation: pair.relation, reason: entry.reason, line: entry.line,
			entry: fmt.Sprintf("[reads] %s %s", pair.pkg, pair.relation)})
	}
	for key, entry := range registry.routines {
		if !entry.write {
			reads = append(reads, read{pkg: key.pkg, relation: key.relation, reason: entry.reason, line: entry.line,
				entry: fmt.Sprintf("[routines] %s %s read %s", key.pkg, key.function, key.relation)})
		}
	}
	var problems []string
	named := map[string]bool{} // record + reader + relation that an upward entry names
	for _, read := range reads {
		owner := registry.relations[read.relation].owner
		reader, readerPlaced := layers.layerOf(read.pkg)
		owned, ownerPlaced := layers.layerOf(owner)
		switch {
		case !readerPlaced || !ownerPlaced:
			problems = append(problems, fmt.Sprintf("%s:%d: %s: %s or %s has no layer in %s",
				ownershipRegistryFile, read.line, read.entry, read.pkg, owner, layerTableFile))
			continue
		case reader >= owned:
			continue
		}
		satisfied := slices.ContainsFunc(reasonTask.FindAllString(read.reason, -1), func(task string) bool {
			return taskState(task) == ""
		})
		for _, id := range acceptedReadID.FindAllString(read.reason, -1) {
			record, ok := accepted[id]
			if ok && record.reader == read.pkg && slices.Contains(record.relations, read.relation) {
				satisfied = true
				named[id+" "+read.pkg+" "+read.relation] = true
			}
		}
		if !satisfied {
			problems = append(problems, fmt.Sprintf(
				"%s:%d: %s reads up the layer stack: %s (%s) reads a relation of %s (%s); remove the read, or name in its reason the open task that removes it or an accepted upward read (%s in docs/spec/platform/persistence.md) that lists it",
				ownershipRegistryFile, read.line, read.entry, read.pkg, layers.names[reader], owner, layers.names[owned],
				strings.TrimPrefix(acceptedReadsHeading, "### ")))
		}
	}
	for id, record := range accepted {
		for _, relation := range record.relations {
			if !named[id+" "+record.reader+" "+relation] {
				problems = append(problems, fmt.Sprintf(
					"docs/spec/platform/persistence.md: %s lists %s reading %s, which no upward registry entry names with %s; delete it from the record",
					id, record.reader, relation, id))
			}
		}
	}
	slices.Sort(problems)
	return problems
}

// TestUpwardReadsNameTheirAcceptance is the layer half of PER-R23: every read
// of a relation whose owner sits higher in the layer stack names its open
// task or accepted upward read, and every accepted upward read is still named.
func TestUpwardReadsNameTheirAcceptance(t *testing.T) {
	root := moduleRootDir(t)
	registry := loadOwnershipRegistry(t, root)
	source, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(layerTableFile)))
	if err != nil {
		t.Fatal(err)
	}
	layers, err := parseLayerTable(source)
	if err != nil {
		t.Fatal(err)
	}
	spec, err := os.ReadFile(filepath.Join(root, "docs", "spec", "platform", "persistence.md"))
	if err != nil {
		t.Fatal(err)
	}
	accepted, problems := parseAcceptedReads(string(spec))
	if len(accepted) == 0 {
		problems = append(problems, "docs/spec/platform/persistence.md lists no accepted upward read")
	}
	problems = append(problems, checkUpwardReads(registry, layers, accepted, openTaskIn(root))...)
	for _, problem := range problems {
		t.Error(problem)
	}
}

// TestUpwardReadCheckReportsEachRule is acceptance A2 of V372-008: an
// ordinary grant of an upward read fails, an open task or an accepted read
// that lists it passes, a finished task does not, and a record that no entry
// names fails.
func TestUpwardReadCheckReportsEachRule(t *testing.T) {
	layers, err := parseLayerTable([]byte(`package archtest

var layers = []layerSpec{
	{name: "platform", role: "below", packages: []string{"internal/telemetry", "internal/artifacts"}},
	{name: "execution", packages: []string{"internal/runstore", "internal/exec/..."}},
	{name: "product", packages: []string{"internal/projectstore", "internal/exec/product"}},
}
`))
	if err != nil {
		t.Fatal(err)
	}
	if index, ok := layers.layerOf("internal/exec/product"); !ok || index != 2 {
		t.Fatalf("an exact entry does not win over a pattern: %d, %t", index, ok)
	}
	if index, ok := layers.layerOf("internal/exec/sub/pkg"); !ok || index != 1 {
		t.Fatalf("a pattern does not cover a package below it: %d, %t", index, ok)
	}
	registry, problems := parseOwnershipRegistry(`
[relations]
runs        internal/runstore       exec
projects    internal/projectstore   projects
members     internal/exec/product   exec

[reads]
internal/telemetry  runs      history reads Runs
internal/artifacts  runs      forks check the Run. Accepted upward read PER-UR-1.
internal/runstore   projects  the queue shows Projects; V372-900 moves it
internal/exec/one   projects  finished task V372-901
internal/runstore   members   sideways? no: product is above execution. Accepted upward read PER-UR-1.
internal/exec/one   runs      the same layer

[routines]
internal/telemetry  run_members  read  projects  retention keeps metrics. Accepted upward read PER-UR-2.
internal/unplaced   run_members  read  runs      no layer
`)
	if len(problems) > 0 {
		t.Fatal(problems)
	}
	accepted, problems := parseAcceptedReads(acceptedReadsHeading + `

| Record | Reader | Relations | Owners |
| --- | --- | --- | --- |
| ` + "`PER-UR-1` | `internal/artifacts` (platform) | `runs` | `internal/runstore` |" + `
| ` + "`PER-UR-2` | `internal/telemetry` (platform) | `projects`, `runs` | `internal/projectstore` |" + `

### Next
| ` + "`PER-UR-9` | `internal/other` | `runs` | ignored |")
	if len(problems) > 0 || len(accepted) != 2 {
		t.Fatalf("records = %+v, problems %v", accepted, problems)
	}
	taskState := func(task string) string {
		if task == "V372-901" {
			return "is finished"
		}
		return ""
	}
	got := checkUpwardReads(registry, layers, accepted, taskState)
	want := []string{
		"docs/spec/platform/persistence.md: PER-UR-2 lists internal/telemetry reading runs, which no upward registry entry names with PER-UR-2; delete it from the record",
		ownershipRegistryFile + ":11: [reads] internal/exec/one projects reads up the layer stack: internal/exec/one (execution) reads a relation of internal/projectstore (product)",
		ownershipRegistryFile + ":12: [reads] internal/runstore members reads up the layer stack: internal/runstore (execution) reads a relation of internal/exec/product (product)",
		ownershipRegistryFile + ":17: [routines] internal/unplaced run_members read runs: internal/unplaced or internal/runstore has no layer in " + layerTableFile,
		ownershipRegistryFile + ":8: [reads] internal/telemetry runs reads up the layer stack: internal/telemetry (platform) reads a relation of internal/runstore (execution)",
	}
	matches := len(got) == len(want)
	for index := 0; matches && index < len(want); index++ {
		matches = strings.HasPrefix(got[index], want[index])
	}
	if !matches {
		t.Fatalf("problems =\n%s\nwant prefixes\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
}
