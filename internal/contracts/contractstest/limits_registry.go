package contractstest

import (
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// LimitsEntry is one row of the limits registry in
// docs/spec/platform/limits.md.
type LimitsEntry struct {
	ID         string
	Value      string
	AppliesTo  string
	EnforcedBy string
	Owner      string
}

// LimitsRegistry is the limits registry by entry ID.
type LimitsRegistry map[string]LimitsEntry

// ReadLimitsRegistry returns every row of the limits registry. A row without
// its five cells or a repeated ID fails the test.
func ReadLimitsRegistry(t testing.TB) LimitsRegistry {
	t.Helper()
	registry := LimitsRegistry{}
	for line := range strings.SplitSeq(string(ReadFile(t, "docs", "spec", "platform", "limits.md")), "\n") {
		if !strings.HasPrefix(line, "| `LIM-") {
			continue
		}
		cells := tableCells(line)
		if len(cells) != 5 {
			t.Fatalf("limits.md row %q has %d cells, want 5", line, len(cells))
		}
		id := strings.Trim(cells[0], "`")
		if _, repeated := registry[id]; repeated {
			t.Fatalf("limits.md repeats %s", id)
		}
		registry[id] = LimitsEntry{ID: id, Value: cells[1], AppliesTo: cells[2], EnforcedBy: cells[3], Owner: cells[4]}
	}
	if len(registry) == 0 {
		t.Fatal("limits.md has no registry rows")
	}
	return registry
}

// Naming returns the one entry whose Enforced by cell names the PostgreSQL
// constraint as a code span.
func (r LimitsRegistry) Naming(t testing.TB, constraint string) LimitsEntry {
	t.Helper()
	var found []LimitsEntry
	for _, entry := range r {
		if r.Names(entry.ID, constraint) {
			found = append(found, entry)
		}
	}
	if len(found) != 1 {
		t.Fatalf("%d limits.md rows name %s as an enforcement point, want one", len(found), constraint)
	}
	return found[0]
}

// Names reports whether the Enforced by cell of entry id names the PostgreSQL
// constraint as a code span.
func (r LimitsRegistry) Names(id, constraint string) bool {
	return strings.Contains(r[id].EnforcedBy, "`"+constraint+"`")
}

var bytesInValue = regexp.MustCompile(`\(([0-9,]+) bytes\)`)

// Bytes returns the one exact byte count the value cell states in the form
// "(2,097,152 bytes)".
func (e LimitsEntry) Bytes(t testing.TB) int {
	t.Helper()
	matches := bytesInValue.FindAllStringSubmatch(e.Value, -1)
	if len(matches) != 1 {
		t.Fatalf("%s value %q states %d exact byte counts, want one", e.ID, e.Value, len(matches))
	}
	value, err := strconv.Atoi(strings.ReplaceAll(matches[0][1], ",", ""))
	if err != nil {
		t.Fatalf("%s value %q: %v", e.ID, e.Value, err)
	}
	return value
}

// tableCells splits one Markdown table row into its trimmed cells; a pipe
// inside a code span does not end a cell.
func tableCells(line string) []string {
	line = strings.TrimSpace(line)
	line = strings.TrimSuffix(strings.TrimPrefix(line, "|"), "|")
	var cells []string
	var cell strings.Builder
	code := false
	for _, character := range line {
		switch {
		case character == '`':
			code = !code
		case character == '|' && !code:
			cells = append(cells, strings.TrimSpace(cell.String()))
			cell.Reset()
			continue
		}
		cell.WriteRune(character)
	}
	return append(cells, strings.TrimSpace(cell.String()))
}
