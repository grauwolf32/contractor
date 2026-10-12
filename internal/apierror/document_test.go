package apierror

import (
	"flag"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// update rewrites the generated table of docs/spec/platform/errors.md:
// go test ./internal/apierror -run TestRegistryTableIsCurrent -update
var update = flag.Bool("update", false, "rewrite the generated public error code table of errors.md")

const (
	registryDocument = specPlatform + "/errors.md"
	tableBegin       = "<!-- BEGIN GENERATED PUBLIC ERROR CODES: edit internal/apierror and run make docs-error-registry -->"
	tableEnd         = "<!-- END GENERATED PUBLIC ERROR CODES -->"
	assembledMarker  = "<!-- registry-table -->"
)

// renderTable renders the registry as the Markdown table of errors.md, one
// row per code in code order.
func renderTable() string {
	rows := slices.Clone(registry)
	slices.SortFunc(rows, func(left, right *entry) int { return strings.Compare(left.code, right.code) })
	var table strings.Builder
	table.WriteString("| Code | HTTP status | Retryable | Meaning | Owner |\n| --- | --- | --- | --- | --- |\n")
	for _, row := range rows {
		status := strconv.Itoa(row.status)
		switch {
		case row.status == 0:
			status = "— (WebSocket error frame)"
		case row.frame:
			status += "; also a WebSocket error frame"
		}
		retryable := "no"
		if row.retryable {
			retryable = "yes"
		}
		table.WriteString("| `" + row.code + "` | " + status + " | " + retryable + " | " + row.meaning +
			" | " + row.owner + " |\n")
	}
	return table.String()
}

// TestRegistryTableIsCurrent keeps the generated table of errors.md equal to
// the registry, and the registry one row per code: no code of the generated
// table has another row in the assembled table, and the assembled table has
// one row per code.
func TestRegistryTableIsCurrent(t *testing.T) {
	path := filepath.Clean(registryDocument)
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(content)
	begin, end := strings.Index(text, tableBegin), strings.Index(text, tableEnd)
	if begin < 0 || end < begin {
		t.Fatalf("%s has no generated public error code table between its markers", path)
	}
	current := text[begin+len(tableBegin)+1 : end]
	if want := renderTable(); current != want {
		if !*update {
			t.Fatalf("%s is stale: run go test ./internal/apierror -run TestRegistryTableIsCurrent -update", path)
		}
		text = text[:begin+len(tableBegin)+1] + want + text[end:]
		if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	assembled := strings.Index(text, assembledMarker)
	if assembled < 0 {
		t.Fatalf("%s has no assembled registry table", path)
	}
	firstCell := regexp.MustCompile("(?m)^\\| `([^`]+)` \\|")
	seen := map[string]bool{}
	for _, match := range firstCell.FindAllStringSubmatch(text[assembled:], -1) {
		code := match[1]
		if _, generated := Lookup(code); generated {
			t.Errorf("%s: %s has a row in the generated and in the assembled table", path, code)
		}
		if seen[code] {
			t.Errorf("%s: the assembled table has two rows for %s", path, code)
		}
		seen[code] = true
	}
}
