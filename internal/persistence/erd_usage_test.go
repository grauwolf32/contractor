package persistence

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// The "where in code" list of each table on the ERD page is the set of
// packages whose SQL the table ownership scan sees reading or writing the
// table, named by package directory as the registry names them. The page is
// built with a database (make docs-erd), so scripts/erd/build_erd.py takes the
// lists from TestERDUsageMatchesTheSQLScan, and make lint runs that test to
// fail while the page lags the code; neither needs a database. A package that
// reaches a table only through a migration function or trigger is not listed:
// telling that needs the migrated function bodies.

// erdPageFile is the generated ERD page.
const erdPageFile = "docs/spec/database-erd.html"

// erdUsageOutput is the environment variable that build_erd.py sets to a file
// path: TestERDUsageMatchesTheSQLScan then writes the usage lists there as
// JSON instead of comparing them with the page.
const erdUsageOutput = "CONTRACTOR_ERD_USAGE_OUT"

// erdPageData is the page's data element, as build_erd.py writes it.
var erdPageData = regexp.MustCompile(`(?s)<script type="application/json" id="data">(.*?)</script>`)

// tableUsage returns, per relation, the sorted directories of the packages
// whose SQL reads or writes it.
func tableUsage(accesses []sqlAccess) map[string][]string {
	usage := map[string][]string{}
	for _, access := range accesses {
		if !slices.Contains(usage[access.relation], access.pkg) {
			usage[access.relation] = append(usage[access.relation], access.pkg)
		}
	}
	for _, packages := range usage {
		slices.Sort(packages)
	}
	return usage
}

// erdPageUsage returns the usage list of each table the page draws.
func erdPageUsage(page string) (map[string][]string, error) {
	data := erdPageData.FindStringSubmatch(page)
	if data == nil {
		return nil, fmt.Errorf("%s has no schema data", erdPageFile)
	}
	var schema struct {
		Tables []struct {
			Name  string   `json:"name"`
			Usage []string `json:"usage"`
		} `json:"tables"`
	}
	if err := json.Unmarshal([]byte(data[1]), &schema); err != nil {
		return nil, fmt.Errorf("%s: unreadable schema data: %w", erdPageFile, err)
	}
	usage := map[string][]string{}
	for _, table := range schema.Tables {
		if _, ok := usage[table.Name]; ok || table.Usage == nil {
			return nil, fmt.Errorf("%s draws table %q twice or without a usage list", erdPageFile, table.Name)
		}
		usage[table.Name] = table.Usage
	}
	return usage, nil
}

// erdUsageProblems returns one message per table whose usage list on the page
// differs from the scan's, sorted. Tables the page does not draw, such as
// views, are left to scripts/check_database_erd.py.
func erdUsageProblems(page, scanned map[string][]string) []string {
	var problems []string
	for table, listed := range page {
		found := scanned[table]
		if found == nil {
			found = []string{}
		}
		if !slices.Equal(listed, found) {
			problems = append(problems, fmt.Sprintf("%s lists %v as the packages using %s; the SQL scan finds %v",
				erdPageFile, listed, table, found))
		}
	}
	slices.Sort(problems)
	return problems
}

// TestERDUsageMatchesTheSQLScan fails while a "where in code" list of the ERD
// page differs from the packages whose SQL the ownership scan sees using the
// table. With CONTRACTOR_ERD_USAGE_OUT set it writes the scan's lists to that
// file instead, for build_erd.py.
func TestERDUsageMatchesTheSQLScan(t *testing.T) {
	root := moduleRootDir(t)
	registry := loadOwnershipRegistry(t, root)
	scanned, files, err := scanModuleSQL(root, registry.relationNames(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if files < 100 {
		t.Fatalf("scanned only %d Go files; the repository walk is broken", files)
	}
	usage := tableUsage(scanned.accesses)
	if output := os.Getenv(erdUsageOutput); output != "" {
		encoded, err := json.MarshalIndent(usage, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(output, encoded, 0o600); err != nil {
			t.Fatal(err)
		}
		return
	}
	page, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(erdPageFile)))
	if err != nil {
		t.Fatal(err)
	}
	listed, err := erdPageUsage(string(page))
	if err != nil {
		t.Fatal(err)
	}
	if problems := erdUsageProblems(listed, usage); len(problems) > 0 {
		t.Errorf("regenerate the page with make docs-erd (scripts/erd/README.md):\n%s", strings.Join(problems, "\n"))
	}
}

// TestERDUsageCheckReportsStaleLists is acceptance A1 of V372-010 without the
// repository: a package that starts or stops using a table, or a page that
// names a package by another path, is reported, and an equal page is not.
func TestERDUsageCheckReportsStaleLists(t *testing.T) {
	scanned := tableUsage([]sqlAccess{
		{pkg: "internal/runstore", relation: "workflow_runs", write: true},
		{pkg: "internal/telemetry", relation: "workflow_runs"},
		{pkg: "internal/runstore", relation: "workflow_runs"},
		{pkg: "internal/auditstore/internal/pgstore", relation: "audits", write: true},
		{pkg: "internal/runstore", relation: "active_runs_view"},
	})
	page := func(usage string) string {
		return `<html><script type="application/json" id="data">{"schemaVersion":"000002","tables":[` + usage +
			`]}</script></html>`
	}
	fresh := page(`{"name":"workflow_runs","usage":["internal/runstore","internal/telemetry"]},` +
		`{"name":"audits","usage":["internal/auditstore/internal/pgstore"]},{"name":"unused","usage":[]}`)
	listed, err := erdPageUsage(fresh)
	if err != nil {
		t.Fatal(err)
	}
	if problems := erdUsageProblems(listed, scanned); len(problems) > 0 {
		t.Fatalf("a fresh page is reported: %v", problems)
	}
	stale := page(`{"name":"workflow_runs","usage":["internal/cli","internal/runstore"]},` +
		`{"name":"audits","usage":["auditstore"]},{"name":"unused","usage":["internal/telemetry"]}`)
	listed, err = erdPageUsage(stale)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		erdPageFile + " lists [auditstore] as the packages using audits; the SQL scan finds [internal/auditstore/internal/pgstore]",
		erdPageFile + " lists [internal/cli internal/runstore] as the packages using workflow_runs; the SQL scan finds [internal/runstore internal/telemetry]",
		erdPageFile + " lists [internal/telemetry] as the packages using unused; the SQL scan finds []",
	}
	if problems := erdUsageProblems(listed, scanned); !slices.Equal(problems, want) {
		t.Fatalf("problems =\n%s\nwant\n%s", strings.Join(problems, "\n"), strings.Join(want, "\n"))
	}
	for _, broken := range []string{
		"<html></html>",
		page(`{"name":"audits"}`),
		page(`{"name":"audits","usage":[]},{"name":"audits","usage":[]}`),
		page(`{"name":"audits","usage":"internal/runstore"}`),
	} {
		if _, err := erdPageUsage(broken); err == nil {
			t.Errorf("erdPageUsage(%q) accepted a broken page", broken)
		}
	}
}
