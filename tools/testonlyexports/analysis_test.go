package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

const fixtureDir = "testdata/fixture"

// fixtureFindings are the declarations of testdata/fixture that the rules
// report; every other exported declaration there is used in production,
// generated, or test code itself.
var fixtureFindings = []string{
	"cmd/app.Helper",          // used only by the command's test
	"core.Asserted",           // named only by a compile-time check and tests
	"core.Asserted.Do",        // an interface method nothing calls
	"core.Box.Peek",           // a generic method used only by tests
	"core.CodeB",              // an enumeration member only tests compare
	"core.ErrOnlyTests",       // a variable only tests compare
	"core.Hidden",             // named only by its own method and tests
	"core.Hidden.Self",        // a method only tests call
	"core.Impl.Do",            // implements only Asserted, which production does not use
	"core.Reader.Close",       // an interface method only tests call
	"core.Recursive",          // its own recursive call is not a use
	"core.Store.TestHook",     // a method only tests call
	"core.TaggedOnly",         // used only by a test behind the fixture build tag
	"core.TestOnly",           // used only by a test of its package
	"core.Unused",             // used by nothing
	"core.UsedByHelper",       // used only by the test helper package coretest
	"core.UsedByProcessTests", // used only by tests/
	"core.XTestOnly",          // used only by the external test package
}

func analyzeFixture(t *testing.T) []Finding {
	t.Helper()
	result, err := Analyze(Options{Dir: fixtureDir, Tags: []string{"fixture"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.HelperImports) != 0 {
		t.Fatalf("helper imports = %+v", result.HelperImports)
	}
	return result.Findings
}

func TestAnalyzeReportsEachExportWithoutProductionReference(t *testing.T) {
	findings := analyzeFixture(t)
	keys := make([]string, len(findings))
	for index, finding := range findings {
		keys[index] = finding.Key
	}
	want := slices.Sorted(slices.Values(fixtureFindings))
	if !slices.Equal(keys, want) {
		t.Fatalf("findings = %q\nwant       %q", keys, want)
	}
	byKey := map[string]Finding{}
	for _, finding := range findings {
		byKey[finding.Key] = finding
	}
	for key, want := range map[string]struct {
		kind  string
		refs  int
		files []string
	}{
		"core.TestOnly":           {"func", 1, []string{"core/core_test.go"}},
		"core.Unused":             {"func", 0, []string{}},
		"core.TaggedOnly":         {"func", 1, []string{"core/tagged_test.go"}},
		"core.XTestOnly":          {"func", 1, []string{"core/external_test.go"}},
		"core.UsedByHelper":       {"func", 1, []string{"coretest/coretest.go"}},
		"core.UsedByProcessTests": {"func", 1, []string{"tests/process/support.go"}},
		"core.Reader.Close":       {"interface method", 1, []string{"core/core_test.go"}},
		"core.CodeB":              {"const", 1, []string{"core/core_test.go"}},
		"core.ErrOnlyTests":       {"var", 2, []string{"core/core_test.go"}},
		"core.Hidden":             {"type", 2, []string{"core/core_test.go"}},
	} {
		got := byKey[key]
		if got.Kind != want.kind || got.TestRefs != want.refs || !slices.Equal(got.TestFiles, want.files) {
			t.Errorf("%s = kind %q, %d references in %q; want %q, %d in %q",
				key, got.Kind, got.TestRefs, got.TestFiles, want.kind, want.refs, want.files)
		}
	}
	if position := byKey["core.TestOnly"].Position; position.Filename != "core/core.go" || position.Line == 0 {
		t.Errorf("core.TestOnly declared at %v", position)
	}
}

func TestAnalyzeLoadsOnlyTheRequestedTags(t *testing.T) {
	result, err := Analyze(Options{Dir: fixtureDir})
	if err != nil {
		t.Fatal(err)
	}
	for _, finding := range result.Findings {
		if finding.Key == "core.TaggedOnly" && finding.TestRefs != 0 {
			t.Fatalf("an untagged load counted the tagged test: %+v", finding)
		}
	}
}

func runFixture(t *testing.T, args ...string) (int, string) {
	t.Helper()
	return runOn(t, fixtureDir, args...)
}

func runOn(t *testing.T, dir string, args ...string) (int, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := run(append(args, dir), &stdout, &stderr)
	return code, stdout.String() + stderr.String()
}

func writeBaseline(t *testing.T, lines ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "baseline.txt")
	content := "# A baseline of the fixture.\n\n" + strings.Join(lines, "\n") + "\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestRunFailsOnAnExportOnlyTestsUse is V369-007's acceptance check: a new
// exported function that only a test calls fails the check, and lint passes
// once it is resolved or listed under an open task.
func TestRunFailsOnAnExportOnlyTestsUse(t *testing.T) {
	code, output := runFixture(t, "-tags", "fixture")
	if code != 1 || !strings.Contains(output,
		"core/core.go:11:6: func core.TestOnly is exported, but only tests reference it (1 reference in core/core_test.go)") ||
		!strings.Contains(output, "func core.Unused is exported, but nothing references it") {
		t.Fatalf("check without a baseline = %d:\n%s", code, output)
	}
	if strings.Contains(output, "core.Used ") || strings.Contains(output, "HelperExport") ||
		strings.Contains(output, "GeneratedOnly") || strings.Contains(output, "ProcessExport") {
		t.Fatalf("check reported production-used, helper, generated or process-test code:\n%s", output)
	}

	lines := make([]string, len(fixtureFindings))
	for index, key := range fixtureFindings {
		lines[index] = key + " V900-001"
	}
	if code, output := runFixture(t, "-tags", "fixture", "-baseline", writeBaseline(t, lines...)); code != 0 || output != "" {
		t.Fatalf("check with every finding listed = %d:\n%s", code, output)
	}
}

func TestRunFailsOnAStaleOrUnownedBaselineEntry(t *testing.T) {
	lines := make([]string, 0, len(fixtureFindings)+1)
	for _, key := range fixtureFindings {
		task := "V900-001"
		switch key {
		case "core.Unused":
			task = "V900-002"
		case "core.Recursive":
			task = "V900-003"
		}
		lines = append(lines, key+" "+task)
	}
	lines = append(lines, "core.Used V900-001")
	code, output := runFixture(t, "-tags", "fixture", "-baseline", writeBaseline(t, lines...))
	for _, want := range []string{
		fmt.Sprintf(":%d: core.Used now has a production reference or no longer exists; delete the entry",
			3+len(fixtureFindings)),
		"core.Unused names V900-002, which is finished",
		"core.Recursive names V900-003, which has no task file",
	} {
		if code != 1 || !strings.Contains(output, want) {
			t.Fatalf("check = %d, want a problem %q:\n%s", code, want, output)
		}
	}
}

// TestRunFailsWhenProductionImportsATestHelper holds the allowlist rule:
// production code that imports a test helper package fails the check
// although the fixture has no finding, since references through the helper
// would count as test references.
func TestRunFailsWhenProductionImportsATestHelper(t *testing.T) {
	const dir = "testdata/helperimport"
	result, err := Analyze(Options{Dir: dir})
	if err != nil {
		t.Fatal(err)
	}
	want := []HelperImport{{Importer: "app", Helper: "apptest"}}
	if !slices.Equal(result.HelperImports, want) {
		t.Fatalf("helper imports = %+v, want %+v", result.HelperImports, want)
	}
	if len(result.Findings) != 0 {
		t.Fatalf("findings = %+v", result.Findings)
	}
	code, output := runOn(t, dir)
	if code != 1 || !strings.Contains(output,
		"app imports the test helper package apptest; production code must not depend on test code") {
		t.Fatalf("check = %d:\n%s", code, output)
	}
}

func TestRunRejectsAMalformedBaseline(t *testing.T) {
	for _, lines := range [][]string{
		{"core.TestOnly"},
		{"core.TestOnly V900-001 extra"},
		{"core.TestOnly v900-001"},
		{"core.TestOnly V900-001", "core.TestOnly V900-001"},
	} {
		code, output := runFixture(t, "-baseline", writeBaseline(t, lines...))
		if code != 1 || !strings.Contains(output, ":3: ") && !strings.Contains(output, ":4: ") {
			t.Fatalf("baseline %q = %d:\n%s", lines, code, output)
		}
	}
}
