//go:build e2e

package uistack

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// This is the subset of Playwright's JSON reporter needed to prove execution.
// A successful process alone can also represent an empty or skipped selection.
type browserReport struct {
	Suites []browserReportSuite `json:"suites"`
	Errors []json.RawMessage    `json:"errors"`
	Stats  *struct {
		Expected   int `json:"expected"`
		Unexpected int `json:"unexpected"`
		Flaky      int `json:"flaky"`
		Skipped    int `json:"skipped"`
	} `json:"stats"`
}

type browserReportSuite struct {
	Specs []struct {
		Title string `json:"title"`
		File  string `json:"file"`
		OK    bool   `json:"ok"`
		Tests []struct {
			ExpectedStatus string `json:"expectedStatus"`
			Status         string `json:"status"`
			Results        []struct {
				Status string            `json:"status"`
				Errors []json.RawMessage `json:"errors"`
			} `json:"results"`
		} `json:"tests"`
	} `json:"specs"`
	Suites []browserReportSuite `json:"suites"`
}

func assertBrowserReport(t *testing.T, path string, requiredFiles []string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read Playwright execution report: %v", err)
	}
	count, err := validateBrowserReport(data, requiredFiles)
	if err != nil {
		t.Fatalf("validate Playwright execution report: %v", err)
	}
	t.Logf("Playwright execution verified: %d passed, 0 skipped/unexpected/flaky; required files=%v; report SHA-256=%x", count, requiredFiles, sha256.Sum256(data))
}

func validateBrowserReport(data []byte, requiredFiles []string) (int, error) {
	var report browserReport
	if err := json.Unmarshal(data, &report); err != nil {
		return 0, fmt.Errorf("invalid JSON: %w", err)
	}
	if len(report.Errors) != 0 {
		return 0, fmt.Errorf("report contains %d runner errors", len(report.Errors))
	}
	if report.Stats == nil || report.Stats.Expected <= 0 {
		return 0, fmt.Errorf("report has no expected executions")
	}
	if report.Stats.Skipped != 0 || report.Stats.Unexpected != 0 || report.Stats.Flaky != 0 {
		return 0, fmt.Errorf("report has skipped=%d unexpected=%d flaky=%d", report.Stats.Skipped, report.Stats.Unexpected, report.Stats.Flaky)
	}
	counts := make(map[string]int, len(requiredFiles))
	for _, file := range requiredFiles {
		name := filepath.Base(file)
		if _, exists := counts[name]; exists {
			return 0, fmt.Errorf("duplicate required browser file %q", name)
		}
		counts[name] = 0
	}
	var visit func([]browserReportSuite) (int, error)
	visit = func(suites []browserReportSuite) (int, error) {
		total := 0
		for _, suite := range suites {
			for _, spec := range suite.Specs {
				if !spec.OK || spec.File == "" || len(spec.Tests) == 0 {
					return 0, fmt.Errorf("spec %q has no successful executions", spec.Title)
				}
				for _, test := range spec.Tests {
					if test.ExpectedStatus != "passed" || test.Status != "expected" || len(test.Results) == 0 {
						return 0, fmt.Errorf("spec %q did not execute as an expected pass", spec.Title)
					}
					for _, result := range test.Results {
						if result.Status != "passed" || len(result.Errors) != 0 {
							return 0, fmt.Errorf("spec %q contains a non-passing attempt", spec.Title)
						}
					}
					total++
					counts[filepath.Base(spec.File)]++
				}
			}
			count, err := visit(suite.Suites)
			if err != nil {
				return 0, err
			}
			total += count
		}
		return total, nil
	}
	total, err := visit(report.Suites)
	if err != nil {
		return 0, err
	}
	if total != report.Stats.Expected {
		return 0, fmt.Errorf("report expected count %d differs from %d executed tests", report.Stats.Expected, total)
	}
	for _, file := range requiredFiles {
		if counts[filepath.Base(file)] == 0 {
			return 0, fmt.Errorf("required browser file %q has no executed tests", file)
		}
	}
	return total, nil
}

func TestBrowserReportRequiresExecutedSelection(t *testing.T) {
	// Includes a describe suite: checking only top-level specs would miss it.
	const valid = `{
		"suites": [{"specs": [{"title": "real stack", "file": "stack.spec.ts", "ok": true,
			"tests": [{"expectedStatus": "passed", "status": "expected", "results": [{"status": "passed", "errors": []}]}]}],
			"suites": [{"specs": [{"title": "nested audit", "file": "audits.spec.ts", "ok": true,
				"tests": [{"expectedStatus": "passed", "status": "expected", "results": [{"status": "passed", "errors": []}]}]}]}]}],
		"errors": [], "stats": {"expected": 2, "unexpected": 0, "flaky": 0, "skipped": 0}
	}`
	required := []string{"e2e/stack.spec.ts", "e2e/audits.spec.ts"}
	if count, err := validateBrowserReport([]byte(valid), required); err != nil || count != 2 {
		t.Fatalf("valid nested report: count=%d err=%v", count, err)
	}
	cases := []struct {
		name, data, want string
	}{
		{"malformed JSON", `{`, "invalid JSON"},
		{"empty selection", `{"suites":[],"stats":{"expected":0}}`, "no expected executions"},
		{"missing statistics", `{"suites":[]}`, "no expected executions"},
		{"missing required file", strings.ReplaceAll(valid, "audits.spec.ts", "unrelated.spec.ts"), "required browser file"},
		{"declared without execution", strings.Replace(valid, `"results": [{"status": "passed", "errors": []}]`, `"results": []`, 1), "did not execute"},
		{"skipped selection", strings.Replace(valid, `"skipped": 0`, `"skipped": 1`, 1), "skipped=1"},
		{"unexpected selection", strings.Replace(valid, `"unexpected": 0`, `"unexpected": 1`, 1), "unexpected=1"},
		{"flaky selection", strings.Replace(valid, `"flaky": 0`, `"flaky": 1`, 1), "flaky=1"},
		{"skipped attempt despite summary", strings.Replace(valid, `"status": "passed"`, `"status": "skipped"`, 1), "non-passing attempt"},
		{"failed attempt despite summary", strings.Replace(valid, `"status": "passed"`, `"status": "failed"`, 1), "non-passing attempt"},
		{"flaky test despite summary", strings.Replace(valid, `"status": "expected"`, `"status": "flaky"`, 1), "did not execute"},
		{"expected failure is not a pass", strings.Replace(valid, `"expectedStatus": "passed"`, `"expectedStatus": "failed"`, 1), "did not execute"},
		{"false spec result", strings.Replace(valid, `"ok": true`, `"ok": false`, 1), "no successful executions"},
		{"inflated count", strings.Replace(valid, `"expected": 2`, `"expected": 3`, 1), "differs from"},
		{"runner error", strings.Replace(valid, `"errors": [], "stats"`, `"errors": [{"message":"global teardown failed"}], "stats"`, 1), "runner errors"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			_, err := validateBrowserReport([]byte(test.data), required)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("invalid execution evidence accepted or wrong error: got %v, want %q", err, test.want)
			}
		})
	}
}
