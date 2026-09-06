package e2e

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestFindingsGateRejectsFalseGreenProcessReports(t *testing.T) {
	required := []string{
		"TestFindingsProducerAndReaderAcrossProcesses",
		"TestFindingsCollectionsRetainOrdinaryAndAuditReceipts",
		"TestFindingsCollectionsRetainOrdinaryAndAuditReceipts/shared-function-preserves-operations",
		"TestFindingsCollectionsRetainOrdinaryAndAuditReceipts/ordinary-hypothesis-and-intake-replay",
		"TestFindingsCollectionsRetainOrdinaryAndAuditReceipts/snapshot-survives-source-deletion",
		"TestFindingsCollectionsRetainOrdinaryAndAuditReceipts/changed-selection-conflicts",
		"TestFindingsCollectionsRetainOrdinaryAndAuditReceipts/pagination-and-exact-evidence",
		"TestFindingsReaderBoundariesAcrossProcesses",
		"TestFindingsReaderBoundariesAcrossProcesses/inaccessible-input",
		"TestFindingsReaderBoundariesAcrossProcesses/empty-is-explicit-success",
		"TestFindingsReaderBoundariesAcrossProcesses/interrupted-preparation-reuses-exact-ref",
		"TestFindingsReaderBoundariesAcrossProcesses/conflicting-document-fails",
		"TestFindingsReaderBoundariesAcrossProcesses/invalid-zip-fails",
	}
	event := func(action, test string) string {
		data, _ := json.Marshal(map[string]any{"Package": "github.com/grauwolf32/contractor/tests/e2e", "Action": action, "Test": test})
		return string(data) + "\n"
	}
	valid := ""
	for _, name := range required {
		valid += event("run", name) + event("pass", name)
	}
	valid += event("pass", "")
	for _, test := range []struct {
		name, body string
		passed     bool
	}{
		{"complete", valid, true}, {"empty", "", false}, {"no-matching-tests", event("pass", ""), false},
		{"missing-case", strings.ReplaceAll(valid, event("pass", required[4]), ""), false},
		{"skip-under-passing-parent", strings.ReplaceAll(valid, event("pass", required[4]), event("skip", required[4])), false},
		{"never-ran", strings.ReplaceAll(valid, event("run", required[0]), ""), false},
		{"failed", valid + event("fail", required[0]), false}, {"malformed", "not JSON", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "go.jsonl")
			if err := os.WriteFile(path, []byte(test.body), 0o600); err != nil {
				t.Fatal(err)
			}
			command := exec.Command("python3", "../../scripts/test-findings-e2e.py", "--verify-go-report", path)
			output, err := command.CombinedOutput()
			if (err == nil) != test.passed {
				t.Fatalf("gate exit = %v: %s", err, output)
			}
		})
	}
}

func TestFindingsGateRejectsSkippedAndAbsentRuntimeCases(t *testing.T) {
	// Runtime membership is checked independently of pytest's zero exit code.
	for _, body := range []string{"<testsuites/>", "<testsuite><testcase name=\"test_empty_collection_is_success\"><skipped/></testcase></testsuite>"} {
		path := filepath.Join(t.TempDir(), "runtime.xml")
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		command := exec.Command("python3", "../../scripts/test-findings-e2e.py", "--verify-python-report", path)
		output, err := command.CombinedOutput()
		if err == nil || !strings.Contains(string(output), "did not") {
			t.Fatalf("false-green runtime gate: %v %s", err, output)
		}
	}
}

func TestFindingsGateRequiresProcessPrerequisites(t *testing.T) {
	command := exec.Command("python3", "../../scripts/test-findings-e2e.py")
	for _, value := range os.Environ() {
		if !strings.HasPrefix(value, "CONTRACTOR_TEST_DATABASE_URL=") {
			command.Env = append(command.Env, value)
		}
	}
	output, err := command.CombinedOutput()
	if err == nil || !strings.Contains(string(output), "CONTRACTOR_TEST_DATABASE_URL is required") {
		t.Fatalf("missing database silently passed: %v %s", err, output)
	}
}

func TestFindingsGateRequiresRuntimePrerequisite(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "scripts"), 0o700); err != nil {
		t.Fatal(err)
	}
	script, err := os.ReadFile("../../scripts/test-findings-e2e.py")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "scripts", "test-findings-e2e.py")
	if err := os.WriteFile(path, script, 0o600); err != nil {
		t.Fatal(err)
	}
	command := exec.Command("python3", path)
	for _, value := range os.Environ() {
		if !strings.HasPrefix(value, "CONTRACTOR_TEST_DATABASE_URL=") {
			command.Env = append(command.Env, value)
		}
	}
	command.Env = append(command.Env, "CONTRACTOR_TEST_DATABASE_URL=postgres://unused-for-preflight")
	output, err := command.CombinedOutput()
	if err == nil || !strings.Contains(string(output), "prepared runtime/.venv/bin/python are required") {
		t.Fatalf("missing Runtime silently passed: %v %s", err, output)
	}
}
