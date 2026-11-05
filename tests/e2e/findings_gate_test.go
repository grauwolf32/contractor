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
	// The verifier owns case membership. Keep these report mutations focused
	// on verification behavior instead of maintaining a second case list.
	output, err := exec.Command("python3", "-c", `import json, runpy, sys
sys.path.insert(0, "../../scripts")
registry = runpy.run_path("../../scripts/test-findings-e2e.py")
print(json.dumps(sorted(registry["REQUIRED_TESTS"])))`).CombinedOutput()
	if err != nil {
		t.Fatalf("read Findings case registry: %v: %s", err, output)
	}
	var required []string
	if err := json.Unmarshal(output, &required); err != nil || len(required) == 0 {
		t.Fatalf("invalid Findings case registry: %v", err)
	}
	child := ""
	for _, name := range required {
		if strings.Contains(name, "/") {
			child = name
			break
		}
	}
	if child == "" {
		t.Fatal("Findings registry must include mandatory subcases")
	}
	event := func(action, test string) string {
		data, _ := json.Marshal(map[string]any{"Package": "github.com/grauwolf32/contractor/tests/e2e", "Action": action, "Test": test})
		return string(data) + "\n"
	}
	valid := ""
	for _, name := range required {
		valid += event("run", name)
	}
	for i := len(required) - 1; i >= 0; i-- {
		valid += event("pass", required[i])
	}
	valid += event("pass", "")
	for _, test := range []struct {
		name, body string
		passed     bool
	}{
		{"complete", valid, true}, {"empty", "", false}, {"no-matching-tests", event("pass", ""), false},
		{"missing-case", strings.ReplaceAll(valid, event("pass", child), ""), false},
		{"skip-under-passing-parent", strings.ReplaceAll(valid, event("pass", child), event("skip", child)), false},
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
	for _, name := range []string{"test-findings-e2e.py", "release_test_reports.py"} {
		script, err := os.ReadFile(filepath.Join("../../scripts", name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "scripts", name), script, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(root, "scripts", "test-findings-e2e.py")
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
