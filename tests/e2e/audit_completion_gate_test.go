package e2e

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func completionGateMatrix(t *testing.T) (map[string][]string, map[string]int) {
	t.Helper()
	data, err := os.ReadFile("../../scripts/audit-completion-matrix.json")
	if err != nil {
		t.Fatal(err)
	}
	var matrix struct {
		Go     map[string][]string
		Python map[string]int
	}
	if err := json.Unmarshal(data, &matrix); err != nil {
		t.Fatal(err)
	}
	if len(matrix.Go) == 0 || len(matrix.Python) == 0 {
		t.Fatal("empty release matrix")
	}
	return matrix.Go, matrix.Python
}

func completionGateVerify(t *testing.T, flag, body string, passed bool) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "report")
	if err := os.WriteFile(path, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	command := exec.Command("python3", "../../scripts/test-audit-completion-e2e.py", flag, path)
	output, err := command.CombinedOutput()
	if (err == nil) != passed {
		t.Fatalf("gate exit = %v: %s", err, output)
	}
}

func TestAuditCompletionGateRejectsFalseGreenGoReports(t *testing.T) {
	required, _ := completionGateMatrix(t)
	event := func(action, pkg, test string) string {
		data, _ := json.Marshal(map[string]any{"Package": pkg, "Action": action, "Test": test})
		return string(data) + "\n"
	}
	valid := ""
	for pkg, names := range required {
		for _, name := range names {
			valid += event("run", pkg, name) + event("pass", pkg, name)
		}
		valid += event("pass", pkg, "")
	}
	bridgePackage := "github.com/grauwolf32/contractor/internal/auditimport"
	bridge := "TestAuditCompletionRuntimeZIPImporter/process-loss-same-run-and-new-child"
	databasePackage := "github.com/grauwolf32/contractor/internal/runservice"
	database := "TestAuditCompletionPostgresAtomicCreationRestartAndAllocation"
	for _, test := range []struct {
		name, body string
		passed     bool
	}{
		{"complete", valid, true}, {"empty", "", false}, {"malformed", "not json", false},
		{"no-matching-tests", event("pass", bridgePackage, ""), false},
		{"missing-bridge", strings.ReplaceAll(valid, event("pass", bridgePackage, bridge), ""), false},
		{"never-ran", strings.ReplaceAll(valid, event("run", bridgePackage, bridge), ""), false},
		{"skipped-bridge", strings.ReplaceAll(valid, event("pass", bridgePackage, bridge), event("skip", bridgePackage, bridge)), false},
		{"skipped-database", strings.ReplaceAll(valid, event("pass", databasePackage, database), event("skip", databasePackage, database)), false},
		{"failed", valid + event("fail", bridgePackage, bridge), false},
		{"unexpected-selected-skip", valid + event("skip", bridgePackage, "FutureFaultCase"), false},
		{"missing-package-pass", strings.ReplaceAll(valid, event("pass", databasePackage, ""), ""), false},
	} {
		t.Run(test.name, func(t *testing.T) { completionGateVerify(t, "--verify-go-report", test.body, test.passed) })
	}
}

func TestAuditCompletionGateRejectsIncompleteRuntimeReports(t *testing.T) {
	_, required := completionGateMatrix(t)
	var cases []string
	for name, count := range required {
		parts := strings.Split(name, "::")
		for i := 0; i < count; i++ {
			cases = append(cases, fmt.Sprintf(`<testcase classname="%s" name="%s[%d]"/>`, parts[0], parts[1], i))
		}
	}
	sort.Strings(cases)
	report := func(cases []string) string {
		return "<testsuites><testsuite>" + strings.Join(cases, "") + "</testsuite></testsuites>"
	}
	valid := report(cases)
	for _, test := range []struct {
		name, body string
		passed     bool
	}{
		{"complete", valid, true}, {"empty", "<testsuites/>", false}, {"malformed", "not XML", false},
		{"missing-parameter", report(cases[1:]), false},
		{"skip", strings.Replace(valid, "/>", "><skipped/></testcase>", 1), false},
		{"failure", strings.Replace(valid, "/>", "><failure/></testcase>", 1), false},
		{"error", strings.Replace(valid, "/>", "><error/></testcase>", 1), false},
		{"wrong-file", strings.Replace(valid, `classname="tests.`, `classname="other.`, 1), false},
	} {
		t.Run(test.name, func(t *testing.T) { completionGateVerify(t, "--verify-python-report", test.body, test.passed) })
	}
}

func TestAuditCompletionGateRequiresPrerequisites(t *testing.T) {
	for _, runtimeMissing := range []bool{false, true} {
		t.Run(fmt.Sprintf("runtime-missing-%t", runtimeMissing), func(t *testing.T) {
			path := "../../scripts/test-audit-completion-e2e.py"
			if runtimeMissing {
				root := t.TempDir()
				if err := os.Mkdir(filepath.Join(root, "scripts"), 0700); err != nil {
					t.Fatal(err)
				}
				data, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				path = filepath.Join(root, "scripts", "test-audit-completion-e2e.py")
				if err := os.WriteFile(path, data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			command := exec.Command("python3", path)
			for _, value := range os.Environ() {
				if !strings.HasPrefix(value, "CONTRACTOR_TEST_DATABASE_URL=") {
					command.Env = append(command.Env, value)
				}
			}
			expected := "CONTRACTOR_TEST_DATABASE_URL is required"
			if runtimeMissing {
				command.Env = append(command.Env, "CONTRACTOR_TEST_DATABASE_URL=postgres://unused")
				expected = "prepared runtime/.venv/bin/python are required"
			}
			output, err := command.CombinedOutput()
			if err == nil || !strings.Contains(string(output), expected) {
				t.Fatalf("missing prerequisite passed: %v %s", err, output)
			}
		})
	}
}

func TestAuditCompletionGateRedactsDatabaseCredentials(t *testing.T) {
	for _, dsn := range []string{
		"postgres://fixture:private-token%22@localhost/test",
		"host=localhost dbname=test password='private-token'",
		"host=localhost dbname=test password=''",
	} {
		command := exec.Command("python3", "-c", `
import json, os, runpy
redact = runpy.run_path('../../scripts/test-audit-completion-e2e.py')['redact']
dsn = os.environ['CONTRACTOR_TEST_DATABASE_URL']
for value in [dsn, json.dumps({'Output': dsn}), 'private-token%22', 'private-token"']:
    output = redact(value)
    if "password=''" not in dsn:
        assert 'private-token' not in output
    assert dsn not in output
`)
		for _, value := range os.Environ() {
			if !strings.HasPrefix(value, "CONTRACTOR_TEST_DATABASE_URL=") {
				command.Env = append(command.Env, value)
			}
		}
		command.Env = append(command.Env, "CONTRACTOR_TEST_DATABASE_URL="+dsn)
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("credential redaction failed: %v %s", err, output)
		}
	}
}
