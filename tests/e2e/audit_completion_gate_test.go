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
	t.Run("matrix-references-declared-tests", func(t *testing.T) {
		// Synthetic JUnit reports cannot detect a stale matrix entry: they copy
		// its names. Check the actual source declarations without importing the
		// Runtime or requiring its dependencies in this offline meta-test.
		const probe = `
import ast
import json
from pathlib import Path

root = Path('../..')
required = json.loads((root / 'scripts/audit-completion-matrix.json').read_text())['python']
declared = {}
missing = []
for case, minimum in required.items():
    module, name = case.split('::')
    assert isinstance(minimum, int) and minimum > 0, 'invalid required case count: ' + case
    path = root / 'runtime' / (module.replace('.', '/') + '.py')
    if path not in declared:
        tree = ast.parse(path.read_text(), filename=str(path))
        declared[path] = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if name not in declared[path]:
        missing.append(case)
assert not missing, 'mandatory Runtime tests are not declared: ' + ', '.join(sorted(missing))
`
		if output, err := exec.Command("python3", "-c", probe).CombinedOutput(); err != nil {
			t.Fatalf("Runtime matrix/source mismatch: %v %s", err, output)
		}
	})
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
	t.Run("subprocess-composition", testAuditCompletionGateCommandComposition)
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

func testAuditCompletionGateCommandComposition(t *testing.T) {
	// Exercise the saved evidence consumed by the verifier, not just redact in isolation.
	const probe = `
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
from urllib.parse import quote

gate = runpy.run_path('../../scripts/test-audit-completion-e2e.py')
mode, password, directory = sys.argv[1], sys.argv[2], Path(sys.argv[3])
dsn = 'postgres://fixture:' + quote(password, safe='') + '@127.0.0.1/test'
os.environ['CONTRACTOR_TEST_DATABASE_URL'] = dsn
required = gate['matrix']()['go']
expected = sorted(package + '/' + name for package, names in required.items() for name in names)
events = []
for package, names in required.items():
    for name in names:
        events.extend({'Package': package, 'Test': name, 'Action': action} for action in ['run', 'pass'])
    events.append({'Package': package, 'Action': 'pass'})
package, names = next(iter(required.items()))
diagnostic = 'plain=' + password + ' escaped=' + json.dumps(password) + ' dsn=' + dsn + ' escaped_dsn=' + json.dumps(dsn) + '\n'
events.insert(0, {'Action': 'output', 'Package': package, 'Output': diagnostic,
                  'unrecognized-field-canary': {'value': 'unrecognized-value-canary', dsn: password}})
if mode in {'skip', 'fail'}:
    events[2]['Action'] = mode
elif mode == 'missing-pass':
    del events[2]
elif mode == 'missing-run':
    del events[1]
elif mode == 'missing-package':
    events = [event for event in events if event != {'Package': package, 'Action': 'pass'}]
elif mode == 'unexpected-skip':
    events.append({'Package': package, 'Test': 'UnexpectedCase', 'Action': 'skip'})
elif mode == 'invalid-event':
    events.append(['invalid-event', password])
elif mode == 'invalid-field':
    events.append({'Action': ['pass', password], 'Package': package})
raw = ''.join(json.dumps(event) + '\n' for event in events)
if mode == 'malformed':
    raw += 'invalid-json ' + diagnostic
stderr = diagnostic if mode == 'stderr' else ''
emitter = directory / 'emitter.py'
emitter.write_text('import sys\nsys.stdout.write(' + repr(raw) + ')\nsys.stdout.flush()\nsys.stderr.write(' + repr(stderr) + ')\nsys.exit(' + ('7' if mode == 'nonzero' else '0') + ')\n')
report = directory / 'go.jsonl'
displayed = io.StringIO()
command_error = None
with contextlib.redirect_stdout(displayed):
    try:
        gate['run_command']([sys.executable, str(emitter)], cwd=Path.cwd(), report=report, go=True)
    except gate['GateError'] as error:
        command_error = error
if mode == 'nonzero':
    assert command_error is not None and 'exit 7' in str(command_error), 'subprocess status was not enforced'
else:
    assert command_error is None, 'unexpected subprocess failure'

def secret_free(text):
    for secret in (password, quote(password, safe=''), dsn):
        assert secret not in text, 'diagnostic exposed a credential'
        assert json.dumps(secret)[1:-1] not in text, 'diagnostic exposed an escaped credential'

secret_free(displayed.getvalue())
saved = report.read_text()
for canary in ('unrecognized-field-canary', 'unrecognized-value-canary'):
    assert canary not in saved + displayed.getvalue(), 'arbitrary event fields were retained'
assert dsn not in saved and json.dumps(dsn)[1:-1] not in saved, 'saved report exposed a DSN'
assert '[redacted]' in saved and '[test database]' in saved, 'diagnostic redaction evidence is missing'
for line in saved.splitlines():
    try:
        event = json.loads(line)
    except ValueError:
        secret_free(line)
        continue
    if isinstance(event, dict) and 'Output' in event:
        secret_free(event['Output'])

try:
    verified = gate['verify_go_report'](report)
except gate['GateError']:
    assert mode not in {'complete', 'nonzero'}, 'valid mandatory evidence was corrupted'
else:
    assert mode in {'complete', 'nonzero'}, 'invalid mandatory evidence was accepted'
    assert verified == expected, 'mandatory case identity changed'
`
	for _, test := range []struct{ mode, password string }{
		{"complete", "contractor"},
		{"complete", "pass"},
		{"complete", "private\"token\\value"},
		{"skip", "contractor"},
		{"fail", "pass"},
		{"missing-pass", "contractor"},
		{"missing-run", "pass"},
		{"missing-package", "contractor"},
		{"unexpected-skip", "pass"},
		{"malformed", "contractor"},
		{"invalid-event", "contractor"},
		{"invalid-field", "contractor"},
		{"stderr", "private\"token\\value"},
		{"nonzero", "private\"token\\value"},
	} {
		t.Run(test.mode+"-"+test.password, func(t *testing.T) {
			command := exec.Command("python3", "-c", probe, test.mode, test.password, t.TempDir())
			if output, err := command.CombinedOutput(); err != nil {
				t.Fatalf("subprocess/report composition failed: %v %s", err, output)
			}
		})
	}
}
