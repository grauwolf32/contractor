//go:build e2e

package e2e

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func writePlacementScannerFixture(t *testing.T, directory, python, marker string) string {
	t.Helper()
	if err := os.MkdirAll(directory, 0700); err != nil {
		t.Fatal(err)
	}
	payload := filepath.Join(directory, "nuclei.py")
	if err := os.WriteFile(payload, []byte(fmt.Sprintf(`import json
import sys
from pathlib import Path
marker = Path(%q)
if marker.exists():
    sys.exit(1)
print(json.dumps({"template-id": "placement-fixture", "info": {"severity": "info"}}))
`, marker)), 0600); err != nil {
		t.Fatal(err)
	}
	// Only the version path uses shell builtins. Actual scans replace this
	// process with the same Python payload and its exact argument vector.
	quote := func(value string) string { return "'" + strings.ReplaceAll(value, "'", "'\"'\"'") + "'" }
	script := fmt.Sprintf(`#!/bin/sh
for arg in "$@"; do
    if [ "$arg" = "-version" ]; then
        printf 'placement-fixture 1\n'
        exit 0
    fi
done
exec %s %s "$@"
`, quote(python), quote(payload))
	binary := filepath.Join(directory, "nuclei")
	if err := os.WriteFile(binary, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	return binary
}

func TestRuntimePlacementScannerFixture(t *testing.T) {
	root := t.TempDir()
	empty := filepath.Join(root, "empty")
	if err := os.Mkdir(empty, 0700); err != nil {
		t.Fatal(err)
	}
	run := func(binary string, arguments ...string) ([]byte, error) {
		command := exec.CommandContext(t.Context(), binary, arguments...)
		command.Env = []string{"PATH=" + empty}
		return command.CombinedOutput()
	}
	marker := filepath.Join(root, "failure marker's file")
	missing := writePlacementScannerFixture(t, filepath.Join(root, "missing python's bin"), filepath.Join(root, "absent python"), marker)
	for _, arguments := range [][]string{{"-version"}, {"-json", "-version", ""}} {
		output, err := run(missing, arguments...)
		if err != nil || string(output) != "placement-fixture 1\n" {
			t.Fatalf("version launched missing Python: output=%q err=%v", output, err)
		}
	}

	python, err := exec.LookPath("python3")
	if err != nil {
		t.Fatal(err)
	}
	python, err = filepath.Abs(python)
	if err != nil {
		t.Fatal(err)
	}
	binary := writePlacementScannerFixture(t, filepath.Join(root, "working scanner's bin"), python, marker)
	output, err := run(binary, "-u", "https://fixture.test/scan", "-jsonl")
	var observation map[string]any
	if err != nil || json.Unmarshal(output, &observation) != nil || len(observation) != 2 || observation["template-id"] != "placement-fixture" || observation["info"].(map[string]any)["severity"] != "info" {
		t.Fatalf("scan observation changed: output=%q err=%v", output, err)
	}
	if err := os.WriteFile(marker, []byte("enabled"), 0600); err != nil {
		t.Fatal(err)
	}
	output, err = run(binary, "-u", "https://fixture.test/scan")
	var failed *exec.ExitError
	if !errors.As(err, &failed) || failed.ExitCode() != 1 || len(output) != 0 {
		t.Fatalf("fail marker changed: output=%q err=%v", output, err)
	}
	if output, err := run(binary, "-version"); err != nil || string(output) != "placement-fixture 1\n" {
		t.Fatalf("version consulted scan fail marker: output=%q err=%v", output, err)
	}

	// A recording interpreter proves exec preserves the payload path and every
	// argument literally, including empty arguments and shell metacharacters.
	recorder := filepath.Join(root, "recording python's executable")
	if err := os.WriteFile(recorder, []byte("#!/bin/sh\nprintf '%s\\n' \"$@\"\n"), 0700); err != nil {
		t.Fatal(err)
	}
	directory := filepath.Join(root, "argument fixture's bin")
	binary = writePlacementScannerFixture(t, directory, recorder, marker)
	arguments := []string{"-u", "a path's value; $literal", "", "-jsonl"}
	output, err = run(binary, arguments...)
	want := strings.Join(append([]string{filepath.Join(directory, "nuclei.py")}, arguments...), "\n") + "\n"
	if err != nil || string(output) != want {
		t.Fatalf("scan argv changed: output=%q want=%q err=%v", output, want, err)
	}
}
