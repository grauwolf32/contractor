package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/agentskills"
)

func TestPackageAndValidateCommands(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "example")
	if err := os.Mkdir(source, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "SKILL.md"), []byte("---\nname: example\ndescription: Example.\n---\n# Example\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(root, "renamed.zip")
	var stdout, stderr bytes.Buffer
	if err := run([]string{"package", source, output}, &stdout, &stderr); err != nil {
		t.Fatalf("package: %v (%s)", err, stderr.String())
	}
	first, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	if err := run([]string{"validate", "--name", "example", output}, &stdout, &stderr); err != nil {
		t.Fatalf("validate: %v", err)
	}
	if !strings.Contains(stdout.String(), `"name":"example"`) {
		t.Fatalf("summary = %s", stdout.String())
	}
	if err := run([]string{"package", source, output}, &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	second, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("CLI output is not deterministic")
	}
	if _, err := agentskills.Validate(second, "example"); err != nil {
		t.Fatal(err)
	}
}

func TestCommandDiagnosticsDoNotExposeInputPath(t *testing.T) {
	secretPath := filepath.Join(t.TempDir(), "secret-name.zip")
	err := run([]string{"validate", secretPath}, &bytes.Buffer{}, &bytes.Buffer{})
	if err == nil || strings.Contains(err.Error(), secretPath) {
		t.Fatalf("unsafe diagnostic: %v", err)
	}
}
