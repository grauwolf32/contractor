package config

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestAgentInstructionsUseLoadedSnapshot(t *testing.T) {
	root := copyConfigTree(t)
	snapshot := mustLoad(t, root, MVPDescriptors())
	before, err := snapshot.AgentInstructions("artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	if before.Template.TemplateID != "artifact_builder" || before.Instructions.Text == "" {
		t.Fatalf("missing loaded instructions: %+v", before)
	}
	if err := os.WriteFile(filepath.Join(root, before.Instructions.Ref), []byte("changed after loading"), 0600); err != nil {
		t.Fatal(err)
	}
	after, err := snapshot.AgentInstructions("artifact_builder@1")
	if err != nil || after != before {
		t.Fatalf("instruction read did not preserve loaded snapshot: %+v, %v", after, err)
	}
	if _, err := snapshot.AgentInstructions("missing@1"); !errors.Is(err, ErrConfigurationNotFound) {
		t.Fatalf("missing template error = %v", err)
	}
	if _, err := snapshot.AgentInstructions("../instructions@1"); err == nil {
		t.Fatal("invalid selector was accepted")
	}
}
