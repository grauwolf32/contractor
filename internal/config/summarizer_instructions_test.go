package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func installSummaryInstructions(t *testing.T, root, text string) {
	t.Helper()
	installWorkerSummarizer(t, root)
	replaceFile(t, filepath.Join(root, "agent-templates/artifact_builder.yaml"), "  summarizer:\n", "  summarizer:\n    instructions:\n      ref: instructions/test-summary.md\n")
	if err := os.WriteFile(filepath.Join(root, "instructions/test-summary.md"), []byte(text), 0600); err != nil {
		t.Fatal(err)
	}
}

func TestSummaryInstructionsResolvePinAndPublish(t *testing.T) {
	root := copyConfigTree(t)
	text := "Составь итог. Preserve {subtaskId} and JSON: {\"result\": \"text\"}.\n"
	installSummaryInstructions(t, root, text)
	snapshot := mustLoad(t, root, MVPDescriptors())
	template, err := snapshot.AgentTemplate("artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	instruction := template.Summarizer.Instructions
	if instruction == nil || instruction.Ref != "instructions/test-summary.md" || instruction.Text != text || instruction.Digest != digestBytes([]byte(text)) {
		t.Fatalf("resolved instructions = %+v", instruction)
	}
	originalDigest := template.Ref.Digest
	safe := agentTemplateResourceBody(template)["summarizer"].(map[string]any)["instructions"].(map[string]any)
	if len(safe) != 2 || safe["ref"] != instruction.Ref || safe["digest"] != instruction.Digest {
		t.Fatalf("safe instructions = %+v", safe)
	}
	instruction.Text = "mutated"
	again, _ := snapshot.AgentTemplate("artifact_builder@1")
	if again.Summarizer.Instructions.Text != text {
		t.Fatal("snapshot instructions are mutable through a returned template")
	}
	if err := os.WriteFile(filepath.Join(root, "instructions/test-summary.md"), []byte(text+"Updated."), 0600); err != nil {
		t.Fatal(err)
	}
	updated, _ := mustLoad(t, root, MVPDescriptors()).AgentTemplate("artifact_builder@1")
	if updated.Ref.Digest == originalDigest {
		t.Fatal("instruction change did not affect template digest")
	}
}

func TestSummaryInstructionsRejectInvalidConfig(t *testing.T) {
	for _, tc := range []struct {
		name, text, want string
		missing          bool
	}{
		{name: "8000 Unicode characters", text: strings.Repeat("界", 8000)},
		{name: "8001 Unicode characters", text: strings.Repeat("界", 8001), want: "8000"},
		{name: "blank", text: " \n\t", want: "empty"},
		{name: "missing file", text: "summary", missing: true, want: "unknown instruction ref"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := copyConfigTree(t)
			installSummaryInstructions(t, root, tc.text)
			if tc.missing {
				if err := os.Remove(filepath.Join(root, "instructions/test-summary.md")); err != nil {
					t.Fatal(err)
				}
			}
			_, err := Load(root, MVPDescriptors())
			if tc.want == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %s", err, tc.want)
			}
		})
	}
}

func TestSummaryInstructionGoldenDigest(t *testing.T) {
	var allocation contracts.AllocationSpec
	encoded := readFile(t, filepath.Join("..", "..", "api", "testdata", "v1alpha1", "valid", "allocation-spec-summarizer-instructions.json"))
	if err := json.Unmarshal(encoded, &allocation); err != nil {
		t.Fatal(err)
	}
	if err := allocation.Validate(); err != nil {
		t.Fatal(err)
	}
	template := allocation.AgentTemplate
	instruction := template.Summarizer.Instructions
	if instruction == nil || instruction.Digest != digestBytes([]byte(instruction.Text)) {
		t.Fatal("incorrect instruction digest")
	}
	got, err := agentTemplateDigest(Selector{ID: template.Ref.TemplateID, Version: template.Ref.Version}, template)
	if err != nil {
		t.Fatal(err)
	}
	if got != template.Ref.Digest {
		t.Fatalf("template digest = %s, want %s", got, template.Ref.Digest)
	}
}
