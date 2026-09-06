package config

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

func TestWorkflowPresentationIsOptionalAndPinnedInSnapshots(t *testing.T) {
	t.Parallel()

	legacy := mustLoad(t, filepath.Join("testdata", "valid"), MVPDescriptors())
	legacyWorkflow, err := legacy.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	if legacyWorkflow.Presentation != nil {
		t.Fatalf("legacy Workflow presentation = %+v, want nil", legacyWorkflow.Presentation)
	}
	legacyJSON, err := json.Marshal(legacyWorkflow)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(legacyJSON), `"presentation"`) {
		t.Fatalf("legacy Workflow snapshot gained presentation: %s", legacyJSON)
	}
	decodedLegacy, err := DecodeResolvedWorkflowSnapshot(legacyJSON)
	if err != nil || decodedLegacy.Presentation != nil {
		t.Fatalf("decode legacy Workflow = (%+v, %v)", decodedLegacy.Presentation, err)
	}

	root := copyConfigTree(t)
	path := filepath.Join(root, "workflows", "artifact_copy.yaml")
	replaceFile(t, path, "spec:\n", "spec:\n  presentation:\n    displayName: Artifact Copier\n    description: Copies one exact text artifact.\n")
	snapshot := mustLoad(t, root, MVPDescriptors())
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	want := &WorkflowPresentation{
		DisplayName: "Artifact Copier",
		Description: "Copies one exact text artifact.",
	}
	if workflow.Presentation == nil || *workflow.Presentation != *want {
		t.Fatalf("Workflow presentation = %+v, want %+v", workflow.Presentation, want)
	}
	encoded, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeResolvedWorkflowSnapshot(encoded)
	if err != nil || decoded.Presentation == nil || *decoded.Presentation != *want {
		t.Fatalf("decoded Workflow presentation = (%+v, %v)", decoded.Presentation, err)
	}

	workflow.Presentation.DisplayName = "caller mutation"
	again, err := snapshot.Workflow("artifact-copy@1")
	if err != nil || again.Presentation == nil || again.Presentation.DisplayName != want.DisplayName {
		t.Fatalf("snapshot presentation was mutable: (%+v, %v)", again.Presentation, err)
	}
}

func TestWorkflowPresentationValidationIsStrictAndUnicodeBounded(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		presentation string
		want         string
	}{
		{name: "unknown field", presentation: "displayName: Copier\n    description: Copies text.\n    color: blue", want: "field color not found"},
		{name: "missing display name", presentation: "description: Copies text.", want: "displayName must be non-empty UTF-8"},
		{name: "blank description", presentation: "displayName: Copier\n    description: '   '", want: "description must be non-empty UTF-8"},
		{name: "long display name", presentation: "displayName: '" + strings.Repeat("界", 161) + "'\n    description: Copies text.", want: "at most 160 Unicode characters"},
		{name: "long description", presentation: "displayName: Copier\n    description: '" + strings.Repeat("界", 2001) + "'", want: "at most 2000 Unicode characters"},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			root := copyConfigTree(t)
			path := filepath.Join(root, "workflows", "artifact_copy.yaml")
			replaceFile(t, path, "spec:\n", "spec:\n  presentation:\n    "+test.presentation+"\n")
			if snapshot, err := Load(root, MVPDescriptors()); err == nil || snapshot != nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load() = (%v, %v), want %q", snapshot, err, test.want)
			}
		})
	}
}
