package config

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestWorkflowRejectsNonportableArtifactBindings(t *testing.T) {
	for _, test := range []struct{ name, old, replacement string }{
		{"input slot", "  inputs:\n    source:", "  inputs:\n    source bundle:"},
		{"output slot", "  outputs:\n    result:", "  outputs:\n    итог:"},
		{"result name", "from: {namespace: builder, name: copied}", "from: {namespace: builder, name: review notes}"},
		{"result namespace", "from: {namespace: builder, name: copied}", "from: {namespace: review team, name: copied}"},
		{"Agent namespace", "template: artifact_builder@1", "template: artifact_builder@1\n          namespace: review team"},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := copyConfigTree(t)
			replaceFile(t, filepath.Join(root, "workflows/artifact_copy.yaml"), test.old, test.replacement)
			snapshot, err := Load(root, MVPDescriptors())
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), "ASCII") {
				t.Fatalf("Load = (%v, %v), want invalid Artifact identifier", snapshot, err)
			}
		})
	}
}
