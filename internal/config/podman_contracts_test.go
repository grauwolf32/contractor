package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPodmanAuthoringChecksWithoutFleetLookup(t *testing.T) {
	for _, test := range []struct {
		name, mode, profile, tool string
		want                      string
	}{
		{"direct", "direct", "podman@1", "exec_command", ""},
		{"overlay", "overlay", "podman@1", "exec_command", "requires context.workspace.mode direct"},
		{"missing-workspace", "", "podman@1", "exec_command", "requires context.workspace.mode direct"},
		{"host-fallback", "direct", "local-workdir@1", "exec_command", "requires SandboxProfile podman@1"},
		{"unknown-tool", "direct", "podman@1", "exec_python", "does not export selected tool"},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := filepath.Join(t.TempDir(), "config")
			if err := os.CopyFS(root, os.DirFS("testdata/valid")); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(root, "agent-templates/artifact_builder.yaml")
			replaceFile(t, path, "sandboxProfile: local-workdir@1", "sandboxProfile: "+test.profile)
			replaceFile(t, path, `    - ref: run-artifacts@1
      tools: [list_artifacts, read_artifact, write_artifact]`, "    - ref: code-execution@1\n      tools: ["+test.tool+"]")
			if test.mode != "" {
				workflow := filepath.Join(root, "workflows/artifact_copy.yaml")
				replaceFile(t, workflow, "      context:\n", "      context:\n        workspace:\n          mode: "+test.mode+"\n          sources: [{artifact: source, target: \"\"}]\n")
				replaceFile(t, workflow, "source: {required: true, mediaTypes: [text/plain]}", "source: {required: true, mediaTypes: [application/zip]}")
			}
			snapshot, err := Load(root)
			if test.want == "" {
				if err != nil || snapshot == nil {
					t.Fatalf("valid Podman authoring: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %s", err, test.want)
			}
		})
	}
}
