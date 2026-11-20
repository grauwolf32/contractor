package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDocumentResponseRecoveryAuthoring(t *testing.T) {
	for _, value := range []string{"document@1", "unknown", "null", "''"} {
		t.Run(value, func(t *testing.T) {
			root := copyCoreFixture(t)
			// The ordinary artifact fixture has document tools only after this explicit
			// selection. Other tests continue to exercise its original tool authority.
			path := filepath.Join(root, "agent-templates/artifact_builder.yaml")
			content, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			text := strings.Replace(string(content), "  runtime: adk@1", "  runtime: adk@1\n  responseRecovery: "+value, 1)
			if err := os.WriteFile(path, []byte(text), 0600); err != nil {
				t.Fatal(err)
			}
			// The fixture has filesystem tools, so even document@1 must be rejected.
			if _, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed")); err == nil {
				t.Fatal("invalid response recovery selection admitted")
			}
		})
	}
}
