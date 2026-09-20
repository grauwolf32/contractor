package runtimeconfig

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestToolsetOriginsSharedPythonGolden(t *testing.T) {
	for _, test := range []struct {
		file  string
		valid bool
	}{{"origins-run.json", true}, {"origins-agent-rejected.json", false}} {
		t.Run(test.file, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("..", "..", "api", "testdata", "toolset-runtime-configuration", test.file))
			if err != nil {
				t.Fatal(err)
			}
			var value ResolvedRuntimeConfigOrigins
			if err := json.Unmarshal(raw, &value); err != nil {
				t.Fatal(err)
			}
			if err := value.Validate(); (err == nil) != test.valid {
				t.Fatalf("valid=%v, error=%v", test.valid, err)
			}
			if test.valid {
				clone := cloneOrigins(value)
				clone.Toolsets["mcp@1"].Configs[0].Name = "changed"
				if value.Toolsets["mcp@1"].Configs[0].Name == "changed" {
					t.Fatal("origins clone aliases config refs")
				}
			}
		})
	}
}
