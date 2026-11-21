package config

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"go.yaml.in/yaml/v4"
)

func TestSharedArtifactCopyStageContract(t *testing.T) {
	var fixture struct {
		Workflow string
		Cases    []struct {
			Name  string
			Path  []string
			Value any
			Valid bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "server-operation-artifact-copy.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Workflow != string(readFile(t, "../operations/artifactcopy/testdata/copy.yaml")) {
		t.Fatal("process catalog fixture drift")
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			var source map[string]any
			if err := yaml.Unmarshal([]byte(fixture.Workflow), &source); err != nil {
				t.Fatal(err)
			}
			stage := source["spec"].(map[string]any)["stages"].(map[string]any)["copy"].(map[string]any)
			for _, part := range c.Path[:len(c.Path)-1] {
				stage = stage[part].(map[string]any)
			}
			stage[c.Path[len(c.Path)-1]] = c.Value
			data, err := yaml.Marshal(source)
			if err != nil {
				t.Fatal(err)
			}
			root := copyCoreFixture(t)
			writeFile(t, filepath.Join(root, "workflows", "copy.yaml"), data)
			_, err = LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
			if (err == nil) != c.Valid {
				t.Fatalf("valid=%t err=%v", c.Valid, err)
			}
		})
	}
}
