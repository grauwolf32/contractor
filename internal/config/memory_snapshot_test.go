package config

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"go.yaml.in/yaml/v4"
)

func TestSharedMemorySnapshotStageContract(t *testing.T) {
	var fixture struct {
		Workflow string
		Cases    []struct {
			Name  string
			Path  []string
			Value any
			Valid bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "server-operation-memory-snapshot.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Workflow != string(readFile(t, "testdata/memory-snapshot/workflows/snapshot.yaml")) {
		t.Fatal("process catalog fixture drift")
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			var source map[string]any
			if err := yaml.Unmarshal([]byte(fixture.Workflow), &source); err != nil {
				t.Fatal(err)
			}
			stage := source["spec"].(map[string]any)["stages"].(map[string]any)["snapshot"].(map[string]any)
			for _, part := range c.Path[:len(c.Path)-1] {
				stage = stage[part].(map[string]any)
			}
			stage[c.Path[len(c.Path)-1]] = c.Value
			data, err := yaml.Marshal(source)
			if err != nil {
				t.Fatal(err)
			}
			root := copyCoreFixture(t)
			writeFile(t, filepath.Join(root, "workflows", "snapshot.yaml"), data)
			_, err = LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
			if (err == nil) != c.Valid {
				t.Fatalf("valid=%t err=%v", c.Valid, err)
			}
		})
	}
}

func TestMemorySnapshotRejectsInvalidLiteralSelectionsAndOutputs(t *testing.T) {
	source := string(readFile(t, "testdata/memory-snapshot/workflows/snapshot.yaml"))
	for _, c := range []struct{ Name, From, To string }{
		{"empty names", "{source: parameter, name: note_names, default: selected}", "{source: literal, value: ''}"},
		{"duplicate names", "{source: parameter, name: note_names, default: selected}", "{source: literal, value: 'selected,selected'}"},
		{"invalid names", "{source: parameter, name: note_names, default: selected}", "{source: literal, value: 'selected-note'}"},
		{"foreign scope", "value: writer", "value: run/foreign"},
		{"reserved namespace", "value: writer", "value: inputs"},
		{"reserved output", "name: selected-notes", "name: memory.selected"},
		{"hidden body output", "name: selected-notes", "name: http.body.selected"},
		{"missing output", "required: true\n            mediaTypes", "required: false\n            mediaTypes"},
		{"context cannot expose memory", "context: {artifacts: {}}", "context: {artifacts: {note: {namespace: writer, name: memory.selected, required: true}}}"},
		{"unknown source argument", "          namespace:", "          run_id: {source: literal, value: foreign}\n          namespace:"},
	} {
		t.Run(c.Name, func(t *testing.T) {
			changed := strings.Replace(source, c.From, c.To, 1)
			if changed == source {
				t.Fatal("case did not mutate fixture")
			}
			root := copyCoreFixture(t)
			writeFile(t, filepath.Join(root, "workflows", "snapshot.yaml"), []byte(changed))
			if _, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed")); err == nil {
				t.Fatal("invalid export configuration accepted")
			}
		})
	}
}
