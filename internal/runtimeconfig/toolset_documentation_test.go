package runtimeconfig

import (
	"context"
	"encoding/json"
	"os"
	"regexp"
	"testing"

	"go.yaml.in/yaml/v4"
)

func TestMCPGuideConfigurationExamplePublishesWithNormalizedDefaults(t *testing.T) {
	document, err := os.ReadFile("../../docs/guides/toolset-connections.md")
	if err != nil {
		t.Fatal(err)
	}
	blocks := regexp.MustCompile("(?s)```yaml\\n(.*?)\\n```").FindAllSubmatch(document, -1)
	found := 0
	for _, block := range blocks {
		var value map[string]any
		if err := yaml.Unmarshal(block[1], &value); err != nil {
			t.Fatal(err)
		}
		if value["kind"] != "RuntimeConfig" {
			continue
		}
		found++
		raw, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		prepared, err := PreparePublication(raw)
		if err != nil {
			t.Fatal(err)
		}
		published, err := prepared.Resolve(context.Background(), nil)
		if err != nil {
			t.Fatal(err)
		}
		config := published.Spec.Worker.Toolsets["mcp@1"].Value.(MCPConfig)
		if config.Endpoint != "https://knowledge.example.com/mcp" || config.ConnectTimeoutSeconds != 10 || config.RequestTimeoutSeconds != 60 {
			t.Fatal("guide example changed normalized settings")
		}
	}
	if found != 1 {
		t.Fatalf("expected one complete RuntimeConfig example, got %d", found)
	}
}
