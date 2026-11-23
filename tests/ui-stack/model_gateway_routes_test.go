//go:build e2e

package uistack

import (
	"context"
	"encoding/json"
	"testing"
)

func TestBrowserGatewayGeneratedModelRoutes(t *testing.T) {
	for _, role := range []string{"planner", "worker"} {
		t.Run(role, func(t *testing.T) {
			g := newModelGateway("test-only")
			defer g.close()
			g.release()
			tools := withMemoryTools([]string{"add_subtask", "execute_current_subtask", "finish", "get_worker_tool_usage", "list_subtasks"})
			marker := streamlineGlobalMarker
			if role == "worker" {
				tools = withMemoryTools([]string{"list_artifacts", "read_artifact", "write_text_artifact"})
				marker = streamlineWorkerMarker
			}
			declarations := make([]any, 0, len(tools))
			for _, name := range tools {
				declarations = append(declarations, map[string]any{"function": map[string]any{"name": name}})
			}
			request := map[string]any{"model": "provider/model@digest/" + role + "-model", "tools": declarations}
			_, reason, model, err := g.nextStreamline(context.Background(), request, marker)
			if err != nil || reason != "tool_calls" || model != request["model"] {
				t.Fatalf("configured %s route: reason=%s model=%s error=%v", role, reason, model, err)
			}
			request["model"] = ""
			if _, _, _, err := g.nextStreamline(context.Background(), request, marker); err == nil {
				t.Fatal("accepted missing model")
			}
			request["model"] = "configured"
			request["tools"] = append(declarations, map[string]any{"function": map[string]any{"name": "unexpected_tool"}})
			if _, _, _, err := g.nextStreamline(context.Background(), request, marker); err == nil {
				t.Fatal("accepted wrong tool contract")
			}
			if role == "planner" {
				request["tools"] = declarations
				request["worker_result"] = map[string]any{"artifact": map[string]any{"namespace": "builder", "name": "copied", "revision": "worker-r1"}}
				if _, _, _, err := g.nextStreamline(context.Background(), request, marker); err != nil {
					t.Fatal(err)
				}
				message, _, _, err := g.nextStreamline(context.Background(), request, marker)
				if err != nil {
					t.Fatal(err)
				}
				function := message["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)
				var arguments struct {
					Artifacts map[string]map[string]any `json:"artifacts"`
				}
				if err := json.Unmarshal([]byte(function["arguments"].(string)), &arguments); err != nil {
					t.Fatal(err)
				}
				selected := arguments.Artifacts["copied"]
				if len(selected) != 2 || selected["namespace"] != "builder" || selected["name"] != "copied" {
					t.Fatalf("finish fixture forwarded a machine version: %+v", selected)
				}
			} else {
				request["tools"] = declarations
				request["read_result"] = map[string]any{"text": "browser-driven streamline input"}
				message, _, _, err := g.nextStreamline(context.Background(), request, marker)
				if err != nil {
					t.Fatal(err)
				}
				function := message["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)
				var arguments map[string]any
				if err := json.Unmarshal([]byte(function["arguments"].(string)), &arguments); err != nil {
					t.Fatal(err)
				}
				if arguments["text"] != "browser-driven streamline input" {
					t.Fatal("copy fixture substituted unrelated canned bytes")
				}
			}
		})
	}
}
