//go:build e2e

package uistack

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	streamlineGlobalMarker = "STREAMLINE_E2E_GLOBAL"
	streamlineWorkerMarker = "STREAMLINE_WORKER_TASK"
	dependencyReport       = `# External-Service Dependency Inventory

| Dependency | Constraint | Ecosystem | Role | Evidence |
| --- | --- | --- | --- | --- |
| fastapi | >=0.116 | Python | inbound HTTP | pyproject.toml:6 |
| httpx | >=0.28 | Python | outbound HTTP | app.py:3,16 |

The service exposes an authenticated HTTP route and calls the inventory service.
`
	projectReport = `# Project Structure and Runtime Inventory

- FastAPI application entry point: app.py:5.
- Authenticated GET /widgets/{widget_id}: app.py:8-17.
- Bearer credentials are checked before the outbound call: app.py:12-14.
- Inventory is called over HTTPS and its availability is returned: app.py:15-17.
`
	openAPIValidationReport = `# OpenAPI validation report

- Candidate and final revisions were selected through OpenAPI tools.
- Structural validation: clean.
- Vacuum available: true.
- Final valid: true.
`
)

type modelGateway struct {
	server *httptest.Server
	token  string

	mu              sync.Mutex
	domainStages    []gatewayStage
	domainStage     int
	domainStep      int
	streamlineSteps map[string]int
	calls           int
	failures        []string
	workerEntered   chan struct{}
	releaseWorker   chan struct{}
	enteredOnce     sync.Once
	releaseOnce     sync.Once
}

type gatewayStage struct {
	name  string
	tools []string
	steps []gatewayStep
}

type gatewayStep struct {
	tool      string
	arguments func(map[string]any) (map[string]any, error)
	summary   string
	artifacts map[string]artifactBinding
}

type artifactBinding struct {
	namespace string
	name      string
}

func TestModelGatewayFindsNamedInputAfterParameterBlock(t *testing.T) {
	request := map[string]any{"messages": []any{map[string]any{
		"content": "String parameters:\n{\"objective\":\"inspect\"}\n\nNamed input artifacts:\n" +
			"{\"source\":{\"namespace\":\"inputs\",\"name\":\"source\",\"revision\":\"rev-1\"}}",
	}}}
	artifact, found := stageArtifact(request, "source")
	if !found || artifact["namespace"] != "inputs" || artifact["name"] != "source" || artifact["revision"] != "rev-1" {
		t.Fatalf("named input artifact = (%v, %t)", artifact, found)
	}
}

func newModelGateway(token string) *modelGateway {
	gateway := &modelGateway{
		token: token, domainStages: openAPIGatewayStages(),
		streamlineSteps: make(map[string]int),
		workerEntered:   make(chan struct{}),
		releaseWorker:   make(chan struct{}),
	}
	gateway.server = httptest.NewServer(http.HandlerFunc(gateway.serveHTTP))
	return gateway
}

func (g *modelGateway) close() { g.server.Close() }
func (g *modelGateway) URL() string {
	return g.server.URL + "/v1"
}

func (g *modelGateway) release() {
	g.releaseOnce.Do(func() { close(g.releaseWorker) })
}

func (g *modelGateway) snapshot() (calls, completedStages int, failures []string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.calls, g.domainStage, append([]string(nil), g.failures...)
}

func (g *modelGateway) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		g.writeFailure(w, http.StatusNotFound, "unsupported model Gateway endpoint")
		return
	}
	if r.Header.Get("Authorization") != "Bearer "+g.token {
		g.writeFailure(w, http.StatusUnauthorized, "invalid model Gateway token")
		return
	}
	defer r.Body.Close()
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20))
	decoder.UseNumber()
	var request map[string]any
	if err := decoder.Decode(&request); err != nil {
		g.writeFailure(w, http.StatusBadRequest, "invalid OpenAI request")
		return
	}
	encoded, _ := json.Marshal(request)
	var message map[string]any
	var finishReason, model string
	var err error
	if strings.Contains(string(encoded), streamlineGlobalMarker) ||
		strings.Contains(string(encoded), streamlineWorkerMarker) {
		message, finishReason, model, err = g.nextStreamline(r.Context(), request, string(encoded))
	} else {
		message, finishReason, model, err = g.nextDomain(request)
	}
	if err != nil {
		g.writeFailure(w, http.StatusBadRequest, err.Error())
		return
	}
	g.mu.Lock()
	g.calls++
	call := g.calls
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id": "chatcmpl-ui-stack-" + fmt.Sprint(call), "object": "chat.completion",
		"created": time.Now().Unix(), "model": model,
		"choices": []any{map[string]any{
			"index": 0, "message": message, "finish_reason": finishReason,
		}},
		"usage": map[string]any{
			"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16,
		},
	})
}

func (g *modelGateway) nextStreamline(
	ctx context.Context, request map[string]any, payload string,
) (map[string]any, string, string, error) {
	model, ok := request["model"].(string)
	if !ok {
		return nil, "", "", errors.New("streamline request has no model")
	}
	scenario := ""
	wantTools := []string{"add_subtask", "execute_current_subtask", "finish", "list_subtasks"}
	if model == "planner-model" && strings.Contains(payload, streamlineGlobalMarker) {
		scenario = "streamline-planner"
	} else if model == "worker-model" && strings.Contains(payload, streamlineWorkerMarker) {
		scenario = "streamline-worker"
		wantTools = []string{"list_artifacts", "read_artifact", "write_artifact"}
	} else {
		return nil, "", "", fmt.Errorf("unexpected streamline model/context %q", model)
	}
	tools, err := requestToolNames(request)
	if err != nil {
		return nil, "", "", err
	}
	sort.Strings(wantTools)
	if !equalStrings(tools, wantTools) {
		return nil, "", "", fmt.Errorf("%s exposed tools %v, want %v", scenario, tools, wantTools)
	}
	g.mu.Lock()
	step := g.streamlineSteps[scenario] + 1
	g.streamlineSteps[scenario] = step
	g.mu.Unlock()
	if scenario == "streamline-worker" && step == 1 {
		g.enteredOnce.Do(func() { close(g.workerEntered) })
		select {
		case <-g.releaseWorker:
		case <-ctx.Done():
			return nil, "", "", errors.New("held streamline Worker request expired")
		case <-time.After(90 * time.Second):
			return nil, "", "", errors.New("held streamline Worker was not released")
		}
	}
	if scenario == "streamline-planner" {
		switch step {
		case 1:
			return toolCallMessage("streamline-add", "add_subtask", map[string]any{
				"objective":    streamlineWorkerMarker,
				"instructions": "Read the immutable context and copy the exact source Artifact",
			}), "tool_calls", model, nil
		case 2:
			return toolCallMessage("streamline-execute", "execute_current_subtask", map[string]any{
				"subtask_id": "0",
			}), "tool_calls", model, nil
		case 3:
			artifact, found := exactArtifactValue(request, "builder", "copied")
			if !found {
				return nil, "", "", errors.New("streamline Planner did not observe Worker Artifact")
			}
			return toolCallMessage("streamline-finish", "finish", map[string]any{
				"outcome": "succeeded", "summary": "streamline browser flow completed",
				"artifacts": map[string]any{"copied": artifact},
			}), "tool_calls", model, nil
		default:
			return nil, "", "", errors.New("streamline Planner exceeded its script")
		}
	}
	switch step {
	case 1:
		return toolCallMessage("streamline-read", "read_artifact", map[string]any{
			"namespace": "inputs", "name": "source", "revision": nil,
		}), "tool_calls", model, nil
	case 2:
		data, found := lastStringValue(request, "dataBase64")
		if !found || data == "" {
			return nil, "", "", errors.New("streamline Worker did not observe input bytes")
		}
		return toolCallMessage("streamline-write", "write_artifact", map[string]any{
			"namespace": "builder", "name": "copied", "media_type": "text/plain",
			"data_base64": data, "expected_revision": nil,
		}), "tool_calls", model, nil
	case 3:
		artifact, found := lastExactArtifact(request, "builder", "copied")
		if !found {
			return nil, "", "", errors.New("streamline Worker did not observe output revision")
		}
		result, _ := json.Marshal(map[string]any{
			"apiVersion": "contractor/v1alpha1", "outcome": "succeeded",
			"summary":   "streamline Worker copied the source",
			"artifacts": map[string]any{"copied": artifact},
		})
		return map[string]any{"role": "assistant", "content": string(result)}, "stop", model, nil
	default:
		return nil, "", "", errors.New("streamline Worker exceeded its script")
	}
}

func (g *modelGateway) nextDomain(
	request map[string]any,
) (map[string]any, string, string, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.domainStage >= len(g.domainStages) {
		return nil, "", "", errors.New("unexpected additional OpenAPI model invocation")
	}
	stage := g.domainStages[g.domainStage]
	if g.domainStep >= len(stage.steps) {
		return nil, "", "", fmt.Errorf("invalid script position for %s", stage.name)
	}
	tools, err := requestToolNames(request)
	if err != nil {
		return nil, "", "", fmt.Errorf("%s: %w", stage.name, err)
	}
	wantTools := append([]string(nil), stage.tools...)
	sort.Strings(wantTools)
	if !equalStrings(tools, wantTools) {
		return nil, "", "", fmt.Errorf("%s exposed tools %v, want %v", stage.name, tools, wantTools)
	}
	step := stage.steps[g.domainStep]
	var message map[string]any
	finishReason := "tool_calls"
	if step.tool != "" {
		arguments, buildErr := step.arguments(request)
		if buildErr != nil {
			return nil, "", "", fmt.Errorf("%s step %d: %w", stage.name, g.domainStep+1, buildErr)
		}
		message = toolCallMessage(
			fmt.Sprintf("domain-%d-%d", g.domainStage+1, g.domainStep+1),
			step.tool, arguments,
		)
	} else {
		artifacts := make(map[string]any, len(step.artifacts))
		for slot, binding := range step.artifacts {
			artifact, found := lastExactArtifact(request, binding.namespace, binding.name)
			if !found {
				return nil, "", "", fmt.Errorf(
					"%s final result has not observed %s/%s", stage.name, binding.namespace, binding.name,
				)
			}
			artifacts[slot] = artifact
		}
		result, _ := json.Marshal(map[string]any{
			"apiVersion": "contractor/v1alpha1", "outcome": "succeeded",
			"summary": step.summary, "artifacts": artifacts,
		})
		message = map[string]any{"role": "assistant", "content": string(result)}
		finishReason = "stop"
	}
	g.domainStep++
	if g.domainStep == len(stage.steps) {
		g.domainStage++
		g.domainStep = 0
	}
	model, _ := request["model"].(string)
	return message, finishReason, model, nil
}

func (g *modelGateway) writeFailure(w http.ResponseWriter, status int, message string) {
	g.mu.Lock()
	g.failures = append(g.failures, message)
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": message, "type": "invalid_request_error"},
	})
}

func openAPIGatewayStages() []gatewayStage {
	analystTools := []string{
		"list_source_files", "open_source_archive", "read_source", "read_text_artifact",
		"search_source", "write_text_artifact",
	}
	builderTools := []string{
		"get_openapi_component", "get_openapi_info", "get_openapi_path", "initialize_openapi",
		"list_openapi_components", "list_openapi_paths", "list_openapi_servers", "list_openapi_tags", "load_openapi",
		"open_source_archive", "read_source", "read_text_artifact", "search_source",
		"set_openapi_info", "set_openapi_servers", "set_openapi_tags", "upsert_openapi_component",
		"upsert_openapi_path", "validate_openapi", "list_source_files",
	}
	validatorTools := []string{
		"get_openapi_component", "get_openapi_info", "get_openapi_path",
		"list_openapi_components", "list_openapi_paths", "list_openapi_servers", "list_openapi_tags", "load_openapi",
		"open_source_archive", "read_source", "read_text_artifact", "remove_openapi_component",
		"remove_openapi_path", "search_source", "set_openapi_info", "set_openapi_servers", "set_openapi_tags",
		"upsert_openapi_component", "upsert_openapi_path", "validate_openapi", "write_text_artifact",
	}
	return []gatewayStage{
		discoveryGatewayStage("openapi/dependency_discovery", analystTools, true),
		discoveryGatewayStage("openapi/project_discovery", analystTools, false),
		openAPIBuildGatewayStage(builderTools),
		openAPIValidateGatewayStage(validatorTools),
	}
}

func discoveryGatewayStage(name string, tools []string, dependency bool) gatewayStage {
	steps := []gatewayStep{toolGatewayStep("open_source_archive", stageRefArguments("source", nil))}
	if dependency {
		steps = append(steps, toolGatewayStep("list_source_files", fixedArguments(map[string]any{
			"pattern": "**/*", "offset": 0, "limit": 50,
		})))
	} else {
		steps = append(steps, toolGatewayStep("read_text_artifact", stageRefArguments("dependency_report", nil)))
	}
	steps = append(steps, toolGatewayStep("read_source", fixedArguments(map[string]any{
		"path": "app.py", "start_line": 1, "max_lines": 100,
	})))
	if dependency {
		steps = append(steps,
			toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{
				"name": "dependencies", "text": dependencyReport,
				"media_type": "text/markdown", "expected_revision": nil,
			})),
			finalGatewayStep("Dependency inventory published", map[string]artifactBinding{
				"dependency_report": {namespace: "analysis", name: "dependencies"},
			}),
		)
	} else {
		steps = append(steps,
			toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{
				"name": "project", "text": projectReport,
				"media_type": "text/markdown", "expected_revision": nil,
			})),
			finalGatewayStep("Project inventory published", map[string]artifactBinding{
				"project_report": {namespace: "analysis", name: "project"},
			}),
		)
	}
	return gatewayStage{name: name, tools: tools, steps: steps}
}

func openAPIBuildGatewayStage(tools []string) gatewayStage {
	return gatewayStage{name: "openapi/openapi_build", tools: tools, steps: []gatewayStep{
		toolGatewayStep("open_source_archive", stageRefArguments("source", nil)),
		toolGatewayStep("read_text_artifact", stageRefArguments("dependency_report", nil)),
		toolGatewayStep("read_text_artifact", stageRefArguments("project_report", nil)),
		toolGatewayStep("read_source", fixedArguments(map[string]any{
			"path": "app.py", "start_line": 1, "max_lines": 100,
		})),
		toolGatewayStep("load_openapi", stageRefArguments("existing_openapi", map[string]any{
			"target_name": "openapi", "expected_revision": nil,
		})),
		toolGatewayStep("upsert_openapi_component", fixedArguments(map[string]any{
			"section": "schemas", "name": "WidgetResponse",
			"component": map[string]any{
				"type": "object", "required": []any{"id", "available"},
				"properties": map[string]any{
					"id": map[string]any{"type": "string"}, "available": map[string]any{"type": "boolean"},
				},
			},
			"evidence_files": []any{"app.py"},
		})),
		toolGatewayStep("upsert_openapi_path", fixedArguments(map[string]any{
			"path": "/widgets/{widget_id}",
			"path_item": map[string]any{"get": map[string]any{
				"operationId": "getWidget", "summary": "Get widget availability",
				"parameters": []any{map[string]any{
					"name": "widget_id", "in": "path", "required": true,
					"schema": map[string]any{"type": "string"},
				}},
				"responses": map[string]any{
					"200": map[string]any{
						"description": "Widget availability",
						"content": map[string]any{"application/json": map[string]any{
							"schema": map[string]any{"$ref": "#/components/schemas/WidgetResponse"},
						}},
					},
					"401": map[string]any{"description": "Invalid bearer token"},
				},
			}},
			"evidence_files": []any{"app.py"},
		})),
		toolGatewayStep("validate_openapi", fixedArguments(nil)),
		finalGatewayStep("OpenAPI candidate built and validated", map[string]artifactBinding{
			"openapi": {namespace: "openapi", name: "openapi"},
		}),
	}}
}

func openAPIValidateGatewayStage(tools []string) gatewayStage {
	return gatewayStage{name: "openapi/openapi_validate", tools: tools, steps: []gatewayStep{
		toolGatewayStep("open_source_archive", stageRefArguments("source", nil)),
		toolGatewayStep("read_text_artifact", stageRefArguments("dependency_report", nil)),
		toolGatewayStep("read_text_artifact", stageRefArguments("project_report", nil)),
		toolGatewayStep("load_openapi", stageRefArguments("openapi_candidate", map[string]any{
			"target_name": "openapi", "expected_revision": nil,
		})),
		toolGatewayStep("validate_openapi", fixedArguments(nil)),
		toolGatewayStep("get_openapi_path", fixedArguments(map[string]any{
			"path": "/widgets/{widget_id}",
		})),
		toolGatewayStep("validate_openapi", fixedArguments(nil)),
		toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{
			"name": "validation-report", "text": openAPIValidationReport,
			"media_type": "text/markdown", "expected_revision": nil,
		})),
		finalGatewayStep("OpenAPI candidate is clean", map[string]artifactBinding{
			"openapi":           {namespace: "openapi", name: "openapi"},
			"validation_report": {namespace: "openapi", name: "validation-report"},
		}),
	}}
}

func toolGatewayStep(
	name string, arguments func(map[string]any) (map[string]any, error),
) gatewayStep {
	return gatewayStep{tool: name, arguments: arguments}
}

func finalGatewayStep(summary string, artifacts map[string]artifactBinding) gatewayStep {
	return gatewayStep{summary: summary, artifacts: artifacts}
}

func fixedArguments(value map[string]any) func(map[string]any) (map[string]any, error) {
	return func(map[string]any) (map[string]any, error) {
		result := make(map[string]any, len(value))
		for key, item := range value {
			result[key] = item
		}
		return result, nil
	}
}

func stageRefArguments(
	slot string, extra map[string]any,
) func(map[string]any) (map[string]any, error) {
	return func(request map[string]any) (map[string]any, error) {
		artifact, found := stageArtifact(request, slot)
		if !found {
			return nil, fmt.Errorf("exact Stage artifact %q is absent", slot)
		}
		result := make(map[string]any, 3+len(extra))
		for _, key := range []string{"namespace", "name", "revision"} {
			result[key] = artifact[key]
		}
		for key, value := range extra {
			result[key] = value
		}
		return result, nil
	}
}

func stageArtifact(value any, slot string) (map[string]any, bool) {
	values := make([]map[string]any, 0)
	collect := func(candidate any) {
		artifact, ok := candidate.(map[string]any)
		if !ok {
			return
		}
		namespace, namespaceOK := artifact["namespace"].(string)
		name, nameOK := artifact["name"].(string)
		revision, revisionOK := artifact["revision"].(string)
		if namespaceOK && namespace != "" && nameOK && name != "" && revisionOK && revision != "" {
			values = append(values, map[string]any{
				"namespace": namespace, "name": name, "revision": revision,
			})
		}
	}
	walkJSON(value, func(object map[string]any) {
		// Legacy prompts embedded the complete StageContentRequest under an
		// `artifacts` field. Runtime-owned result projection now exposes only a
		// model-facing map of named inputs, so accept that direct shape as well.
		if artifacts, ok := object["artifacts"].(map[string]any); ok {
			collect(artifacts[slot])
		}
		collect(object[slot])
	})
	if len(values) == 0 {
		return nil, false
	}
	return values[len(values)-1], true
}

func toolCallMessage(id, name string, arguments map[string]any) map[string]any {
	encoded, _ := json.Marshal(arguments)
	return map[string]any{
		"role": "assistant", "content": nil,
		"tool_calls": []any{map[string]any{
			"id": id, "type": "function",
			"function": map[string]any{"name": name, "arguments": string(encoded)},
		}},
	}
}

func requestToolNames(request map[string]any) ([]string, error) {
	raw, ok := request["tools"].([]any)
	if !ok || len(raw) == 0 {
		return nil, errors.New("model-visible tool set is absent")
	}
	names := make([]string, 0, len(raw))
	for _, item := range raw {
		tool, ok := item.(map[string]any)
		if !ok {
			return nil, errors.New("invalid tool declaration")
		}
		function, ok := tool["function"].(map[string]any)
		if !ok {
			return nil, errors.New("invalid function declaration")
		}
		name, ok := function["name"].(string)
		if !ok || name == "" {
			return nil, errors.New("unnamed function declaration")
		}
		names = append(names, name)
	}
	sort.Strings(names)
	for index := 1; index < len(names); index++ {
		if names[index] == names[index-1] {
			return nil, fmt.Errorf("duplicate model-visible tool %q", names[index])
		}
	}
	return names, nil
}

func lastStringValue(value any, key string) (string, bool) {
	values := make([]string, 0)
	walkJSON(value, func(object map[string]any) {
		if current, ok := object[key].(string); ok {
			values = append(values, current)
		}
	})
	if len(values) == 0 {
		return "", false
	}
	return values[len(values)-1], true
}

func lastExactArtifact(value any, namespace, name string) (map[string]any, bool) {
	values := make([]map[string]any, 0)
	walkJSON(value, func(object map[string]any) {
		artifact, ok := object["artifact"].(map[string]any)
		if !ok || artifact["namespace"] != namespace || artifact["name"] != name {
			return
		}
		if revision, ok := artifact["revision"].(string); ok && revision != "" {
			values = append(values, artifact)
		}
	})
	if len(values) == 0 {
		return nil, false
	}
	result := make(map[string]any, len(values[len(values)-1]))
	for key, current := range values[len(values)-1] {
		result[key] = current
	}
	return result, true
}

func exactArtifactValue(value any, namespace, name string) (map[string]any, bool) {
	var result map[string]any
	walkJSON(value, func(object map[string]any) {
		if object["namespace"] != namespace || object["name"] != name {
			return
		}
		if revision, ok := object["revision"].(string); !ok || revision == "" {
			return
		}
		result = make(map[string]any, len(object))
		for key, current := range object {
			result[key] = current
		}
	})
	return result, result != nil
}

func walkJSON(value any, visit func(map[string]any)) {
	switch current := value.(type) {
	case map[string]any:
		visit(current)
		for _, child := range current {
			walkJSON(child, visit)
		}
	case []any:
		for _, child := range current {
			walkJSON(child, visit)
		}
	case string:
		var decoded any
		if json.Unmarshal([]byte(current), &decoded) == nil {
			walkJSON(decoded, visit)
			return
		}
		for index, character := range current {
			if character != '{' && character != '[' {
				continue
			}
			decoder := json.NewDecoder(strings.NewReader(current[index:]))
			decoder.UseNumber()
			if decoder.Decode(&decoded) == nil {
				walkJSON(decoded, visit)
			}
		}
	}
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}
