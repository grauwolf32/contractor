//go:build e2e

package e2e

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
	dependencyReport = `# External-Service Dependency Inventory

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
	likeC4ValidationReport = `# LikeC4 validation report

- Candidate and final revisions were selected through LikeC4 tools.
- Direct LikeC4 CLI available: true.
- Initial issues: 0.
- Final issues: 0.
- Final valid: true.
`
	likeC4ModelBlock = `
model {
  client = actor 'API Client'
  widgets = system 'Widget Service' {
    api = container 'FastAPI API' {
      #public
      description '''Authenticated widget endpoint. Evidence: app.py:8-17.'''
    }
  }
  inventory = external 'Inventory Service' {
    #external
    description '''Remote inventory API. Evidence: app.py:15-16.'''
  }
  client -> widgets.api 'GET /widgets/{widget_id} with bearer token'
  widgets.api -[calls]-> inventory 'HTTPS GET with service credentials'
}
`
	likeC4ViewsBlock = `
views {
  view index {
    include *
    autoLayout LeftRight
  }
}
`
)

func TestDomainGatewayFindsNamedInputAfterParameterBlock(t *testing.T) {
	request := map[string]any{"messages": []any{map[string]any{
		"content": "String parameters:\n{\"objective\":\"inspect\"}\n\nNamed input artifacts:\n" +
			"{\"source\":{\"namespace\":\"inputs\",\"name\":\"source\",\"revision\":\"rev-1\"}}",
	}}}
	artifact, found := stageArtifact(request, "source")
	if !found || artifact["namespace"] != "inputs" || artifact["name"] != "source" || artifact["revision"] != "rev-1" {
		t.Fatalf("named input artifact = (%v, %t)", artifact, found)
	}
}

type domainGateway struct {
	server *httptest.Server
	token  string

	mu           sync.Mutex
	stages       []domainGatewayStage
	stageIndex   int
	stepIndex    int
	calls        int
	failures     []string
	observations []domainGatewayObservation

	blockNext  bool
	blocked    chan struct{}
	release    chan struct{}
	releaseOne sync.Once
}

type domainGatewayStage struct {
	name  string
	tools []string
	steps []domainGatewayStep
}

type domainGatewayStep struct {
	tool      string
	arguments func(map[string]any) (map[string]any, error)
	validate  func(map[string]any) error
	summary   string
	artifacts map[string]domainArtifactBinding
	plain     bool
	outcome   string
	errorCode string
	retryable bool
}

type domainArtifactBinding struct {
	namespace string
	name      string
}

type domainGatewayObservation struct {
	Stage string
	Step  int
	Tool  string
	Tools []string
}

func newDomainGateway(token string) *domainGateway {
	gateway := &domainGateway{token: token, stages: domainGatewayStages()}
	gateway.server = httptest.NewServer(http.HandlerFunc(gateway.serveHTTP))
	return gateway
}

func (g *domainGateway) close() {
	g.releaseBlockedRequest()
	g.server.Close()
}

func (g *domainGateway) URL() string { return g.server.URL + "/v1" }

func (g *domainGateway) Calls() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.calls
}

func (g *domainGateway) Failures() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.failures...)
}

func (g *domainGateway) Observations() []domainGatewayObservation {
	g.mu.Lock()
	defer g.mu.Unlock()
	result := make([]domainGatewayObservation, len(g.observations))
	for index, observation := range g.observations {
		result[index] = observation
		result[index].Tools = append([]string(nil), observation.Tools...)
	}
	return result
}

func (g *domainGateway) CompletedStages() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.stageIndex
}

func newBlockedDomainGateway(token string, stages []domainGatewayStage) *domainGateway {
	gateway := &domainGateway{
		token: token, stages: stages, blockNext: true,
		blocked: make(chan struct{}), release: make(chan struct{}),
	}
	gateway.server = httptest.NewServer(http.HandlerFunc(gateway.serveHTTP))
	return gateway
}

func (g *domainGateway) blockedRequest() <-chan struct{} { return g.blocked }

func (g *domainGateway) releaseBlockedRequest() {
	if g.release != nil {
		g.releaseOne.Do(func() { close(g.release) })
	}
}

func (g *domainGateway) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		g.writeFailure(w, http.StatusNotFound, "unsupported fake gateway endpoint")
		return
	}
	if r.Header.Get("Authorization") != "Bearer "+g.token {
		g.writeFailure(w, http.StatusUnauthorized, "invalid fake gateway token")
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
	if !g.awaitInitialRelease(r.Context()) {
		return
	}

	message, finishReason, call, err := g.next(request)
	if err != nil {
		g.writeFailure(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":      fmt.Sprintf("chatcmpl-domain-e2e-%d", call),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   "worker-model",
		"choices": []any{map[string]any{
			"index": 0, "message": message, "finish_reason": finishReason,
		}},
		"usage": map[string]any{
			"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16,
		},
	})
}

func (g *domainGateway) awaitInitialRelease(ctx context.Context) bool {
	g.mu.Lock()
	if !g.blockNext {
		g.mu.Unlock()
		return true
	}
	g.blockNext = false
	blocked, release := g.blocked, g.release
	g.mu.Unlock()
	close(blocked)
	select {
	case <-release:
		return true
	case <-ctx.Done():
		return false
	}
}

func (g *domainGateway) next(request map[string]any) (map[string]any, string, int, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.stageIndex >= len(g.stages) {
		return nil, "", 0, errors.New("unexpected additional LLM invocation")
	}
	stage := g.stages[g.stageIndex]
	if g.stepIndex >= len(stage.steps) {
		return nil, "", 0, fmt.Errorf("invalid script position for %s", stage.name)
	}
	tools, err := requestToolNames(request)
	if err != nil {
		return nil, "", 0, fmt.Errorf("%s: %w", stage.name, err)
	}
	wantTools := append([]string(nil), stage.tools...)
	sort.Strings(wantTools)
	if !equalStringSlices(tools, wantTools) {
		return nil, "", 0, fmt.Errorf("%s exposed tools %v, want %v", stage.name, tools, wantTools)
	}

	stepNumber := g.stepIndex + 1
	step := stage.steps[g.stepIndex]
	if step.validate != nil {
		if validateErr := step.validate(request); validateErr != nil {
			return nil, "", 0, fmt.Errorf("%s step %d: %w", stage.name, stepNumber, validateErr)
		}
	}
	var message map[string]any
	finishReason := "tool_calls"
	if step.tool != "" {
		arguments, buildErr := step.arguments(request)
		if buildErr != nil {
			return nil, "", 0, fmt.Errorf("%s step %d: %w", stage.name, stepNumber, buildErr)
		}
		message = toolCallMessage(
			fmt.Sprintf("domain-%d-%d", g.stageIndex+1, stepNumber), step.tool, arguments,
		)
	} else {
		artifacts := make(map[string]any, len(step.artifacts))
		for slot, binding := range step.artifacts {
			artifact, ok := lastExactArtifact(request, binding.namespace, binding.name)
			if !ok {
				return nil, "", 0, fmt.Errorf(
					"%s final result has not observed %s/%s", stage.name, binding.namespace, binding.name,
				)
			}
			artifacts[slot] = artifact
		}
		outcome := step.outcome
		if outcome == "" {
			outcome = "succeeded"
		}
		if step.plain {
			if outcome != "succeeded" || step.errorCode != "" {
				return nil, "", 0, errors.New("plain final response must succeed")
			}
			message = map[string]any{"role": "assistant", "content": step.summary}
		} else {
			resultBody := map[string]any{
				"apiVersion": "contractor/v1alpha1",
				"outcome":    outcome,
				"summary":    step.summary,
				"artifacts":  artifacts,
			}
			if step.errorCode != "" {
				resultBody["error"] = map[string]any{
					"code": step.errorCode, "message": "deterministic retry request",
					"retryable": step.retryable,
				}
			}
			result, marshalErr := json.Marshal(resultBody)
			if marshalErr != nil {
				return nil, "", 0, marshalErr
			}
			message = map[string]any{"role": "assistant", "content": string(result)}
		}
		finishReason = "stop"
	}

	g.calls++
	call := g.calls
	tool := step.tool
	if tool == "" {
		tool = "<final>"
	}
	g.observations = append(g.observations, domainGatewayObservation{
		Stage: stage.name, Step: stepNumber, Tool: tool, Tools: tools,
	})
	g.stepIndex++
	if g.stepIndex == len(stage.steps) {
		g.stageIndex++
		g.stepIndex = 0
	}
	return message, finishReason, call, nil
}

func (g *domainGateway) writeFailure(w http.ResponseWriter, status int, message string) {
	g.mu.Lock()
	g.failures = append(g.failures, message)
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": message, "type": "invalid_request_error"},
	})
}

func domainGatewayStages() []domainGatewayStage {
	analystTools := []string{
		"list_source_files", "open_source_archive", "read_source", "read_text_artifact",
		"search_source", "write_text_artifact",
	}
	openAPIBuilderTools := []string{
		"get_openapi_component", "get_openapi_info", "get_openapi_path", "initialize_openapi",
		"list_openapi_components", "list_openapi_paths", "list_openapi_servers", "list_openapi_tags", "load_openapi",
		"open_source_archive", "read_source", "read_text_artifact", "search_source",
		"set_openapi_info", "set_openapi_servers", "set_openapi_tags", "upsert_openapi_component",
		"upsert_openapi_path", "validate_openapi", "list_source_files",
	}
	openAPIValidatorTools := []string{
		"get_openapi_component", "get_openapi_info", "get_openapi_path",
		"list_openapi_components", "list_openapi_paths", "list_openapi_servers", "list_openapi_tags", "load_openapi",
		"open_source_archive", "read_source", "read_text_artifact", "remove_openapi_component",
		"remove_openapi_path", "search_source", "set_openapi_info", "set_openapi_servers", "set_openapi_tags",
		"upsert_openapi_component", "upsert_openapi_path", "validate_openapi", "write_text_artifact",
	}
	likeC4BuilderTools := []string{
		"append_likec4", "list_source_files", "load_likec4", "open_source_archive",
		"read_likec4", "read_source", "read_text_artifact", "replace_likec4",
		"search_source", "validate_likec4", "write_likec4",
	}
	likeC4ValidatorTools := []string{
		"append_likec4", "load_likec4", "open_source_archive", "read_likec4", "read_source",
		"read_text_artifact", "replace_likec4", "search_source", "validate_likec4",
		"write_likec4", "write_text_artifact",
	}

	stages := make([]domainGatewayStage, 0, 8)
	stages = append(stages,
		discoveryGatewayStage("openapi/dependency_discovery", analystTools, true),
		discoveryGatewayStage("openapi/project_discovery", analystTools, false),
		openAPIBuildGatewayStage(openAPIBuilderTools),
		openAPIValidateGatewayStage(openAPIValidatorTools),
		discoveryGatewayStage("likec4/dependency_discovery", analystTools, true),
		discoveryGatewayStage("likec4/project_discovery", analystTools, false),
		likeC4BuildGatewayStage(likeC4BuilderTools),
		likeC4ValidateGatewayStage(likeC4ValidatorTools),
	)
	return stages
}

func discoveryGatewayStage(name string, tools []string, dependency bool) domainGatewayStage {
	steps := []domainGatewayStep{
		toolGatewayStep("open_source_archive", stageRefArguments("source", nil)),
	}
	if dependency {
		steps = append(steps,
			toolGatewayStep("list_source_files", fixedArguments(map[string]any{
				"pattern": "**/*", "offset": 0, "limit": 50,
			})),
		)
	} else {
		steps = append(steps,
			toolGatewayStep("read_text_artifact", stageRefArguments("dependency_report", nil)),
		)
	}
	steps = append(steps,
		toolGatewayStep("read_source", fixedArguments(map[string]any{
			"path": "app.py", "start_line": 1, "max_lines": 100,
		})),
	)
	if dependency {
		steps = append(steps,
			toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{
				"name": "dependencies", "text": dependencyReport, "media_type": "text/markdown",
				"expected_revision": nil,
			})),
			finalGatewayStep("Dependency inventory published", map[string]domainArtifactBinding{
				"dependency_report": {namespace: "analysis", name: "dependencies"},
			}),
		)
	} else {
		steps = append(steps,
			toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{
				"name": "project", "text": projectReport, "media_type": "text/markdown",
				"expected_revision": nil,
			})),
			finalGatewayStep("Project inventory published", map[string]domainArtifactBinding{
				"project_report": {namespace: "analysis", name: "project"},
			}),
		)
	}
	return domainGatewayStage{name: name, tools: tools, steps: steps}
}

func openAPIBuildGatewayStage(tools []string) domainGatewayStage {
	return domainGatewayStage{name: "openapi/openapi_build", tools: tools, steps: []domainGatewayStep{
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
					"id":        map[string]any{"type": "string"},
					"available": map[string]any{"type": "boolean"},
				},
			},
			"evidence_files": []any{"app.py"},
		})),
		toolGatewayStep("upsert_openapi_path", fixedArguments(map[string]any{
			"path": "/widgets/{widget_id}",
			"path_item": map[string]any{
				"get": map[string]any{
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
				},
			},
			"evidence_files": []any{"app.py"},
		})),
		toolGatewayStep("validate_openapi", fixedArguments(nil)),
		finalGatewayStep("OpenAPI candidate built and validated", map[string]domainArtifactBinding{
			"openapi": {namespace: "openapi", name: "openapi"},
		}),
	}}
}

func openAPIValidateGatewayStage(tools []string) domainGatewayStage {
	return domainGatewayStage{name: "openapi/openapi_validate", tools: tools, steps: []domainGatewayStep{
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
		finalGatewayStep("OpenAPI candidate is clean", map[string]domainArtifactBinding{
			"openapi":           {namespace: "openapi", name: "openapi"},
			"validation_report": {namespace: "openapi", name: "validation-report"},
		}),
	}}
}

func likeC4BuildGatewayStage(tools []string) domainGatewayStage {
	return domainGatewayStage{name: "likec4/likec4_build", tools: tools, steps: []domainGatewayStep{
		toolGatewayStep("open_source_archive", stageRefArguments("source", nil)),
		toolGatewayStep("read_text_artifact", stageRefArguments("dependency_report", nil)),
		toolGatewayStep("read_text_artifact", stageRefArguments("project_report", nil)),
		toolGatewayStep("read_source", fixedArguments(map[string]any{
			"path": "app.py", "start_line": 1, "max_lines": 100,
		})),
		toolGatewayStep("load_likec4", stageRefArguments("existing_likec4", map[string]any{
			"target_name": "architecture", "expected_revision": nil,
		})),
		toolGatewayStep("validate_likec4", fixedArguments(nil)),
		toolGatewayStep("append_likec4", fixedArguments(map[string]any{"content": likeC4ModelBlock})),
		toolGatewayStep("validate_likec4", fixedArguments(nil)),
		toolGatewayStep("append_likec4", fixedArguments(map[string]any{"content": likeC4ViewsBlock})),
		toolGatewayStep("validate_likec4", fixedArguments(nil)),
		finalGatewayStep("LikeC4 model built in validated phases", map[string]domainArtifactBinding{
			"architecture": {namespace: "likec4", name: "architecture"},
		}),
	}}
}

func likeC4ValidateGatewayStage(tools []string) domainGatewayStage {
	return domainGatewayStage{name: "likec4/likec4_validate", tools: tools, steps: []domainGatewayStep{
		toolGatewayStep("open_source_archive", stageRefArguments("source", nil)),
		toolGatewayStep("read_text_artifact", stageRefArguments("dependency_report", nil)),
		toolGatewayStep("read_text_artifact", stageRefArguments("project_report", nil)),
		toolGatewayStep("load_likec4", stageRefArguments("architecture_candidate", map[string]any{
			"target_name": "architecture", "expected_revision": nil,
		})),
		toolGatewayStep("validate_likec4", fixedArguments(nil)),
		toolGatewayStep("read_likec4", fixedArguments(map[string]any{
			"start_line": 1, "max_lines": 200,
		})),
		toolGatewayStep("validate_likec4", fixedArguments(nil)),
		toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{
			"name": "validation-report", "text": likeC4ValidationReport,
			"media_type": "text/markdown", "expected_revision": nil,
		})),
		finalGatewayStep("LikeC4 candidate is clean", map[string]domainArtifactBinding{
			"architecture":      {namespace: "likec4", name: "architecture"},
			"validation_report": {namespace: "likec4", name: "validation-report"},
		}),
	}}
}

func toolGatewayStep(
	name string,
	arguments func(map[string]any) (map[string]any, error),
) domainGatewayStep {
	return domainGatewayStep{tool: name, arguments: arguments}
}

func finalGatewayStep(
	summary string,
	artifacts map[string]domainArtifactBinding,
) domainGatewayStep {
	return domainGatewayStep{summary: summary, artifacts: artifacts}
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
	slot string,
	extra map[string]any,
) func(map[string]any) (map[string]any, error) {
	return func(request map[string]any) (map[string]any, error) {
		artifact, ok := stageArtifact(request, slot)
		if !ok {
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
		if !namespaceOK || namespace == "" || !nameOK || name == "" || !revisionOK || revision == "" {
			return
		}
		values = append(values, map[string]any{
			"namespace": namespace, "name": name, "revision": revision,
		})
	}
	walkDomainJSON(value, func(object map[string]any) {
		// Accept both the legacy StageContentRequest wrapper and the current
		// model-facing map rendered below `Named input artifacts`.
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

func walkDomainJSON(value any, visit func(map[string]any)) {
	switch current := value.(type) {
	case map[string]any:
		visit(current)
		for _, child := range current {
			walkDomainJSON(child, visit)
		}
	case []any:
		for _, child := range current {
			walkDomainJSON(child, visit)
		}
	case string:
		var decoded any
		if json.Unmarshal([]byte(current), &decoded) == nil {
			walkDomainJSON(decoded, visit)
			return
		}
		for index, character := range current {
			if character != '{' && character != '[' {
				continue
			}
			decoder := json.NewDecoder(strings.NewReader(current[index:]))
			decoder.UseNumber()
			if decoder.Decode(&decoded) == nil {
				walkDomainJSON(decoded, visit)
			}
		}
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

func equalStringSlices(left, right []string) bool {
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
