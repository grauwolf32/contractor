//go:build e2e

package e2e

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	streamlineGlobalMarker            = "STREAMLINE_E2E_GLOBAL"
	routerGlobalMarker                = "ROUTER_E2E_GLOBAL"
	escalationGlobalMarker            = "ESCALATION_E2E_GLOBAL"
	streamlineWorkerMarker            = "STREAMLINE_WORKER_TASK"
	routerWorkerMarker                = "ROUTER_REVIEWER_TASK"
	escalationWorkerMarker            = "ESCALATED_WORKER_TASK"
	observationStreamlineGlobalMarker = "WORKER_OBSERVATIONS_STREAMLINE"
	observationRouterGlobalMarker     = "WORKER_OBSERVATIONS_ROUTER"
	observationBuilderMarker          = "OBSERVATION_BUILDER_TASK"
	observationReviewerMarker         = "OBSERVATION_REVIEWER_TASK"
)

type routingGateway struct {
	server *httptest.Server
	token  string

	mu           sync.Mutex
	steps        map[string]int
	modelCalls   map[string]int
	observations []routingGatewayObservation
	stateUsages  map[string]workerObservationUsage
	failures     []string
}

type workerObservationUsage struct {
	ModelCalls  int64
	ToolCalls   int64
	TotalTokens int64
	Tools       map[string]int64
}

type routingGatewayObservation struct {
	Scenario string
	Model    string
	Step     int
	Tool     string
}

func newRoutingGateway(token string) *routingGateway {
	result := &routingGateway{
		token: token, steps: map[string]int{}, modelCalls: map[string]int{},
		stateUsages: map[string]workerObservationUsage{},
	}
	result.server = httptest.NewServer(http.HandlerFunc(result.serveHTTP))
	return result
}

func (g *routingGateway) close() { g.server.Close() }

func (g *routingGateway) URL() string { return g.server.URL + "/v1" }

func (g *routingGateway) Failures() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.failures...)
}

func (g *routingGateway) ModelCalls(modelName string) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.modelCalls[modelName]
}

func (g *routingGateway) Observations() []routingGatewayObservation {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]routingGatewayObservation(nil), g.observations...)
}

func (g *routingGateway) StateUsages() map[string]workerObservationUsage {
	g.mu.Lock()
	defer g.mu.Unlock()
	result := make(map[string]workerObservationUsage, len(g.stateUsages))
	for key, usage := range g.stateUsages {
		usage.Tools = cloneStringInt64Map(usage.Tools)
		result[key] = usage
	}
	return result
}

func (g *routingGateway) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		g.writeFailure(w, http.StatusNotFound, "unsupported routing Gateway endpoint")
		return
	}
	if r.Header.Get("Authorization") != "Bearer "+g.token {
		g.writeFailure(w, http.StatusUnauthorized, "invalid routing Gateway token")
		return
	}
	defer r.Body.Close()
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20))
	decoder.UseNumber()
	var request map[string]any
	if err := decoder.Decode(&request); err != nil {
		g.writeFailure(w, http.StatusBadRequest, "invalid routing Gateway request")
		return
	}
	message, finishReason, modelName, call, err := g.next(request)
	if err != nil {
		g.writeFailure(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":      fmt.Sprintf("chatcmpl-routing-%d", call),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   modelName,
		"choices": []any{map[string]any{
			"index": 0, "message": message, "finish_reason": finishReason,
		}},
		"usage": map[string]any{
			"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18,
		},
	})
}

func (g *routingGateway) next(
	request map[string]any,
) (message map[string]any, finishReason string, modelName string, call int, err error) {
	modelName, ok := request["model"].(string)
	if !ok || strings.TrimSpace(modelName) == "" {
		return nil, "", "", 0, fmt.Errorf("routing request has no model")
	}
	encoded, marshalErr := json.Marshal(request)
	if marshalErr != nil {
		return nil, "", "", 0, marshalErr
	}
	payload := string(encoded)
	scenario, mode, worker, detectErr := routingScenario(modelName, payload)
	if detectErr != nil {
		return nil, "", "", 0, detectErr
	}
	if worker {
		if !strings.Contains(payload, mode) || !strings.Contains(payload, "inputs") ||
			!strings.Contains(payload, "source") {
			return nil, "", "", 0, fmt.Errorf("%s Worker did not receive complete StageContext", scenario)
		}
	}
	tools, toolsErr := requestToolNames(request)
	if toolsErr != nil {
		return nil, "", "", 0, fmt.Errorf("%s: %w", scenario, toolsErr)
	}
	wantTools := []string{
		"add_subtask", "execute_current_subtask", "finish", "get_worker_tool_usage", "list_subtasks",
	}
	if worker {
		wantTools = []string{"read_artifact", "write_artifact"}
	}
	if strings.HasPrefix(scenario, "observation-") {
		wantTools = []string{
			"add_subtask", "execute_current_subtask", "finish", "get_worker_tool_usage",
			"get_workspace_coverage", "list_read_files", "list_subtasks", "list_unread_files",
		}
		if worker && strings.Contains(scenario, "-builder-worker") {
			wantTools = []string{"edit", "grep", "read_file", "write_text_artifact"}
		} else if worker {
			wantTools = []string{"read_file", "write_text_artifact"}
		}
	}
	sort.Strings(wantTools)
	if !equalStringSlices(tools, wantTools) {
		return nil, "", "", 0, fmt.Errorf("%s exposed tools %v, want %v", scenario, tools, wantTools)
	}
	if scenario == "router-planner" {
		if !strings.Contains(payload, "worker_name") || !strings.Contains(payload, "builder") ||
			!strings.Contains(payload, "reviewer") ||
			!strings.Contains(payload, "Reviews the exact input") ||
			strings.Contains(payload, "/private/v1/allocations/") {
			return nil, "", "", 0, fmt.Errorf("Router model context lacks logical schema or leaks placement")
		}
	} else if strings.Contains(scenario, "-router-") && !worker {
		if !strings.Contains(payload, "worker_name") || !strings.Contains(payload, "builder") ||
			!strings.Contains(payload, "reviewer") ||
			strings.Contains(payload, "/private/v1/allocations/") {
			return nil, "", "", 0, fmt.Errorf("observation Router context lacks logical schema or leaks placement")
		}
	}

	g.mu.Lock()
	step := g.steps[scenario] + 1
	g.steps[scenario] = step
	g.modelCalls[modelName]++
	call = 0
	for _, count := range g.modelCalls {
		call += count
	}
	g.mu.Unlock()

	message, finishReason, tool, buildErr := routingResponse(scenario, step, request)
	if buildErr != nil {
		return nil, "", "", 0, buildErr
	}
	g.mu.Lock()
	if usage, workerName, ok := completedObservationUsage(scenario, step, request); ok {
		g.stateUsages[scenario+"/"+workerName] = usage
	}
	g.observations = append(g.observations, routingGatewayObservation{
		Scenario: scenario, Model: modelName, Step: step, Tool: tool,
	})
	g.mu.Unlock()
	return message, finishReason, modelName, call, nil
}

func routingScenario(modelName, payload string) (scenario string, mode string, worker bool, err error) {
	switch modelName {
	case "planner-model":
		switch {
		case strings.Contains(payload, observationStreamlineGlobalMarker):
			return "observation-streamline-initial-planner", "observations-streamline", false, nil
		case strings.Contains(payload, observationRouterGlobalMarker) && strings.Contains(payload, "observations-reuse"):
			return "observation-router-reuse-planner", "observations-reuse", false, nil
		case strings.Contains(payload, observationRouterGlobalMarker):
			return "observation-router-initial-planner", "observations-initial", false, nil
		case strings.Contains(payload, streamlineGlobalMarker):
			return "streamline-planner", "streamline-strict", false, nil
		case strings.Contains(payload, routerGlobalMarker):
			return "router-planner", "router-strict", false, nil
		case strings.Contains(payload, escalationGlobalMarker):
			return "base-escalation-planner", "escalation-strict", false, nil
		}
	case "strong-planner-model":
		if strings.Contains(payload, escalationGlobalMarker) {
			return "strong-escalation-planner", "escalation-strict", false, nil
		}
	case "worker-model":
		switch {
		case strings.Contains(payload, observationBuilderMarker) && strings.Contains(payload, "observations-reuse"):
			return "observation-router-reuse-builder-worker", "observations-reuse", true, nil
		case strings.Contains(payload, observationBuilderMarker) && strings.Contains(payload, "observations-streamline"):
			return "observation-streamline-initial-builder-worker", "observations-streamline", true, nil
		case strings.Contains(payload, observationBuilderMarker):
			return "observation-router-initial-builder-worker", "observations-initial", true, nil
		case strings.Contains(payload, streamlineWorkerMarker):
			return "streamline-worker", "streamline-strict", true, nil
		case strings.Contains(payload, routerWorkerMarker):
			return "router-builder-unexpected", "router-strict", true, fmt.Errorf("Router builder Worker was invoked")
		}
	case "reviewer-worker-model":
		if strings.Contains(payload, observationReviewerMarker) && strings.Contains(payload, "observations-reuse") {
			return "observation-router-reuse-reviewer-worker", "observations-reuse", true, nil
		}
		if strings.Contains(payload, observationReviewerMarker) {
			return "observation-router-initial-reviewer-worker", "observations-initial", true, nil
		}
		if strings.Contains(payload, routerWorkerMarker) {
			return "router-reviewer", "router-strict", true, nil
		}
	case "strong-worker-model":
		if strings.Contains(payload, escalationWorkerMarker) {
			return "strong-escalation-worker", "escalation-strict", true, nil
		}
	}
	return "", "", false, fmt.Errorf("unexpected routing model/context %q", modelName)
}

func routingResponse(
	scenario string,
	step int,
	request map[string]any,
) (map[string]any, string, string, error) {
	if strings.HasPrefix(scenario, "observation-") {
		if strings.HasSuffix(scenario, "-planner") {
			return observationPlannerResponse(scenario, step, request)
		}
		return observationWorkerResponse(scenario, step, request)
	}
	if scenario == "base-escalation-planner" {
		if step != 1 {
			return nil, "", "", fmt.Errorf("base escalation Planner exceeded one call")
		}
		return toolCallMessage("base-finish", "finish", map[string]any{
			"outcome": "failed", "summary": "base policy intentionally rejected the work",
			"artifacts": map[string]any{},
			"error": map[string]any{
				"code": "base_policy_rejected", "message": "Use the pinned stronger policy", "retryable": false,
			},
		}), "tool_calls", "finish", nil
	}
	if strings.HasSuffix(scenario, "planner") {
		return routingPlannerResponse(scenario, step, request)
	}
	return routingWorkerResponse(scenario, step, request)
}

type observationFixture struct {
	BuilderPath    string
	ReviewerPath   string
	UnreadPath     string
	Before         string
	After          string
	BuilderReport  string
	ReviewerReport string
}

func fixtureForObservationScenario(scenario string) observationFixture {
	if strings.Contains(scenario, "-reuse-") {
		return observationFixture{
			BuilderPath: "pkg/fresh.py", ReviewerPath: "README-fresh.md",
			UnreadPath: "docs/fresh-unread.md", Before: `value = "fresh-before"`,
			After:          `value = "fresh-after"`,
			BuilderReport:  "# Fresh builder observation report",
			ReviewerReport: "# Fresh reviewer observation report",
		}
	}
	return observationFixture{
		BuilderPath: "src/app.py", ReviewerPath: "README.md",
		UnreadPath: "docs/unread.md", Before: `value = "old-value"`,
		After:          `value = "new-value"`,
		BuilderReport:  "# Builder observation report",
		ReviewerReport: "# Reviewer observation report",
	}
}

func observationPlannerResponse(
	scenario string,
	step int,
	request map[string]any,
) (map[string]any, string, string, error) {
	router := strings.Contains(scenario, "-router-")
	variant := "observations-initial"
	if strings.Contains(scenario, "-reuse-") {
		variant = "observations-reuse"
	} else if !router {
		variant = "observations-streamline"
	}
	workerArgs := func(workerName string, values map[string]any) map[string]any {
		result := make(map[string]any, len(values)+1)
		for key, value := range values {
			result[key] = value
		}
		if router && workerName != "" {
			result["worker_name"] = workerName
		}
		return result
	}
	call := func(id, name, workerName string, values map[string]any) (map[string]any, string, string, error) {
		return toolCallMessage(scenario+"-"+id, name, workerArgs(workerName, values)), "tool_calls", name, nil
	}
	fixture := fixtureForObservationScenario(scenario)

	// Both profiles first exercise the complete builder projection, including a
	// cursor crossing a real Control Plane -> Runtime ETag revalidation.
	switch step {
	case 1:
		return call("add-builder", "add_subtask", "", map[string]any{
			"objective":    observationBuilderMarker + " " + variant,
			"instructions": "Read two files, match and edit the implementation, then publish the report",
		})
	case 2:
		return call("execute-builder", "execute_current_subtask", "builder", map[string]any{"subtask_id": "0"})
	case 3:
		if err := requireObservationWorkerResult(request, fixture.BuilderReport, "builder", "report"); err != nil {
			return nil, "", "", fmt.Errorf("%s: %w", scenario, err)
		}
		return call("coverage-builder", "get_workspace_coverage", "builder", map[string]any{})
	case 4:
		if err := requireObservationCoverage(request, scenario+"-coverage-builder", 3, 2, 1, 1, 1); err != nil {
			return nil, "", "", err
		}
		return call("read-builder-1", "list_read_files", "builder", map[string]any{"cursor": "", "limit": 1})
	case 5:
		cursor, err := requireObservationPage(request, scenario+"-read-builder-1", []string{fixture.BuilderPath}, true)
		if err != nil {
			return nil, "", "", err
		}
		return call("read-builder-2", "list_read_files", "builder", map[string]any{"cursor": cursor, "limit": 1})
	case 6:
		if _, err := requireObservationPage(request, scenario+"-read-builder-2", []string{fixture.ReviewerPath}, false); err != nil {
			return nil, "", "", err
		}
		return call("unread-builder", "list_unread_files", "builder", map[string]any{"cursor": "", "limit": 100})
	case 7:
		if _, err := requireObservationPage(request, scenario+"-unread-builder", []string{fixture.UnreadPath}, false); err != nil {
			return nil, "", "", err
		}
		return call("usage-builder", "get_worker_tool_usage", "builder", map[string]any{})
	case 8:
		if _, err := requireObservationUsage(request, scenario+"-usage-builder", observationBuilderUsage()); err != nil {
			return nil, "", "", err
		}
		if !router {
			artifact, ok := exactArtifactValue(request, "builder", "report")
			if !ok {
				return nil, "", "", errors.New("observation Streamline lost exact builder report")
			}
			return call("finish", "finish", "", map[string]any{
				"outcome": "succeeded", "summary": "Worker observations inspected",
				"artifacts": map[string]any{"report": artifact},
			})
		}
		return call("add-reviewer", "add_subtask", "", map[string]any{
			"objective":    observationReviewerMarker + " " + variant,
			"instructions": "Read the review document and publish the reviewer report",
		})
	case 9:
		if !router {
			break
		}
		return call("execute-reviewer", "execute_current_subtask", "reviewer", map[string]any{"subtask_id": "1"})
	case 10:
		if !router {
			break
		}
		if err := requireObservationWorkerResult(request, fixture.ReviewerReport, "review", "report"); err != nil {
			return nil, "", "", fmt.Errorf("%s: %w", scenario, err)
		}
		return call("coverage-reviewer", "get_workspace_coverage", "reviewer", map[string]any{})
	case 11:
		if !router {
			break
		}
		if err := requireObservationCoverage(request, scenario+"-coverage-reviewer", 3, 1, 0, 0, 2); err != nil {
			return nil, "", "", err
		}
		return call("read-reviewer", "list_read_files", "reviewer", map[string]any{"cursor": "", "limit": 100})
	case 12:
		if !router {
			break
		}
		if _, err := requireObservationPage(request, scenario+"-read-reviewer", []string{fixture.ReviewerPath}, false); err != nil {
			return nil, "", "", err
		}
		return call("unread-reviewer", "list_unread_files", "reviewer", map[string]any{"cursor": "", "limit": 100})
	case 13:
		if !router {
			break
		}
		if _, err := requireObservationPage(
			request, scenario+"-unread-reviewer", []string{fixture.UnreadPath, fixture.BuilderPath}, false,
		); err != nil {
			return nil, "", "", err
		}
		return call("usage-reviewer", "get_worker_tool_usage", "reviewer", map[string]any{})
	case 14:
		if !router {
			break
		}
		if _, err := requireObservationUsage(request, scenario+"-usage-reviewer", observationReviewerUsage()); err != nil {
			return nil, "", "", err
		}
		builderArtifact, builderOK := exactArtifactValue(request, "builder", "report")
		reviewerArtifact, reviewerOK := exactArtifactValue(request, "review", "report")
		if !builderOK || !reviewerOK {
			return nil, "", "", errors.New("observation Router lost exact Worker reports")
		}
		return call("finish", "finish", "", map[string]any{
			"outcome": "succeeded", "summary": "Both Worker observations inspected",
			"artifacts": map[string]any{
				"builder_report": builderArtifact, "reviewer_report": reviewerArtifact,
			},
		})
	}
	return nil, "", "", fmt.Errorf("%s exceeded its bounded observation script at step %d", scenario, step)
}

func observationWorkerResponse(
	scenario string,
	step int,
	request map[string]any,
) (map[string]any, string, string, error) {
	fixture := fixtureForObservationScenario(scenario)
	reviewer := strings.Contains(scenario, "-reviewer-worker")
	if reviewer {
		switch step {
		case 1:
			return toolCallMessage(scenario+"-read", "read_file", map[string]any{
				"path": fixture.ReviewerPath, "start_line": 1, "max_lines": 20,
			}), "tool_calls", "read_file", nil
		case 2:
			if err := requireToolResponseContains(request, scenario+"-read", fixture.ReviewerPath); err != nil {
				return nil, "", "", err
			}
			return toolCallMessage(scenario+"-write", "write_text_artifact", map[string]any{
				"name": "report", "text": fixture.ReviewerReport,
				"media_type": "text/markdown", "expected_revision": nil,
			}), "tool_calls", "write_text_artifact", nil
		case 3:
			if _, ok := lastExactArtifact(request, "review", "report"); !ok {
				return nil, "", "", errors.New("observation reviewer did not observe exact report")
			}
			message, err := workerModelResultMessage(request, fixture.ReviewerReport)
			return message, "stop", "<final>", err
		}
		return nil, "", "", fmt.Errorf("%s exceeded its three-call Worker script", scenario)
	}

	switch step {
	case 1:
		return toolCallMessage(scenario+"-read-code", "read_file", map[string]any{
			"path": fixture.BuilderPath, "start_line": 1, "max_lines": 20,
		}), "tool_calls", "read_file", nil
	case 2:
		if err := requireToolResponseContains(request, scenario+"-read-code", fixture.Before); err != nil {
			return nil, "", "", err
		}
		return toolCallMessage(scenario+"-read-review", "read_file", map[string]any{
			"path": fixture.ReviewerPath, "start_line": 1, "max_lines": 20,
		}), "tool_calls", "read_file", nil
	case 3:
		if err := requireToolResponseContains(request, scenario+"-read-review", fixture.ReviewerPath); err != nil {
			return nil, "", "", err
		}
		root := strings.Split(fixture.BuilderPath, "/")[0]
		return toolCallMessage(scenario+"-grep", "grep", map[string]any{
			"pattern": fixture.Before, "path": root, "glob": "**/*.py",
			"regex": false, "case_sensitive": true, "cursor": "", "limit": 20,
		}), "tool_calls", "grep", nil
	case 4:
		if err := requireToolResponseContains(request, scenario+"-grep", fixture.BuilderPath); err != nil {
			return nil, "", "", err
		}
		return toolCallMessage(scenario+"-edit", "edit", map[string]any{
			"path": fixture.BuilderPath, "old": fixture.Before, "new": fixture.After,
			"replace_all": false,
		}), "tool_calls", "edit", nil
	case 5:
		if err := requireToolResponseContains(request, scenario+"-edit", `"changed":true`); err != nil {
			return nil, "", "", err
		}
		return toolCallMessage(scenario+"-write", "write_text_artifact", map[string]any{
			"name": "report", "text": fixture.BuilderReport,
			"media_type": "text/markdown", "expected_revision": nil,
		}), "tool_calls", "write_text_artifact", nil
	case 6:
		if _, ok := lastExactArtifact(request, "builder", "report"); !ok {
			return nil, "", "", errors.New("observation builder did not observe exact report")
		}
		message, err := workerModelResultMessage(request, fixture.BuilderReport)
		return message, "stop", "<final>", err
	default:
		return nil, "", "", fmt.Errorf("%s exceeded its six-call Worker script", scenario)
	}
}

func requireToolResponseContains(request map[string]any, callID, fragment string) error {
	response, ok := toolResponse(request, callID)
	if !ok {
		return fmt.Errorf("tool response %q is absent", callID)
	}
	if fragment == `"changed":true` {
		changed := false
		walkJSON(response, func(object map[string]any) {
			changed = changed || object["changed"] == true
		})
		if changed {
			return nil
		}
	} else if jsonStringContains(response, fragment) {
		return nil
	}
	return fmt.Errorf("tool response %q does not contain expected bounded evidence", callID)
}

func jsonStringContains(value any, fragment string) bool {
	switch typed := value.(type) {
	case string:
		return strings.Contains(typed, fragment)
	case map[string]any:
		for key, child := range typed {
			if strings.Contains(key, fragment) || jsonStringContains(child, fragment) {
				return true
			}
		}
	case []any:
		for _, child := range typed {
			if jsonStringContains(child, fragment) {
				return true
			}
		}
	}
	return false
}

func requireObservationWorkerResult(
	request map[string]any,
	result, namespace, name string,
) error {
	encoded, err := json.Marshal(request)
	if err != nil {
		return err
	}
	if !strings.Contains(string(encoded), result) {
		return errors.New("Planner did not receive the typed Worker result")
	}
	if _, ok := exactArtifactValue(request, namespace, name); !ok {
		return errors.New("Planner did not receive the Runtime-selected exact artifact")
	}
	return nil
}

func observationProjection(
	request map[string]any,
	callID, field string,
) (map[string]any, error) {
	response, ok := toolResponse(request, callID)
	if !ok {
		return nil, fmt.Errorf("State response %q is absent", callID)
	}
	var projection map[string]any
	walkJSON(response, func(object map[string]any) {
		if object["ok"] != true {
			return
		}
		if value, valid := object[field].(map[string]any); valid {
			projection = value
		}
	})
	if projection == nil {
		return nil, fmt.Errorf("State response %q has no successful %s projection", callID, field)
	}
	encoded, _ := json.Marshal(projection)
	for _, forbidden := range []string{
		"allocationId", "invocationId", "stateRevision", "runtimeUrl", "workspaceDigest",
	} {
		if strings.Contains(string(encoded), forbidden) {
			return nil, fmt.Errorf("State projection %q exposed %s", callID, forbidden)
		}
	}
	return projection, nil
}

func requireObservationCoverage(
	request map[string]any,
	callID string,
	scoped, read, matched, modified, unread int64,
) error {
	coverage, err := observationProjection(request, callID, "coverage")
	if err != nil {
		return err
	}
	want := map[string]int64{
		"scopedFiles": scoped, "readFiles": read, "matchedFiles": matched,
		"modifiedFiles": modified, "unreadFiles": unread,
	}
	for key, expected := range want {
		actual, ok := jsonInteger(coverage[key])
		if !ok || actual != expected {
			return fmt.Errorf("coverage %q %s = %v, want %d", callID, key, coverage[key], expected)
		}
	}
	if coverage["scopeComplete"] != true || coverage["detailComplete"] != true {
		return fmt.Errorf("coverage %q is unexpectedly incomplete: %+v", callID, coverage)
	}
	return nil
}

func requireObservationPage(
	request map[string]any,
	callID string,
	want []string,
	wantCursor bool,
) (string, error) {
	page, err := observationProjection(request, callID, "page")
	if err != nil {
		return "", err
	}
	rawFiles, ok := page["files"].([]any)
	if !ok || len(rawFiles) != len(want) {
		return "", fmt.Errorf("State page %q files = %v, want %v", callID, page["files"], want)
	}
	for index, expected := range want {
		if rawFiles[index] != expected {
			return "", fmt.Errorf("State page %q files = %v, want %v", callID, rawFiles, want)
		}
	}
	cursor, _ := page["nextCursor"].(string)
	if wantCursor != (cursor != "") || page["complete"] != true {
		return "", fmt.Errorf("State page %q cursor/complete = (%t, %v)", callID, cursor != "", page["complete"])
	}
	return cursor, nil
}

func observationBuilderUsage() workerObservationUsage {
	return workerObservationUsage{
		ModelCalls: 6, ToolCalls: 5, TotalTokens: 108,
		Tools: map[string]int64{"edit": 1, "grep": 1, "read_file": 2, "write_text_artifact": 1},
	}
}

func observationReviewerUsage() workerObservationUsage {
	return workerObservationUsage{
		ModelCalls: 3, ToolCalls: 2, TotalTokens: 54,
		Tools: map[string]int64{"read_file": 1, "write_text_artifact": 1},
	}
}

func requireObservationUsage(
	request map[string]any,
	callID string,
	want workerObservationUsage,
) (workerObservationUsage, error) {
	usage, err := observationProjection(request, callID, "usage")
	if err != nil {
		return workerObservationUsage{}, err
	}
	result := workerObservationUsage{Tools: map[string]int64{}}
	for field, target := range map[string]*int64{
		"modelCalls": &result.ModelCalls, "toolCalls": &result.ToolCalls,
		"totalTokens": &result.TotalTokens,
	} {
		value, ok := jsonInteger(usage[field])
		if !ok {
			return workerObservationUsage{}, fmt.Errorf("usage %q lacks integer %s", callID, field)
		}
		*target = value
	}
	rawTools, ok := usage["tools"].([]any)
	if !ok {
		return workerObservationUsage{}, fmt.Errorf("usage %q tools are invalid", callID)
	}
	for _, raw := range rawTools {
		item, ok := raw.(map[string]any)
		calls, callsOK := jsonInteger(item["calls"])
		name, nameOK := item["name"].(string)
		failures, failuresOK := jsonInteger(item["failures"])
		if !ok || !nameOK || !callsOK || !failuresOK || failures != 0 {
			return workerObservationUsage{}, fmt.Errorf("usage %q tool item is invalid: %+v", callID, raw)
		}
		result.Tools[name] = calls
	}
	if result.ModelCalls != want.ModelCalls || result.ToolCalls != want.ToolCalls ||
		result.TotalTokens != want.TotalTokens || !equalStringInt64Maps(result.Tools, want.Tools) {
		return workerObservationUsage{}, fmt.Errorf("usage %q = %+v, want %+v", callID, result, want)
	}
	return result, nil
}

func completedObservationUsage(
	scenario string,
	step int,
	request map[string]any,
) (workerObservationUsage, string, bool) {
	workerName := ""
	callID := ""
	want := workerObservationUsage{}
	switch {
	case strings.HasPrefix(scenario, "observation-streamline-") && step == 8:
		workerName, callID, want = "builder", scenario+"-usage-builder", observationBuilderUsage()
	case strings.Contains(scenario, "-router-") && step == 8:
		workerName, callID, want = "builder", scenario+"-usage-builder", observationBuilderUsage()
	case strings.Contains(scenario, "-router-") && step == 14:
		workerName, callID, want = "reviewer", scenario+"-usage-reviewer", observationReviewerUsage()
	default:
		return workerObservationUsage{}, "", false
	}
	usage, err := requireObservationUsage(request, callID, want)
	return usage, workerName, err == nil
}

func cloneStringInt64Map(source map[string]int64) map[string]int64 {
	result := make(map[string]int64, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func equalStringInt64Maps(left, right map[string]int64) bool {
	if len(left) != len(right) {
		return false
	}
	for key, value := range left {
		if right[key] != value {
			return false
		}
	}
	return true
}

func routingPlannerResponse(
	scenario string,
	step int,
	request map[string]any,
) (map[string]any, string, string, error) {
	workerMarker := streamlineWorkerMarker
	workerName := ""
	namespace := "builder"
	resultSlot := "copied"
	if scenario == "router-planner" {
		workerMarker = routerWorkerMarker
		workerName = "reviewer"
		namespace = "review"
		resultSlot = "reviewed"
	} else if scenario == "strong-escalation-planner" {
		workerMarker = escalationWorkerMarker
	}
	switch step {
	case 1:
		return toolCallMessage(scenario+"-add", "add_subtask", map[string]any{
			"objective":    workerMarker,
			"instructions": "Read the complete immutable context and write the exact copied artifact",
		}), "tool_calls", "add_subtask", nil
	case 2:
		arguments := map[string]any{"subtask_id": "0"}
		if workerName != "" {
			arguments["worker_name"] = workerName
		}
		return toolCallMessage(
			scenario+"-execute", "execute_current_subtask", arguments,
		), "tool_calls", "execute_current_subtask", nil
	case 3:
		artifact, ok := exactArtifactValue(request, namespace, "copied")
		if !ok {
			return nil, "", "", fmt.Errorf("%s did not observe the selected Worker artifact", scenario)
		}
		return toolCallMessage(scenario+"-finish", "finish", map[string]any{
			"outcome": "succeeded", "summary": scenario + " completed",
			"artifacts": map[string]any{resultSlot: artifact},
		}), "tool_calls", "finish", nil
	default:
		return nil, "", "", fmt.Errorf("%s exceeded its three-call script", scenario)
	}
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

func routingWorkerResponse(
	scenario string,
	step int,
	request map[string]any,
) (map[string]any, string, string, error) {
	namespace := "builder"
	if scenario == "router-reviewer" {
		namespace = "review"
	}
	switch step {
	case 1:
		return toolCallMessage(scenario+"-read", "read_artifact", map[string]any{
			"namespace": "inputs", "name": "source", "revision": nil,
		}), "tool_calls", "read_artifact", nil
	case 2:
		data, ok := lastStringValue(request, "dataBase64")
		if !ok || data == "" {
			return nil, "", "", fmt.Errorf("%s did not observe read_artifact bytes", scenario)
		}
		return toolCallMessage(scenario+"-write", "write_artifact", map[string]any{
			"namespace": namespace, "name": "copied", "media_type": "text/plain",
			"data_base64": data, "expected_revision": nil,
		}), "tool_calls", "write_artifact", nil
	case 3:
		_, ok := lastExactArtifact(request, namespace, "copied")
		if !ok {
			return nil, "", "", fmt.Errorf("%s did not observe write_artifact revision", scenario)
		}
		message, err := workerModelResultMessage(request, scenario+" copied the source")
		return message, "stop", "<final>", err
	default:
		return nil, "", "", fmt.Errorf("%s exceeded its three-call script", scenario)
	}
}

func (g *routingGateway) writeFailure(w http.ResponseWriter, status int, message string) {
	g.mu.Lock()
	g.failures = append(g.failures, message)
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": message, "type": "invalid_request_error"},
	})
}
