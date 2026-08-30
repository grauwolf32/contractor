//go:build e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	streamlineGlobalMarker = "STREAMLINE_E2E_GLOBAL"
	routerGlobalMarker     = "ROUTER_E2E_GLOBAL"
	escalationGlobalMarker = "ESCALATION_E2E_GLOBAL"
	streamlineWorkerMarker = "STREAMLINE_WORKER_TASK"
	routerWorkerMarker     = "ROUTER_REVIEWER_TASK"
	escalationWorkerMarker = "ESCALATED_WORKER_TASK"
)

type routingGateway struct {
	server *httptest.Server
	token  string

	mu           sync.Mutex
	steps        map[string]int
	modelCalls   map[string]int
	observations []routingGatewayObservation
	failures     []string
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
	wantTools := []string{"add_subtask", "execute_current_subtask", "finish", "list_subtasks"}
	if worker {
		wantTools = []string{"read_artifact", "write_artifact"}
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
		case strings.Contains(payload, streamlineWorkerMarker):
			return "streamline-worker", "streamline-strict", true, nil
		case strings.Contains(payload, routerWorkerMarker):
			return "router-builder-unexpected", "router-strict", true, fmt.Errorf("Router builder Worker was invoked")
		}
	case "reviewer-worker-model":
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
		artifact, ok := lastExactArtifact(request, namespace, "copied")
		if !ok {
			return nil, "", "", fmt.Errorf("%s did not observe write_artifact revision", scenario)
		}
		result, _ := json.Marshal(map[string]any{
			"apiVersion": "contractor/v1alpha1", "outcome": "succeeded",
			"summary":   scenario + " copied the source",
			"artifacts": map[string]any{"copied": artifact},
		})
		return map[string]any{"role": "assistant", "content": string(result)}, "stop", "<final>", nil
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
