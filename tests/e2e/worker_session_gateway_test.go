//go:build e2e

package e2e

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	sessionStreamlineIsolatedStage = "SESSION_MODE_STREAMLINE_ISOLATED_STAGE"
	sessionStreamlineAfterStage    = "SESSION_MODE_STREAMLINE_AFTER_STAGE"
	sessionRouterSharedStage       = "SESSION_MODE_ROUTER_SHARED_STAGE"
	sessionRouterAfterStage        = "SESSION_MODE_ROUTER_AFTER_STAGE"

	sessionIsolatedFirstTask  = "SESSION_ISOLATED_FIRST_TASK"
	sessionIsolatedSecondTask = "SESSION_ISOLATED_SECOND_TASK"
	sessionIsolatedAfterTask  = "SESSION_ISOLATED_AFTER_TASK"
	sessionSharedFirstTask    = "SESSION_SHARED_FIRST_TASK"
	sessionSharedSecondTask   = "SESSION_SHARED_SECOND_TASK"
	sessionSharedAfterTask    = "SESSION_SHARED_AFTER_TASK"

	sessionIsolatedFirstResult  = "SESSION_ISOLATED_FIRST_RESULT"
	sessionIsolatedSecondResult = "SESSION_ISOLATED_SECOND_RESULT"
	sessionIsolatedAfterResult  = "SESSION_ISOLATED_AFTER_RESULT"
	sessionSharedFirstResult    = "SESSION_SHARED_FIRST_RESULT"
	sessionSharedSecondResult   = "SESSION_SHARED_SECOND_RESULT"
	sessionSharedAfterResult    = "SESSION_SHARED_AFTER_RESULT"
)

type workerSessionGateway struct {
	server *httptest.Server
	token  string

	mu          sync.Mutex
	plannerStep map[string]int
	primarySeen map[string]int
	finalized   map[string]int
	observed    []string
	failures    []string
}

func newWorkerSessionGateway(token string) *workerSessionGateway {
	gateway := &workerSessionGateway{
		token: token, plannerStep: map[string]int{}, primarySeen: map[string]int{},
		finalized: map[string]int{},
	}
	gateway.server = httptest.NewServer(http.HandlerFunc(gateway.serveHTTP))
	return gateway
}

func (g *workerSessionGateway) close() { g.server.Close() }

func (g *workerSessionGateway) URL() string { return g.server.URL + "/v1" }

func (g *workerSessionGateway) Failures() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.failures...)
}

func (g *workerSessionGateway) Observations() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.observed...)
}

func (g *workerSessionGateway) assertComplete(t *testing.T) {
	t.Helper()
	g.mu.Lock()
	defer g.mu.Unlock()
	wantPlannerSteps := map[string]int{
		"streamline-isolated": 5,
		"streamline-after":    3,
		"router-shared":       5,
		"router-after":        3,
	}
	for scenario, want := range wantPlannerSteps {
		if got := g.plannerStep[scenario]; got != want {
			t.Fatalf("session gateway Planner steps for %s = %d, want %d; observed=%v",
				scenario, got, want, g.observed)
		}
	}
	for _, result := range []string{
		sessionIsolatedFirstResult, sessionIsolatedSecondResult, sessionIsolatedAfterResult,
		sessionSharedFirstResult, sessionSharedSecondResult, sessionSharedAfterResult,
	} {
		if g.primarySeen[result] != 1 || g.finalized[result] != 1 {
			t.Fatalf("session gateway lifecycle for %s = primary:%d finalizer:%d, want 1/1; observed=%v",
				result, g.primarySeen[result], g.finalized[result], g.observed)
		}
	}
}

func (g *workerSessionGateway) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		g.writeFailure(w, http.StatusNotFound, "unsupported session Gateway endpoint")
		return
	}
	if r.Header.Get("Authorization") != "Bearer "+g.token {
		g.writeFailure(w, http.StatusUnauthorized, "invalid session Gateway token")
		return
	}
	defer r.Body.Close()
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20))
	decoder.UseNumber()
	var request map[string]any
	if err := decoder.Decode(&request); err != nil {
		g.writeFailure(w, http.StatusBadRequest, "invalid session Gateway request")
		return
	}
	message, finishReason, model, err := g.next(request)
	if err != nil {
		g.writeFailure(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":      fmt.Sprintf("chatcmpl-session-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []any{map[string]any{
			"index": 0, "message": message, "finish_reason": finishReason,
		}},
		"usage": map[string]any{
			"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13,
		},
	})
}

func (g *workerSessionGateway) next(
	request map[string]any,
) (map[string]any, string, string, error) {
	model, ok := request["model"].(string)
	if !ok || model == "" {
		return nil, "", "", errors.New("session Gateway request has no model")
	}
	encoded, err := json.Marshal(request)
	if err != nil {
		return nil, "", "", err
	}
	payload := string(encoded)

	if finalizerInput, finalizer, decodeErr := decodeWorkerResultFinalizerRequest(request); decodeErr != nil {
		return nil, "", "", decodeErr
	} else if finalizer {
		if model != "worker-model" || !requestHasNoModelTools(request) {
			return nil, "", "", errors.New("session Worker result finalizer has an invalid model surface")
		}
		var candidate struct {
			SubtaskID string `json:"subtaskId"`
			Result    string `json:"result"`
		}
		if err := json.Unmarshal([]byte(finalizerInput.ResultText), &candidate); err != nil ||
			candidate.SubtaskID != finalizerInput.SubtaskID || !knownSessionResult(candidate.Result) {
			return nil, "", "", fmt.Errorf("unexpected session finalizer result %q", finalizerInput.ResultText)
		}
		message, encodeErr := workerResultFinalizerMessage(finalizerInput)
		if encodeErr != nil {
			return nil, "", "", encodeErr
		}
		g.mu.Lock()
		g.finalized[candidate.Result]++
		g.observed = append(g.observed, "finalizer:"+candidate.Result)
		g.mu.Unlock()
		return message, "stop", model, nil
	}

	switch model {
	case "planner-model":
		return g.nextPlanner(request, payload, model)
	case "worker-model":
		return g.nextWorker(request, payload, model)
	default:
		return nil, "", "", fmt.Errorf("unexpected session Gateway model %q", model)
	}
}

func (g *workerSessionGateway) nextPlanner(
	request map[string]any, payload, model string,
) (map[string]any, string, string, error) {
	scenario := ""
	router := false
	switch {
	case strings.Contains(payload, sessionStreamlineIsolatedStage):
		scenario = "streamline-isolated"
	case strings.Contains(payload, sessionStreamlineAfterStage):
		scenario = "streamline-after"
	case strings.Contains(payload, sessionRouterSharedStage):
		scenario, router = "router-shared", true
	case strings.Contains(payload, sessionRouterAfterStage):
		scenario = "router-after"
	default:
		return nil, "", "", errors.New("cannot identify session Planner Stage")
	}

	g.mu.Lock()
	g.plannerStep[scenario]++
	step := g.plannerStep[scenario]
	g.observed = append(g.observed, fmt.Sprintf("planner:%s:%d", scenario, step))
	g.mu.Unlock()

	firstTask, firstResult := sessionIsolatedFirstTask, sessionIsolatedFirstResult
	secondTask, secondResult := sessionIsolatedSecondTask, sessionIsolatedSecondResult
	afterTask, afterResult := sessionIsolatedAfterTask, sessionIsolatedAfterResult
	if strings.HasPrefix(scenario, "router-") {
		firstTask, firstResult = sessionSharedFirstTask, sessionSharedFirstResult
		secondTask, secondResult = sessionSharedSecondTask, sessionSharedSecondResult
		afterTask, afterResult = sessionSharedAfterTask, sessionSharedAfterResult
	}
	if strings.HasSuffix(scenario, "-after") {
		switch step {
		case 1:
			return plannerSessionToolCall(scenario+"-add", "add_subtask", router, map[string]any{
				"objective": afterTask, "instructions": "Return the allocation-boundary probe",
			}), "tool_calls", model, nil
		case 2:
			return plannerSessionToolCall(scenario+"-execute", "execute_current_subtask", router,
				map[string]any{"subtask_id": "0"}), "tool_calls", model, nil
		case 3:
			if !strings.Contains(payload, afterResult) {
				return nil, "", "", fmt.Errorf("%s Planner did not observe the Worker result", scenario)
			}
			return toolCallMessage(scenario+"-finish", "finish", map[string]any{
				"outcome": "succeeded", "summary": scenario + " completed", "artifacts": map[string]any{},
			}), "tool_calls", model, nil
		default:
			return nil, "", "", fmt.Errorf("%s Planner exceeded three calls", scenario)
		}
	}

	switch step {
	case 1:
		return plannerSessionToolCall(scenario+"-add-first", "add_subtask", router, map[string]any{
			"objective": firstTask, "instructions": "Return the first session probe",
		}), "tool_calls", model, nil
	case 2:
		return plannerSessionToolCall(scenario+"-execute-first", "execute_current_subtask", router,
			map[string]any{"subtask_id": "0"}), "tool_calls", model, nil
	case 3:
		if !strings.Contains(payload, firstResult) {
			return nil, "", "", fmt.Errorf("%s Planner did not observe its first Worker result", scenario)
		}
		return plannerSessionToolCall(scenario+"-add-second", "add_subtask", router, map[string]any{
			"objective": secondTask, "instructions": "Return the second session probe",
		}), "tool_calls", model, nil
	case 4:
		return plannerSessionToolCall(scenario+"-execute-second", "execute_current_subtask", router,
			map[string]any{"subtask_id": "1"}), "tool_calls", model, nil
	case 5:
		if !strings.Contains(payload, secondResult) {
			return nil, "", "", fmt.Errorf("%s Planner did not observe its second Worker result", scenario)
		}
		return toolCallMessage(scenario+"-finish", "finish", map[string]any{
			"outcome": "succeeded", "summary": scenario + " completed", "artifacts": map[string]any{},
		}), "tool_calls", model, nil
	default:
		return nil, "", "", fmt.Errorf("%s Planner exceeded five calls", scenario)
	}
}

func plannerSessionToolCall(
	id, name string, router bool, arguments map[string]any,
) map[string]any {
	if router && name == "execute_current_subtask" {
		arguments["worker_name"] = "builder"
	}
	return toolCallMessage(id, name, arguments)
}

func (g *workerSessionGateway) nextWorker(
	request map[string]any, payload, model string,
) (map[string]any, string, string, error) {
	result := ""
	var required, forbidden []string
	switch {
	case strings.Contains(payload, sessionIsolatedAfterTask):
		result = sessionIsolatedAfterResult
		forbidden = []string{
			sessionIsolatedFirstTask, sessionIsolatedFirstResult,
			sessionIsolatedSecondTask, sessionIsolatedSecondResult,
		}
	case strings.Contains(payload, sessionIsolatedSecondTask):
		result = sessionIsolatedSecondResult
		forbidden = []string{sessionIsolatedFirstTask, sessionIsolatedFirstResult}
	case strings.Contains(payload, sessionIsolatedFirstTask):
		result = sessionIsolatedFirstResult
		forbidden = []string{sessionIsolatedSecondTask, sessionIsolatedAfterTask}
	case strings.Contains(payload, sessionSharedAfterTask):
		result = sessionSharedAfterResult
		forbidden = []string{
			sessionSharedFirstTask, sessionSharedFirstResult,
			sessionSharedSecondTask, sessionSharedSecondResult,
		}
	case strings.Contains(payload, sessionSharedSecondTask):
		result = sessionSharedSecondResult
		required = []string{sessionSharedFirstTask, sessionSharedFirstResult}
	case strings.Contains(payload, sessionSharedFirstTask):
		result = sessionSharedFirstResult
		forbidden = []string{sessionSharedSecondTask, sessionSharedAfterTask}
	default:
		return nil, "", "", errors.New("cannot identify current session Worker subtask")
	}
	if !requestHasNoModelTools(request) {
		return nil, "", "", fmt.Errorf("session probe Worker %s unexpectedly received tools", result)
	}
	for _, value := range required {
		if !strings.Contains(payload, value) {
			return nil, "", "", fmt.Errorf("session probe %s did not retain %s", result, value)
		}
	}
	for _, value := range forbidden {
		if strings.Contains(payload, value) {
			return nil, "", "", fmt.Errorf("session probe %s leaked %s", result, value)
		}
	}
	message, err := sessionWorkerModelResultMessage(result)
	if err != nil {
		return nil, "", "", err
	}
	g.mu.Lock()
	g.primarySeen[result]++
	g.observed = append(g.observed, "worker:"+result)
	g.mu.Unlock()
	return message, "stop", model, nil
}

func sessionWorkerModelResultMessage(result string) (map[string]any, error) {
	subtaskID := "0"
	if result == sessionIsolatedSecondResult || result == sessionSharedSecondResult {
		subtaskID = "1"
	}
	encoded, err := json.Marshal(map[string]any{
		"subtaskId": subtaskID,
		"result":    result,
	})
	if err != nil {
		return nil, err
	}
	return map[string]any{"role": "assistant", "content": string(encoded)}, nil
}

func knownSessionResult(value string) bool {
	switch value {
	case sessionIsolatedFirstResult, sessionIsolatedSecondResult, sessionIsolatedAfterResult,
		sessionSharedFirstResult, sessionSharedSecondResult, sessionSharedAfterResult:
		return true
	default:
		return false
	}
}

func (g *workerSessionGateway) writeFailure(w http.ResponseWriter, status int, message string) {
	g.mu.Lock()
	g.failures = append(g.failures, message)
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": message, "type": "invalid_request_error"},
	})
}
