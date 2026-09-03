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
	memoryCoordinateObjective = "Coordinate a shared note through one retry and a later Stage"
	memoryConfirmObjective    = "Confirm the same Run shared note in a later Stage"
	memoryRouterObjective     = "Route isolated shared notes through two logical Workers"

	streamlineNoteName        = "shared_note"
	streamlineSeed            = `streamline seed "canary/v9" Ω%`
	streamlineFirstAppend     = `streamline first append "v9/a" Ω%`
	streamlineRetryAppend     = `streamline retry append "v9/b" Ω%`
	streamlineConfirmAppend   = `streamline confirm append "v9/c" Ω%`
	streamlineDescription     = `streamline description "canary/v9" Ω%`
	streamlineTag             = "streamline_tag_v9"
	routerNoteName            = "route_note"
	routerBuilderSeed         = `router builder seed "canary/v9" Ж%`
	routerBuilderPlanner      = `router builder planner append "v9/a" Ж%`
	routerBuilderWorker       = `router builder worker append "v9/b" Ж%`
	routerBuilderDescription  = `router builder description "canary/v9" Ж%`
	routerBuilderTag          = "router_builder_tag_v9"
	routerReviewerSeed        = `router reviewer seed "canary/v9" Д%`
	routerReviewerDescription = `router reviewer description "canary/v9" Д%`
	routerReviewerTag         = "router_reviewer_tag_v9"
)

var sharedMemoryToolNames = map[string]bool{
	"list_memories": true, "read_memory": true, "write_memory": true,
	"append_memory": true, "search_memory": true, "list_memory_tags": true,
}

type sharedMemoryGateway struct {
	server *httptest.Server
	token  string

	mu                       sync.Mutex
	calls                    int
	emptyCoordinateStarts    int
	existingCoordinateStarts int
	observations             []sharedMemoryGatewayObservation
	transcripts              []string
	transcriptBytes          int
	failures                 []string
}

type sharedMemoryGatewayObservation struct {
	Scenario string
	Step     int
	Tool     string
}

func newSharedMemoryGateway(token string) *sharedMemoryGateway {
	result := &sharedMemoryGateway{token: token}
	result.server = httptest.NewServer(http.HandlerFunc(result.serveHTTP))
	return result
}

func (g *sharedMemoryGateway) close()      { g.server.Close() }
func (g *sharedMemoryGateway) URL() string { return g.server.URL + "/v1" }

func (g *sharedMemoryGateway) Failures() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.failures...)
}

func (g *sharedMemoryGateway) Observations() []sharedMemoryGatewayObservation {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]sharedMemoryGatewayObservation(nil), g.observations...)
}

func (g *sharedMemoryGateway) StartCounts() (empty, existing int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.emptyCoordinateStarts, g.existingCoordinateStarts
}

func (g *sharedMemoryGateway) Transcripts() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.transcripts...)
}

func (g *sharedMemoryGateway) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		g.writeFailure(w, http.StatusNotFound, "unsupported shared-memory Gateway endpoint")
		return
	}
	if r.Header.Get("Authorization") != "Bearer "+g.token {
		g.writeFailure(w, http.StatusUnauthorized, "invalid shared-memory Gateway token")
		return
	}
	defer r.Body.Close()
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20))
	decoder.UseNumber()
	var request map[string]any
	if err := decoder.Decode(&request); err != nil {
		g.writeFailure(w, http.StatusBadRequest, "invalid shared-memory Gateway request")
		return
	}
	message, finishReason, modelName, scenario, step, toolName, err := g.next(request)
	if err != nil {
		g.writeFailure(w, http.StatusBadRequest, err.Error())
		return
	}
	g.mu.Lock()
	g.calls++
	call := g.calls
	g.observations = append(g.observations, sharedMemoryGatewayObservation{
		Scenario: scenario, Step: step, Tool: toolName,
	})
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":      fmt.Sprintf("chatcmpl-memory-%d", call),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   modelName,
		"choices": []any{map[string]any{
			"index": 0, "message": message, "finish_reason": finishReason,
		}},
		"usage": map[string]any{
			"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16,
		},
	})
}

func (g *sharedMemoryGateway) next(
	request map[string]any,
) (
	message map[string]any,
	finishReason string,
	modelName string,
	scenario string,
	step int,
	toolName string,
	err error,
) {
	modelName, ok := request["model"].(string)
	if !ok || strings.TrimSpace(modelName) == "" {
		return nil, "", "", "", 0, "", errors.New("shared-memory request has no model")
	}
	encoded, marshalErr := json.Marshal(request)
	if marshalErr != nil {
		return nil, "", "", "", 0, "", marshalErr
	}
	scenario, worker, detectErr := sharedMemoryScenario(modelName, string(encoded))
	if detectErr != nil {
		return nil, "", "", "", 0, "", detectErr
	}
	g.mu.Lock()
	if g.transcriptBytes+len(encoded) > 16<<20 {
		g.mu.Unlock()
		return nil, "", "", "", 0, "", errors.New("shared-memory Gateway transcript exceeded 16 MiB")
	}
	g.transcriptBytes += len(encoded)
	g.transcripts = append(g.transcripts, string(encoded))
	g.mu.Unlock()
	if schemaErr := validateSharedMemoryTools(request, scenario, worker); schemaErr != nil {
		return nil, "", "", "", 0, "", fmt.Errorf("%s: %w", scenario, schemaErr)
	}
	if transcriptErr := validateMemoryTranscript(request); transcriptErr != nil {
		return nil, "", "", "", 0, "", fmt.Errorf("%s: %w", scenario, transcriptErr)
	}
	priorCalls, callsErr := messageFunctionCalls(request)
	if callsErr != nil {
		return nil, "", "", "", 0, "", callsErr
	}
	step = len(priorCalls) + 1
	if worker {
		message, finishReason, toolName, err = sharedMemoryWorkerResponse(scenario, step, request)
	} else {
		message, finishReason, toolName, err = g.sharedMemoryPlannerResponse(scenario, step, request, priorCalls)
	}
	return message, finishReason, modelName, scenario, step, toolName, err
}

func sharedMemoryScenario(modelName, payload string) (string, bool, error) {
	if modelName == "planner-model" {
		switch {
		case strings.Contains(payload, memoryCoordinateObjective):
			return "coordinate-planner", false, nil
		case strings.Contains(payload, memoryConfirmObjective):
			return "confirm-planner", false, nil
		case strings.Contains(payload, memoryRouterObjective):
			return "router-planner", false, nil
		}
	}
	if modelName == "worker-model" {
		for marker, scenario := range map[string]string{
			"MEMORY_STREAMLINE_FIRST_WORKER":    "streamline-first-worker",
			"MEMORY_STREAMLINE_RETRY_WORKER":    "streamline-retry-worker",
			"MEMORY_STREAMLINE_CONTINUE_WORKER": "streamline-confirm-worker",
			"MEMORY_ROUTER_BUILDER_WORKER":      "router-builder-worker",
			"MEMORY_ROUTER_REVIEWER_WORKER":     "router-reviewer-worker",
		} {
			if strings.Contains(payload, marker) {
				return scenario, true, nil
			}
		}
	}
	return "", false, fmt.Errorf("unexpected shared-memory model/context %q", modelName)
}

func (g *sharedMemoryGateway) sharedMemoryPlannerResponse(
	scenario string,
	step int,
	request map[string]any,
	priorCalls []gatewayFunctionCall,
) (map[string]any, string, string, error) {
	switch scenario {
	case "coordinate-planner":
		return g.coordinatePlannerResponse(step, request, priorCalls)
	case "confirm-planner":
		return confirmPlannerResponse(step, request)
	case "router-planner":
		return routerMemoryPlannerResponse(step, request)
	default:
		return nil, "", "", fmt.Errorf("unknown Planner scenario %q", scenario)
	}
}

func (g *sharedMemoryGateway) coordinatePlannerResponse(
	step int,
	request map[string]any,
	priorCalls []gatewayFunctionCall,
) (map[string]any, string, string, error) {
	firstAttempt := hasFunctionCall(priorCalls, "write_memory")
	switch step {
	case 1:
		return gatewayToolCall("coordinate-list", "list_memories", map[string]any{})
	case 2:
		items, ok := toolResponseList(request, "coordinate-list")
		if !ok {
			return nil, "", "", errors.New("coordinate Planner did not receive list_memories result")
		}
		if len(items) == 0 {
			g.mu.Lock()
			g.emptyCoordinateStarts++
			g.mu.Unlock()
			return gatewayToolCall("coordinate-seed", "write_memory", map[string]any{
				"name": streamlineNoteName, "content": streamlineSeed,
				"description": streamlineDescription, "tags": []string{streamlineTag},
			})
		}
		if len(items) != 1 {
			return nil, "", "", fmt.Errorf("coordinate retry listed %d notes, want one", len(items))
		}
		g.mu.Lock()
		g.existingCoordinateStarts++
		g.mu.Unlock()
		return gatewayToolCall("coordinate-existing", "read_memory", map[string]any{"name": streamlineNoteName})
	case 3:
		marker := "MEMORY_STREAMLINE_RETRY_WORKER"
		expected := streamlineSeed + "\n" + streamlineFirstAppend
		responseID := "coordinate-existing"
		if firstAttempt {
			marker = "MEMORY_STREAMLINE_FIRST_WORKER"
			expected = streamlineSeed
			responseID = "coordinate-seed"
		}
		if err := requireMemoryNote(request, responseID, streamlineNoteName, expected, streamlineDescription, []string{streamlineTag}, 0); err != nil {
			return nil, "", "", err
		}
		return gatewayToolCall("coordinate-add", "add_subtask", map[string]any{
			"objective":    marker,
			"instructions": "Read and append the exact shared note, then return success without artifacts",
		})
	case 4:
		return gatewayToolCall("coordinate-execute", "execute_current_subtask", map[string]any{"subtask_id": "0"})
	case 5:
		return gatewayToolCall("coordinate-final-read", "read_memory", map[string]any{"name": streamlineNoteName})
	case 6:
		expected := streamlineSeed + "\n" + streamlineFirstAppend + "\n" + streamlineRetryAppend
		outcome := "succeeded"
		arguments := map[string]any{
			"outcome": outcome, "summary": "retry observed the shared note", "artifacts": map[string]any{},
		}
		if firstAttempt {
			expected = streamlineSeed + "\n" + streamlineFirstAppend
			arguments["outcome"] = "failed"
			arguments["summary"] = "intentional retry after shared note update"
			arguments["error"] = map[string]any{
				"code": "memory_retry_probe", "message": "Retry to prove namespace continuity", "retryable": true,
			}
		}
		if err := requireMemoryNote(request, "coordinate-final-read", streamlineNoteName, expected, streamlineDescription, []string{streamlineTag}, 0); err != nil {
			return nil, "", "", err
		}
		firstResponse := "coordinate-existing"
		if firstAttempt {
			firstResponse = "coordinate-seed"
		}
		if err := requireMemoryTimestampTransition(request, firstResponse, "coordinate-final-read", true); err != nil {
			return nil, "", "", err
		}
		return gatewayToolCall("coordinate-finish", "finish", arguments)
	default:
		return nil, "", "", fmt.Errorf("coordinate Planner exceeded six calls (step %d)", step)
	}
}

func confirmPlannerResponse(step int, request map[string]any) (map[string]any, string, string, error) {
	before := streamlineSeed + "\n" + streamlineFirstAppend + "\n" + streamlineRetryAppend
	switch step {
	case 1:
		return gatewayToolCall("confirm-list", "list_memories", map[string]any{})
	case 2:
		items, ok := toolResponseList(request, "confirm-list")
		if !ok || len(items) != 1 {
			return nil, "", "", fmt.Errorf("later Stage list = (%d, %t), want one note", len(items), ok)
		}
		return gatewayToolCall("confirm-read", "read_memory", map[string]any{"name": streamlineNoteName})
	case 3:
		if err := requireMemoryNote(request, "confirm-read", streamlineNoteName, before, streamlineDescription, []string{streamlineTag}, 0); err != nil {
			return nil, "", "", err
		}
		return gatewayToolCall("confirm-add", "add_subtask", map[string]any{
			"objective":    "MEMORY_STREAMLINE_CONTINUE_WORKER",
			"instructions": "Read and append the exact later-Stage shared note",
		})
	case 4:
		return gatewayToolCall("confirm-execute", "execute_current_subtask", map[string]any{"subtask_id": "0"})
	case 5:
		return gatewayToolCall("confirm-final-read", "read_memory", map[string]any{"name": streamlineNoteName})
	case 6:
		if err := requireMemoryNote(
			request, "confirm-final-read", streamlineNoteName,
			before+"\n"+streamlineConfirmAppend, streamlineDescription, []string{streamlineTag}, 0,
		); err != nil {
			return nil, "", "", err
		}
		if err := requireMemoryTimestampTransition(request, "confirm-read", "confirm-final-read", true); err != nil {
			return nil, "", "", err
		}
		return gatewayToolCall("confirm-finish", "finish", map[string]any{
			"outcome": "succeeded", "summary": "later Stage observed shared note", "artifacts": map[string]any{},
		})
	default:
		return nil, "", "", fmt.Errorf("confirm Planner exceeded six calls (step %d)", step)
	}
}

func routerMemoryPlannerResponse(step int, request map[string]any) (map[string]any, string, string, error) {
	builderPlannerContent := routerBuilderSeed + "\n" + routerBuilderPlanner
	builderWorkerContent := builderPlannerContent + "\n" + routerBuilderWorker
	switch step {
	case 1:
		return gatewayToolCall("router-write-builder", "write_memory", map[string]any{
			"worker_name": "builder", "name": routerNoteName, "content": routerBuilderSeed,
			"description": routerBuilderDescription, "tags": []string{routerBuilderTag},
		})
	case 2:
		if err := requireMemoryNote(request, "router-write-builder", routerNoteName, routerBuilderSeed, routerBuilderDescription, []string{routerBuilderTag}, 0); err != nil {
			return nil, "", "", err
		}
		return gatewayToolCall("router-write-reviewer", "write_memory", map[string]any{
			"worker_name": "reviewer", "name": routerNoteName, "content": routerReviewerSeed,
			"description": routerReviewerDescription, "tags": []string{routerReviewerTag},
		})
	case 3:
		if err := requireMemoryNote(request, "router-write-reviewer", routerNoteName, routerReviewerSeed, routerReviewerDescription, []string{routerReviewerTag}, 0); err != nil {
			return nil, "", "", err
		}
		return gatewayToolCall("router-append-builder", "append_memory", map[string]any{
			"worker_name": "builder", "name": routerNoteName, "content": routerBuilderPlanner,
		})
	case 4:
		if err := requireMemoryNote(request, "router-append-builder", routerNoteName, builderPlannerContent, routerBuilderDescription, []string{routerBuilderTag}, 0); err != nil {
			return nil, "", "", err
		}
		if err := requireMemoryTimestampTransition(request, "router-write-builder", "router-append-builder", true); err != nil {
			return nil, "", "", err
		}
		return gatewayToolCall("router-read-builder", "read_memory", map[string]any{
			"worker_name": "builder", "name": routerNoteName,
		})
	case 5:
		if err := requireMemoryNote(request, "router-read-builder", routerNoteName, builderPlannerContent, routerBuilderDescription, []string{routerBuilderTag}, 0); err != nil {
			return nil, "", "", err
		}
		return gatewayToolCall("router-read-reviewer", "read_memory", map[string]any{
			"worker_name": "reviewer", "name": routerNoteName,
		})
	case 6:
		if err := requireMemoryNote(request, "router-read-reviewer", routerNoteName, routerReviewerSeed, routerReviewerDescription, []string{routerReviewerTag}, 0); err != nil {
			return nil, "", "", err
		}
		return gatewayToolCall("router-add-builder", "add_subtask", map[string]any{
			"objective": "MEMORY_ROUTER_BUILDER_WORKER", "instructions": "Read and append only the builder note",
		})
	case 7:
		return gatewayToolCall("router-execute-builder", "execute_current_subtask", map[string]any{
			"subtask_id": "0", "worker_name": "builder",
		})
	case 8:
		return gatewayToolCall("router-read-builder-final", "read_memory", map[string]any{
			"worker_name": "builder", "name": routerNoteName,
		})
	case 9:
		if err := requireMemoryNote(request, "router-read-builder-final", routerNoteName, builderWorkerContent, routerBuilderDescription, []string{routerBuilderTag}, 0); err != nil {
			return nil, "", "", err
		}
		if err := requireMemoryTimestampTransition(request, "router-read-builder", "router-read-builder-final", true); err != nil {
			return nil, "", "", err
		}
		return gatewayToolCall("router-add-reviewer", "add_subtask", map[string]any{
			"objective": "MEMORY_ROUTER_REVIEWER_WORKER", "instructions": "Read only the reviewer note",
		})
	case 10:
		return gatewayToolCall("router-execute-reviewer", "execute_current_subtask", map[string]any{
			"subtask_id": "1", "worker_name": "reviewer",
		})
	case 11:
		return gatewayToolCall("router-read-reviewer-final", "read_memory", map[string]any{
			"worker_name": "reviewer", "name": routerNoteName,
		})
	case 12:
		if err := requireMemoryNote(request, "router-read-reviewer-final", routerNoteName, routerReviewerSeed, routerReviewerDescription, []string{routerReviewerTag}, 0); err != nil {
			return nil, "", "", err
		}
		if err := requireMemoryTimestampTransition(request, "router-write-reviewer", "router-read-reviewer-final", false); err != nil {
			return nil, "", "", err
		}
		return gatewayToolCall("router-finish", "finish", map[string]any{
			"outcome": "succeeded", "summary": "both isolated logical Workers completed", "artifacts": map[string]any{},
		})
	default:
		return nil, "", "", fmt.Errorf("Router Planner exceeded twelve calls (step %d)", step)
	}
}

func sharedMemoryWorkerResponse(
	scenario string, step int, request map[string]any,
) (map[string]any, string, string, error) {
	name := streamlineNoteName
	before, after, description, tag, appendValue := "", "", streamlineDescription, streamlineTag, ""
	switch scenario {
	case "streamline-first-worker":
		before, appendValue = streamlineSeed, streamlineFirstAppend
	case "streamline-retry-worker":
		before, appendValue = streamlineSeed+"\n"+streamlineFirstAppend, streamlineRetryAppend
	case "streamline-confirm-worker":
		before = streamlineSeed + "\n" + streamlineFirstAppend + "\n" + streamlineRetryAppend
		appendValue = streamlineConfirmAppend
	case "router-builder-worker":
		name = routerNoteName
		before = routerBuilderSeed + "\n" + routerBuilderPlanner
		appendValue, description, tag = routerBuilderWorker, routerBuilderDescription, routerBuilderTag
	case "router-reviewer-worker":
		name, before = routerNoteName, routerReviewerSeed
		description, tag = routerReviewerDescription, routerReviewerTag
	default:
		return nil, "", "", fmt.Errorf("unknown Worker scenario %q", scenario)
	}
	after = before
	if appendValue != "" {
		after += "\n" + appendValue
	}
	switch step {
	case 1:
		return gatewayToolCall(scenario+"-read", "read_memory", map[string]any{"name": name})
	case 2:
		if err := requireMemoryNote(request, scenario+"-read", name, before, description, []string{tag}, 0); err != nil {
			return nil, "", "", err
		}
		if appendValue == "" {
			message, err := workerModelResultMessage(request, scenario+" read isolated note")
			return message, "stop", "<final>", err
		}
		return gatewayToolCall(scenario+"-append", "append_memory", map[string]any{
			"name": name, "content": appendValue,
		})
	case 3:
		if appendValue == "" {
			return nil, "", "", fmt.Errorf("%s called after its final response", scenario)
		}
		if err := requireMemoryNote(request, scenario+"-append", name, after, description, []string{tag}, 0); err != nil {
			return nil, "", "", err
		}
		if err := requireMemoryTimestampTransition(request, scenario+"-read", scenario+"-append", true); err != nil {
			return nil, "", "", err
		}
		message, err := workerModelResultMessage(request, scenario+" updated isolated note")
		return message, "stop", "<final>", err
	default:
		return nil, "", "", fmt.Errorf("%s exceeded its bounded script (step %d)", scenario, step)
	}
}

func gatewayToolCall(id, name string, arguments map[string]any) (map[string]any, string, string, error) {
	return toolCallMessage(id, name, arguments), "tool_calls", name, nil
}

type gatewayFunctionCall struct {
	ID        string
	Name      string
	Arguments map[string]any
}

func messageFunctionCalls(request map[string]any) ([]gatewayFunctionCall, error) {
	messages, ok := request["messages"].([]any)
	if !ok {
		return nil, errors.New("shared-memory request has no messages")
	}
	result := make([]gatewayFunctionCall, 0)
	for _, rawMessage := range messages {
		message, ok := rawMessage.(map[string]any)
		if !ok {
			return nil, errors.New("shared-memory request contains an invalid message")
		}
		calls, present := message["tool_calls"].([]any)
		if !present {
			continue
		}
		for _, rawCall := range calls {
			call, ok := rawCall.(map[string]any)
			if !ok {
				return nil, errors.New("shared-memory request contains an invalid tool call")
			}
			function, ok := call["function"].(map[string]any)
			if !ok {
				return nil, errors.New("shared-memory request contains an invalid function call")
			}
			id, _ := call["id"].(string)
			name, _ := function["name"].(string)
			arguments, err := decodeJSONObject(function["arguments"])
			if id == "" || name == "" || err != nil {
				return nil, errors.New("shared-memory request contains an incomplete function call")
			}
			result = append(result, gatewayFunctionCall{ID: id, Name: name, Arguments: arguments})
		}
	}
	return result, nil
}

func hasFunctionCall(calls []gatewayFunctionCall, name string) bool {
	for _, call := range calls {
		if call.Name == name {
			return true
		}
	}
	return false
}

func decodeJSONObject(value any) (map[string]any, error) {
	if object, ok := value.(map[string]any); ok {
		return object, nil
	}
	text, ok := value.(string)
	if !ok {
		return nil, errors.New("function arguments are not JSON")
	}
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	var result map[string]any
	if err := decoder.Decode(&result); err != nil {
		return nil, err
	}
	return result, nil
}

func toolResponse(request map[string]any, callID string) (any, bool) {
	messages, _ := request["messages"].([]any)
	for _, rawMessage := range messages {
		message, ok := rawMessage.(map[string]any)
		if !ok || message["tool_call_id"] != callID {
			continue
		}
		content := message["content"]
		if text, ok := content.(string); ok {
			decoder := json.NewDecoder(strings.NewReader(text))
			decoder.UseNumber()
			var decoded any
			if decoder.Decode(&decoded) == nil {
				return decoded, true
			}
		}
		return content, true
	}
	return nil, false
}

func toolResponseList(request map[string]any, callID string) ([]any, bool) {
	response, ok := toolResponse(request, callID)
	if !ok {
		return nil, false
	}
	if direct, ok := response.([]any); ok {
		return direct, true
	}
	var result []any
	found := false
	walkJSON(response, func(object map[string]any) {
		if values, ok := object["result"].([]any); ok {
			result, found = values, true
		}
	})
	return result, found
}

func requireMemoryNote(
	request map[string]any,
	callID, name, content, description string,
	tags []string,
	ordinal int64,
) error {
	response, ok := toolResponse(request, callID)
	if !ok {
		return fmt.Errorf("Memory response %q is absent", callID)
	}
	note := fullMemoryNote(response, name)
	if note == nil {
		return fmt.Errorf("Memory response %q has no full note %q", callID, name)
	}
	if note["content"] != content || note["description"] != description {
		return fmt.Errorf("Memory response %q note = (%q, %q), want (%q, %q)",
			callID, note["content"], note["description"], content, description)
	}
	gotTags, ok := note["tags"].([]any)
	if !ok || len(gotTags) != len(tags) {
		return fmt.Errorf("Memory response %q tags = %v, want %v", callID, note["tags"], tags)
	}
	for index := range tags {
		if gotTags[index] != tags[index] {
			return fmt.Errorf("Memory response %q tags = %v, want %v", callID, gotTags, tags)
		}
	}
	gotOrdinal, ok := jsonInteger(note["ordinal"])
	if !ok || gotOrdinal != ordinal {
		return fmt.Errorf("Memory response %q ordinal = %v, want %d", callID, note["ordinal"], ordinal)
	}
	for _, field := range []string{"created_at", "updated_at"} {
		value, ok := note[field].(string)
		if !ok {
			return fmt.Errorf("Memory response %q lacks %s", callID, field)
		}
		parsed, err := time.Parse(time.RFC3339Nano, value)
		if err != nil || parsed.IsZero() {
			return fmt.Errorf("Memory response %q %s = %q, want a Server RFC3339 timestamp", callID, field, value)
		}
	}
	return rejectStorageKeys(note)
}

func fullMemoryNote(value any, name string) map[string]any {
	var note map[string]any
	walkJSON(value, func(object map[string]any) {
		if object["name"] == name && object["content"] != nil {
			note = object
		}
	})
	return note
}

func requireMemoryTimestampTransition(
	request map[string]any, firstCallID, lastCallID string, changed bool,
) error {
	firstResponse, firstOK := toolResponse(request, firstCallID)
	lastResponse, lastOK := toolResponse(request, lastCallID)
	if !firstOK || !lastOK {
		return fmt.Errorf("Memory timestamp responses %q/%q are incomplete", firstCallID, lastCallID)
	}
	var first, last map[string]any
	walkJSON(firstResponse, func(object map[string]any) {
		if object["content"] != nil && object["created_at"] != nil {
			first = object
		}
	})
	walkJSON(lastResponse, func(object map[string]any) {
		if object["content"] != nil && object["created_at"] != nil {
			last = object
		}
	})
	if first == nil || last == nil {
		return fmt.Errorf("Memory timestamp responses %q/%q contain no full notes", firstCallID, lastCallID)
	}
	firstCreated, firstCreatedErr := time.Parse(time.RFC3339Nano, fmt.Sprint(first["created_at"]))
	lastCreated, lastCreatedErr := time.Parse(time.RFC3339Nano, fmt.Sprint(last["created_at"]))
	firstUpdated, firstUpdatedErr := time.Parse(time.RFC3339Nano, fmt.Sprint(first["updated_at"]))
	lastUpdated, lastUpdatedErr := time.Parse(time.RFC3339Nano, fmt.Sprint(last["updated_at"]))
	if firstCreatedErr != nil || lastCreatedErr != nil || firstUpdatedErr != nil || lastUpdatedErr != nil ||
		!firstCreated.Equal(lastCreated) {
		return fmt.Errorf("Memory transition %q -> %q did not preserve created_at", firstCallID, lastCallID)
	}
	if changed && !lastUpdated.After(firstUpdated) {
		return fmt.Errorf("Memory transition %q -> %q did not advance updated_at", firstCallID, lastCallID)
	}
	if !changed && !lastUpdated.Equal(firstUpdated) {
		return fmt.Errorf("Memory read %q -> %q changed updated_at", firstCallID, lastCallID)
	}
	return nil
}

func jsonInteger(value any) (int64, bool) {
	switch typed := value.(type) {
	case json.Number:
		result, err := typed.Int64()
		return result, err == nil
	case float64:
		result := int64(typed)
		return result, float64(result) == typed
	case int64:
		return typed, true
	default:
		return 0, false
	}
}

func validateMemoryTranscript(request map[string]any) error {
	calls, err := messageFunctionCalls(request)
	if err != nil {
		return err
	}
	for _, call := range calls {
		if !sharedMemoryToolNames[call.Name] {
			continue
		}
		if err := rejectStorageKeys(call.Arguments); err != nil {
			return fmt.Errorf("Memory arguments for %s: %w", call.Name, err)
		}
		if response, ok := toolResponse(request, call.ID); ok {
			if err := rejectStorageKeys(response); err != nil {
				return fmt.Errorf("Memory result for %s: %w", call.Name, err)
			}
		}
	}
	return nil
}

func rejectStorageKeys(value any) error {
	var rejected string
	walkJSON(value, func(object map[string]any) {
		for key := range object {
			normalized := strings.ToLower(strings.ReplaceAll(key, "_", ""))
			switch normalized {
			case "artifact", "artifactref", "revision", "expectedrevision", "namespace", "etag":
				rejected = key
			}
		}
	})
	if rejected != "" {
		return fmt.Errorf("model-visible storage field %q", rejected)
	}
	return nil
}

func validateSharedMemoryTools(request map[string]any, scenario string, worker bool) error {
	names, err := requestToolNames(request)
	if err != nil {
		return err
	}
	plannerTools := []string{
		"add_subtask", "append_memory", "execute_current_subtask", "finish",
		"get_worker_tool_usage", "list_memories", "list_subtasks", "read_memory", "write_memory",
	}
	want := plannerTools
	if worker {
		want = []string{"append_memory", "list_memories", "read_memory", "write_memory"}
		if scenario == "router-reviewer-worker" {
			want = []string{"read_memory", "write_memory"}
		}
	}
	if !equalStringSlices(names, want) {
		return fmt.Errorf("exposed tools %v, want %v", names, want)
	}
	declarations, err := requestFunctionDeclarations(request)
	if err != nil {
		return err
	}
	for name, declaration := range declarations {
		if !sharedMemoryToolNames[name] {
			continue
		}
		parameters, _ := declaration["parameters"].(map[string]any)
		properties, _ := parameters["properties"].(map[string]any)
		if properties == nil {
			return fmt.Errorf("Memory tool %s has no object properties", name)
		}
		if _, leaked := properties["namespace"]; leaked {
			return fmt.Errorf("Memory tool %s exposes namespace", name)
		}
		workerProperty, routed := properties["worker_name"]
		if scenario != "router-planner" {
			if routed {
				return fmt.Errorf("%s Memory tool %s exposes worker_name", scenario, name)
			}
			continue
		}
		if !routed {
			return fmt.Errorf("Router Memory tool %s lacks worker_name", name)
		}
		property, _ := workerProperty.(map[string]any)
		rawEnum, _ := property["enum"].([]any)
		workers := make([]string, 0, len(rawEnum))
		for _, value := range rawEnum {
			if text, ok := value.(string); ok {
				workers = append(workers, text)
			}
		}
		sort.Strings(workers)
		wantWorkers := []string{"builder", "reviewer"}
		if name == "append_memory" || name == "list_memories" {
			wantWorkers = []string{"builder"}
		}
		if !equalStringSlices(workers, wantWorkers) {
			return fmt.Errorf("Router %s worker enum = %v, want %v", name, workers, wantWorkers)
		}
	}
	return nil
}

func requestFunctionDeclarations(request map[string]any) (map[string]map[string]any, error) {
	raw, ok := request["tools"].([]any)
	if !ok {
		return nil, errors.New("model-visible tool set is absent")
	}
	result := make(map[string]map[string]any, len(raw))
	for _, item := range raw {
		tool, ok := item.(map[string]any)
		if !ok {
			return nil, errors.New("invalid tool declaration")
		}
		function, ok := tool["function"].(map[string]any)
		if !ok {
			return nil, errors.New("invalid function declaration")
		}
		name, _ := function["name"].(string)
		if name == "" {
			return nil, errors.New("unnamed function declaration")
		}
		result[name] = function
	}
	return result, nil
}

func (g *sharedMemoryGateway) writeFailure(w http.ResponseWriter, status int, message string) {
	g.mu.Lock()
	g.failures = append(g.failures, message)
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": message, "type": "invalid_request_error"},
	})
}
