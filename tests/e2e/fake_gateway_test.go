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
	"time"
)

type fakeGateway struct {
	server    *httptest.Server
	token     string
	copyTools []string

	mu       sync.Mutex
	calls    int
	failures []string
}

func newFakeGateway(token string) *fakeGateway {
	return newFakeGatewayWithCopyTools(token, []string{"read_artifact", "write_artifact"})
}

func newCapabilityGateway(token string) *fakeGateway {
	return newFakeGatewayWithCopyTools(
		token,
		[]string{"read_artifact", "validate_likec4", "write_artifact"},
	)
}

func newFakeGatewayWithCopyTools(token string, tools []string) *fakeGateway {
	gateway := &fakeGateway{token: token, copyTools: append([]string(nil), tools...)}
	gateway.server = httptest.NewServer(http.HandlerFunc(gateway.serveHTTP))
	return gateway
}

func (g *fakeGateway) close() { g.server.Close() }

func (g *fakeGateway) URL() string { return g.server.URL + "/v1" }

func (g *fakeGateway) Calls() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.calls
}

func (g *fakeGateway) ResetScenario() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.calls = 0
}

func (g *fakeGateway) Failures() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.failures...)
}

func (g *fakeGateway) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		g.fail(w, http.StatusNotFound, "unsupported fake gateway endpoint")
		return
	}
	if r.Header.Get("Authorization") != "Bearer "+g.token {
		g.fail(w, http.StatusUnauthorized, "invalid fake gateway token")
		return
	}
	defer r.Body.Close()
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	decoder.UseNumber()
	var request map[string]any
	if err := decoder.Decode(&request); err != nil {
		g.fail(w, http.StatusBadRequest, "invalid OpenAI request")
		return
	}
	g.mu.Lock()
	nextCall := g.calls + 1
	g.mu.Unlock()
	expectedTools := append([]string(nil), g.copyTools...)
	if nextCall > 3 {
		expectedTools = []string{"read_artifact"}
	}
	if !selectedTools(request, expectedTools...) {
		g.fail(w, http.StatusBadRequest, "unexpected model-visible tool set")
		return
	}

	g.mu.Lock()
	g.calls++
	call := g.calls
	g.mu.Unlock()

	var message map[string]any
	switch call {
	case 1:
		message = toolCallMessage("read-1", "read_artifact", map[string]any{
			"namespace": "inputs", "name": "source", "revision": nil,
		})
	case 2:
		data, ok := lastStringValue(request, "dataBase64")
		if !ok || data == "" {
			g.fail(w, http.StatusBadRequest, "read_artifact result was not returned to the model")
			return
		}
		message = toolCallMessage("write-1", "write_artifact", map[string]any{
			"namespace": "builder", "name": "copied", "media_type": "text/plain",
			"data_base64": data, "expected_revision": nil,
		})
	case 3:
		_, ok := lastExactArtifact(request, "builder", "copied")
		if !ok {
			g.fail(w, http.StatusBadRequest, "write_artifact exact result was not returned to the model")
			return
		}
		finalMessage, err := workerModelResultMessage(request, "Source artifact copied byte-for-byte")
		if err != nil {
			g.fail(w, http.StatusBadRequest, err.Error())
			return
		}
		message = finalMessage
	default:
		message = toolCallMessage(fmt.Sprintf("bounded-read-%d", call), "read_artifact", map[string]any{
			"namespace": "inputs", "name": "source", "revision": nil,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":      fmt.Sprintf("chatcmpl-e2e-%d", call),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   "worker-model",
		"choices": []any{map[string]any{
			"index": 0, "message": message,
			"finish_reason": map[bool]string{true: "stop", false: "tool_calls"}[call == 3],
		}},
		"usage": map[string]any{
			"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
		},
	})
}

// workerModelResultMessage mirrors the only model-facing Worker result schema.
// The requested subtask ID is deliberately copied from the Runtime-rendered task
// prompt so process fixtures exercise the same correlation check as a real model.
func workerModelResultMessage(request map[string]any, result string) (map[string]any, error) {
	subtaskID, err := workerRequestSubtaskID(request)
	if err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(map[string]any{
		"subtaskId": subtaskID,
		"result":    result,
	})
	if err != nil {
		return nil, fmt.Errorf("encode WorkerModelResult: %w", err)
	}
	return map[string]any{"role": "assistant", "content": string(encoded)}, nil
}

func workerRequestSubtaskID(request map[string]any) (string, error) {
	const marker = "Subtask ID:\n"
	var found string
	conflicting := false
	var visit func(any)
	visit = func(value any) {
		switch typed := value.(type) {
		case map[string]any:
			for _, child := range typed {
				visit(child)
			}
		case []any:
			for _, child := range typed {
				visit(child)
			}
		case string:
			index := strings.Index(typed, marker)
			if index < 0 {
				return
			}
			candidate := typed[index+len(marker):]
			if end := strings.Index(candidate, "\n\n"); end >= 0 {
				candidate = candidate[:end]
			}
			candidate = strings.TrimSpace(candidate)
			if validFixtureSubtaskID(candidate) {
				if found != "" && found != candidate {
					conflicting = true
				} else {
					found = candidate
				}
			}
		}
	}
	visit(request)
	if conflicting {
		return "", errors.New("Worker request contains conflicting Runtime-rendered subtask IDs")
	}
	if found == "" {
		return "", errors.New("Worker request has no valid Runtime-rendered subtask ID")
	}
	return found, nil
}

func validFixtureSubtaskID(value string) bool {
	if len(value) == 0 || len(value) > 128 {
		return false
	}
	for index := 0; index < len(value); index++ {
		character := value[index]
		if character >= 'A' && character <= 'Z' || character >= 'a' && character <= 'z' ||
			character >= '0' && character <= '9' {
			continue
		}
		if index == 0 || !strings.ContainsRune("._:-", rune(character)) {
			return false
		}
	}
	return true
}

func (g *fakeGateway) fail(w http.ResponseWriter, status int, message string) {
	g.mu.Lock()
	g.failures = append(g.failures, message)
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": message, "type": "invalid_request_error"},
	})
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

func selectedTools(request map[string]any, expected ...string) bool {
	raw, ok := request["tools"].([]any)
	if !ok || len(raw) != len(expected) {
		return false
	}
	seen := make(map[string]bool, len(raw))
	for _, item := range raw {
		tool, ok := item.(map[string]any)
		if !ok {
			return false
		}
		function, ok := tool["function"].(map[string]any)
		if !ok {
			return false
		}
		name, ok := function["name"].(string)
		if !ok {
			return false
		}
		seen[name] = true
	}
	for _, name := range expected {
		if !seen[name] {
			return false
		}
	}
	return true
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
		}
	}
}
