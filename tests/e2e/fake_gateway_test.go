//go:build e2e

package e2e

import (
	"encoding/json"
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
		artifact, ok := lastExactArtifact(request, "builder", "copied")
		if !ok {
			g.fail(w, http.StatusBadRequest, "write_artifact exact result was not returned to the model")
			return
		}
		result, _ := json.Marshal(map[string]any{
			"apiVersion": "contractor/v1alpha1",
			"outcome":    "succeeded",
			"summary":    "Source artifact copied byte-for-byte",
			"artifacts":  map[string]any{"copied": artifact},
		})
		message = map[string]any{"role": "assistant", "content": string(result)}
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
