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
	"time"
)

const summarizerTranscriptCanary = "ephemeral-worker-summary-transcript-canary"

type summarizerGatewayCapture struct {
	Mode              string
	Model             string
	Tools             []string
	ResponseSchema    bool
	MaxOutputTokens   int
	TranscriptPresent bool
	SecretRedacted    bool
}

// summarizerGateway is an offline OpenAI-compatible boundary fixture. It
// deliberately retains only safe request metadata; provider request bodies
// and credentials are inspected in-flight and then discarded.
type summarizerGateway struct {
	server *httptest.Server
	token  string

	mu             sync.Mutex
	normalCalls    map[string]int
	summaryCalls   map[string]int
	captures       []summarizerGatewayCapture
	failures       []string
	summaryStarted chan string
	releaseBlocked chan struct{}
	releaseOnce    sync.Once
}

func newSummarizerGateway(token string) *summarizerGateway {
	gateway := &summarizerGateway{
		token: token, normalCalls: map[string]int{}, summaryCalls: map[string]int{},
		summaryStarted: make(chan string, 16), releaseBlocked: make(chan struct{}),
	}
	gateway.server = httptest.NewServer(http.HandlerFunc(gateway.serveHTTP))
	return gateway
}

func (g *summarizerGateway) URL() string { return g.server.URL + "/v1" }

func (g *summarizerGateway) close() {
	g.releaseOnce.Do(func() { close(g.releaseBlocked) })
	g.server.Close()
}

func (g *summarizerGateway) Failures() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.failures...)
}

func (g *summarizerGateway) Counts(mode string) (normal, summary int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.normalCalls[mode], g.summaryCalls[mode]
}

func (g *summarizerGateway) Captures() []summarizerGatewayCapture {
	g.mu.Lock()
	defer g.mu.Unlock()
	result := make([]summarizerGatewayCapture, len(g.captures))
	for index, capture := range g.captures {
		result[index] = capture
		result[index].Tools = append([]string(nil), capture.Tools...)
	}
	return result
}

func (g *summarizerGateway) WaitForSummary(
	ctx context.Context,
	mode string,
) error {
	for {
		select {
		case observed := <-g.summaryStarted:
			if observed == mode {
				return nil
			}
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func (g *summarizerGateway) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		g.fail(w, http.StatusNotFound, "unsupported summarizer Gateway endpoint")
		return
	}
	if r.Header.Get("Authorization") != "Bearer "+g.token {
		g.fail(w, http.StatusUnauthorized, "invalid summarizer Gateway credential")
		return
	}
	defer r.Body.Close()
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20))
	decoder.UseNumber()
	var request map[string]any
	if err := decoder.Decode(&request); err != nil {
		g.fail(w, http.StatusBadRequest, "invalid summarizer Gateway request")
		return
	}
	mode, err := summarizerRequestMode(request)
	if err != nil {
		g.fail(w, http.StatusBadRequest, err.Error())
		return
	}
	model, ok := request["model"].(string)
	if !ok || model == "" {
		g.fail(w, http.StatusBadRequest, "summarizer Gateway request omitted model")
		return
	}
	model = strings.TrimPrefix(model, "openai/")
	isSummary := model == "worker-summarizer-model"
	if !isSummary && model != "worker-model" {
		g.fail(w, http.StatusBadRequest, "unexpected summarizer Gateway model alias")
		return
	}
	tools, err := optionalRequestToolNames(request)
	if err != nil {
		g.fail(w, http.StatusBadRequest, err.Error())
		return
	}
	if isSummary && len(tools) != 0 {
		g.fail(w, http.StatusBadRequest, "terminal summarizer received model-visible tools")
		return
	}
	if !isSummary && !equalStringSlices(tools, []string{"write_text_artifact"}) {
		g.fail(w, http.StatusBadRequest, fmt.Sprintf("normal Worker exposed tools %v", tools))
		return
	}

	encoded, err := json.Marshal(request)
	if err != nil {
		g.fail(w, http.StatusBadRequest, "summarizer Gateway request could not be inspected")
		return
	}
	responseSchema := request["response_format"] != nil
	maxOutputTokens := requestInteger(request, "max_completion_tokens", "max_tokens")
	capture := summarizerGatewayCapture{
		Mode: mode, Model: model, Tools: append([]string(nil), tools...),
		ResponseSchema: responseSchema, MaxOutputTokens: maxOutputTokens,
		TranscriptPresent: strings.Contains(string(encoded), summarizerTranscriptCanary),
		SecretRedacted: !strings.Contains(string(encoded), g.token) &&
			strings.Contains(string(encoded), "[REDACTED]"),
	}

	g.mu.Lock()
	if isSummary {
		g.summaryCalls[mode]++
	} else {
		g.normalCalls[mode]++
	}
	normalCall := g.normalCalls[mode]
	summaryCall := g.summaryCalls[mode]
	g.captures = append(g.captures, capture)
	g.mu.Unlock()

	if isSummary {
		if summaryCall != 1 {
			g.fail(w, http.StatusBadRequest, "terminal summarizer was invoked more than once")
			return
		}
		if !responseSchema || maxOutputTokens != 2048 {
			g.fail(w, http.StatusBadRequest, "terminal summarizer policy was not pinned")
			return
		}
		if !capture.TranscriptPresent || !capture.SecretRedacted {
			g.fail(w, http.StatusBadRequest, "terminal summarizer projection was missing or unsafe")
			return
		}
		select {
		case g.summaryStarted <- mode:
		default:
			g.fail(w, http.StatusInternalServerError, "terminal summarizer signal queue overflow")
			return
		}
		if mode == "cancel-summary" {
			select {
			case <-r.Context().Done():
				return
			case <-g.releaseBlocked:
			}
		}
		if mode == "provider-timeout" {
			g.scriptedProviderFailure(w)
			return
		}
		if mode == "invalid-summary" {
			g.writeCompletion(w, model, map[string]any{
				"role": "assistant", "content": "not a structured Worker result",
			}, "stop", summarizerUsage(13, 5, 18))
			return
		}
		message, resultErr := summarizerResultMessage(request, "Terminal summary for "+mode)
		if resultErr != nil {
			g.fail(w, http.StatusBadRequest, resultErr.Error())
			return
		}
		g.writeCompletion(w, model, message, "stop", summarizerUsage(13, 5, 18))
		return
	}

	if normalCall == 1 {
		message := toolCallMessage("summary-write-"+mode, "write_text_artifact", map[string]any{
			"name": "result", "text": "artifact produced for " + mode,
			"media_type": "text/plain", "expected_revision": nil,
		})
		// This text is model-visible history, but neither a result nor an
		// artifact. It proves that the bounded transcript reaches only the live
		// summarizer request. The configured Gateway token proves value redaction.
		message["content"] = summarizerTranscriptCanary + " " + g.token
		usage := normalUsage(mode, normalCall)
		g.writeCompletion(w, model, message, "tool_calls", usage)
		return
	}
	if normalCall != 2 || !normalModeCompletesNormally(mode) {
		g.fail(w, http.StatusBadRequest, "unexpected additional normal Worker invocation")
		return
	}
	message, resultErr := workerModelResultMessage(request, "Normal Worker result for "+mode)
	if resultErr != nil {
		g.fail(w, http.StatusBadRequest, resultErr.Error())
		return
	}
	g.writeCompletion(w, model, message, "stop", normalUsage(mode, normalCall))
}

func (g *summarizerGateway) writeCompletion(
	w http.ResponseWriter,
	model string,
	message map[string]any,
	finishReason string,
	usage map[string]any,
) {
	w.Header().Set("Content-Type", "application/json")
	payload := map[string]any{
		"id": "chatcmpl-worker-summarizer", "object": "chat.completion",
		"created": time.Now().Unix(), "model": model,
		"choices": []any{map[string]any{
			"index": 0, "message": message, "finish_reason": finishReason,
		}},
	}
	if usage != nil {
		payload["usage"] = usage
	}
	_ = json.NewEncoder(w).Encode(payload)
}

func (g *summarizerGateway) scriptedProviderFailure(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusGatewayTimeout)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": "deterministic upstream timeout", "type": "timeout_error",
		},
	})
}

func (g *summarizerGateway) fail(w http.ResponseWriter, status int, message string) {
	g.mu.Lock()
	g.failures = append(g.failures, message)
	g.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": message, "type": "invalid_request_error"},
	})
}

func normalUsage(mode string, call int) map[string]any {
	switch mode {
	case "cumulative", "invalid-summary", "cancel-summary", "provider-timeout":
		return summarizerUsage(100, 19_900, 20_000)
	case "context-window":
		return summarizerUsage(7168, 3, 7171)
	case "normal-final", "reuse-after-failure", "reuse-after-cancel":
		if call == 1 {
			return summarizerUsage(7, 3, 10)
		}
		return summarizerUsage(19_987, 3, 19_990)
	case "missing-usage":
		if call == 1 {
			return nil
		}
		return summarizerUsage(7, 3, 10)
	case "disabled":
		if call == 1 {
			return summarizerUsage(100, 24_900, 25_000)
		}
		return summarizerUsage(7, 3, 10)
	default:
		return summarizerUsage(7, 3, 10)
	}
}

func summarizerUsage(prompt, completion, total int) map[string]any {
	return map[string]any{
		"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total,
	}
}

func normalModeCompletesNormally(mode string) bool {
	switch mode {
	case "normal-final", "missing-usage", "disabled", "reuse-after-failure", "reuse-after-cancel":
		return true
	default:
		return false
	}
}

func summarizerRequestMode(request map[string]any) (string, error) {
	allowed := map[string]bool{
		"cumulative": true, "context-window": true, "normal-final": true,
		"missing-usage": true, "disabled": true, "invalid-summary": true,
		"cancel-summary": true, "provider-timeout": true,
		"reuse-after-failure": true, "reuse-after-cancel": true,
	}
	values := map[string]bool{}
	walkDomainJSON(request, func(object map[string]any) {
		if parameters, ok := object["parameters"].(map[string]any); ok {
			if mode, ok := parameters["mode"].(string); ok && allowed[mode] {
				values[mode] = true
			}
		}
		if mode, ok := object["mode"].(string); ok && allowed[mode] {
			values[mode] = true
		}
	})
	if len(values) != 1 {
		return "", fmt.Errorf("summarizer Gateway request has %d recognized modes", len(values))
	}
	for value := range values {
		return value, nil
	}
	return "", errors.New("summarizer Gateway request omitted mode")
}

func summarizerResultMessage(request map[string]any, result string) (map[string]any, error) {
	var subtaskID string
	conflicting := false
	walkDomainJSON(request, func(object map[string]any) {
		candidate, ok := object["subtaskId"].(string)
		if !ok || !validFixtureSubtaskID(candidate) {
			return
		}
		if subtaskID != "" && candidate != subtaskID {
			conflicting = true
			return
		}
		subtaskID = candidate
	})
	if conflicting || subtaskID == "" {
		return nil, errors.New("terminal summarizer request has no unique valid subtaskId")
	}
	encoded, err := json.Marshal(map[string]any{"subtaskId": subtaskID, "result": result})
	if err != nil {
		return nil, err
	}
	return map[string]any{"role": "assistant", "content": string(encoded)}, nil
}

func optionalRequestToolNames(request map[string]any) ([]string, error) {
	raw, exists := request["tools"]
	if !exists || raw == nil {
		return []string{}, nil
	}
	values, ok := raw.([]any)
	if !ok {
		return nil, errors.New("invalid model-visible tool set")
	}
	if len(values) == 0 {
		return []string{}, nil
	}
	names, err := requestToolNames(request)
	if err != nil {
		return nil, err
	}
	sort.Strings(names)
	return names, nil
}

func requestInteger(request map[string]any, names ...string) int {
	for _, name := range names {
		switch value := request[name].(type) {
		case json.Number:
			parsed, err := value.Int64()
			if err == nil {
				return int(parsed)
			}
		case float64:
			return int(value)
		case int:
			return value
		}
	}
	return 0
}
