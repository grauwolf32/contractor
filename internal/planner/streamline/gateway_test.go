package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	"github.com/grauwolf32/contractor/internal/planner"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

func TestSharedGatewayResponseByteCeiling(t *testing.T) {
	var fixture struct {
		MaxResponseBytes int `json:"max_response_bytes"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "gateway-response-limits.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if maxGatewayResponseBytes != fixture.MaxResponseBytes {
		t.Fatalf("Gateway response limit = %d, shared fixture = %d", maxGatewayResponseBytes, fixture.MaxResponseBytes)
	}
}

func TestOpenAICompatibleModelConvertsADKToolConversation(t *testing.T) {
	const token = "sk-gateway-test"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer "+token ||
			!requestid.Valid(r.Header.Get(requestid.Header)) {
			t.Fatalf("request path=%q headers=%v", r.URL.Path, r.Header)
		}
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		if payload["model"] != "planner-model" || payload["parallel_tool_calls"] != false {
			t.Fatalf("payload = %+v", payload)
		}
		if _, exists := payload["response_format"]; exists {
			t.Error("ordinary Planner request acquired an output schema")
		}
		messages, _ := payload["messages"].([]any)
		tools, _ := payload["tools"].([]any)
		if len(messages) != 2 || len(tools) != 1 {
			t.Fatalf("messages=%+v tools=%+v", messages, tools)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
  "model":"planner-model",
  "choices":[{"finish_reason":"tool_calls","message":{"content":"","tool_calls":[{
    "id":"call-finish","type":"function","function":{"name":"finish","arguments":"{\"outcome\":\"succeeded\",\"summary\":\"done\",\"artifacts\":{}}"}
  }]}}],
  "usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
}`))
	}))
	defer server.Close()

	llm, err := NewOpenAICompatibleModel(GatewaySettings{
		URL: server.URL + "/v1", Token: contracts.NewSecretString(token), Model: "planner-model",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	request := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("stage context", genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("planner instructions", genai.RoleUser),
			Tools: []*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name: "finish", Description: "finish the Stage",
				ParametersJsonSchema: map[string]any{"type": "object"},
			}}}},
		},
	}
	var response *model.LLMResponse
	for current, currentErr := range llm.GenerateContent(t.Context(), request, false) {
		if currentErr != nil {
			t.Fatal(currentErr)
		}
		response = current
	}
	if response == nil || response.Content == nil || len(response.Content.Parts) != 1 ||
		response.Content.Parts[0].FunctionCall == nil ||
		response.Content.Parts[0].FunctionCall.Name != "finish" ||
		response.UsageMetadata == nil || response.UsageMetadata.TotalTokenCount != 18 {
		t.Fatalf("response = %+v", response)
	}
}

func TestOpenAICompatibleModelOmitsAuthorizationForUnauthenticatedGateway(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if values := r.Header.Values("Authorization"); len(values) != 0 {
			t.Fatalf("unauthenticated request Authorization = %v", values)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
  "model":"local-model",
  "choices":[{"finish_reason":"stop","message":{"content":"done"}}],
  "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
}`))
	}))
	defer server.Close()
	llm, err := NewOpenAICompatibleModel(GatewaySettings{
		URL: server.URL + "/v1", Model: "local-model", HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	request := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)},
	}
	var got *model.LLMResponse
	for response, currentErr := range llm.GenerateContent(t.Context(), request, false) {
		if currentErr != nil {
			t.Fatal(currentErr)
		}
		got = response
	}
	if got == nil || got.Content == nil || got.Content.Parts[0].Text != "done" {
		t.Fatalf("response = %+v", got)
	}
}

func TestOpenAICompatibleModelRedactsProviderErrorBodyAndToken(t *testing.T) {
	const secret = "sk-secret-embedded-by-provider"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":"credential `+secret+` rejected"}`, http.StatusBadGateway)
	}))
	defer server.Close()
	llm, err := NewOpenAICompatibleModel(GatewaySettings{
		URL: server.URL, Token: contracts.NewSecretString(secret), Model: "planner-model",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	request := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)},
		Config:   &genai.GenerateContentConfig{},
	}
	var gotErr error
	for _, currentErr := range llm.GenerateContent(context.Background(), request, false) {
		gotErr = currentErr
	}
	if gotErr == nil || strings.Contains(gotErr.Error(), secret) ||
		strings.Contains(gotErr.Error(), "credential") {
		t.Fatalf("unsafe gateway error = %v", gotErr)
	}
}

func TestOpenAICompatibleModelDoesNotFollowGatewayRedirect(t *testing.T) {
	targetCalled := false
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		targetCalled = true
	}))
	defer target.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()
	llm, err := NewOpenAICompatibleModel(GatewaySettings{
		URL: redirect.URL, Token: contracts.NewSecretString("redirect-token"), Model: "planner-model",
		HTTPClient: redirect.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	request := &model.LLMRequest{Contents: []*genai.Content{
		genai.NewContentFromText("hello", genai.RoleUser),
	}}
	var gotErr error
	for _, currentErr := range llm.GenerateContent(t.Context(), request, false) {
		gotErr = currentErr
	}
	var failure *gatewayrecovery.FailureError
	if !errors.As(gotErr, &failure) || failure.Failure != (gatewayrecovery.Failure{Code: "gateway_request_rejected"}) ||
		targetCalled {
		t.Fatalf("redirect error=%v targetCalled=%v", gotErr, targetCalled)
	}
}

func TestOpenAICompatibleModelRejectsOversizedRequestBeforeTransport(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		called = true
	}))
	defer server.Close()
	llm, err := NewOpenAICompatibleModel(GatewaySettings{
		URL: server.URL, Token: contracts.NewSecretString("test-token"), Model: "planner-model",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	request := &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText(
		strings.Repeat("x", maxGatewayRequestBytes), genai.RoleUser,
	)}}
	var gotErr error
	for _, currentErr := range llm.GenerateContent(t.Context(), request, false) {
		gotErr = currentErr
	}
	if gotErr == nil || !strings.Contains(gotErr.Error(), "oversized") || called {
		t.Fatalf("oversized request error=%v transportCalled=%v", gotErr, called)
	}
}

func TestDecodeChatResponseRequiresProviderTokenUsage(t *testing.T) {
	_, err := decodeChatResponse([]byte(`{
  "model":"planner-model",
  "choices":[{"message":{"content":"text","tool_calls":[]}}]
}`))
	if err == nil || !strings.Contains(err.Error(), "token usage") {
		t.Fatalf("missing usage error = %v", err)
	}
}

func TestPlannerChargesAndClassifiesRejectedGatewayResponses(t *testing.T) {
	tests := []struct {
		name, finishReason, arguments, wantCode string
		includeTool, omitUsage, badUsage        bool
		status                                  int
		maxTokens                               int
		wantInput, wantOutput                   int64
	}{
		{name: "length with truncated tool", finishReason: "length", arguments: "{", includeTool: true,
			wantCode: "planner_gateway_invalid_response", wantInput: 40, wantOutput: 50},
		{name: "length with valid tool", finishReason: "length", arguments: `{"subtask_id":"0"}`, includeTool: true,
			wantCode: "planner_gateway_invalid_response", wantInput: 40, wantOutput: 50},
		{name: "length with empty choice", finishReason: "length",
			wantCode: "planner_gateway_invalid_response", wantInput: 40, wantOutput: 50},
		{name: "malformed tool arguments", finishReason: "tool_calls", arguments: "{", includeTool: true,
			wantCode: "planner_gateway_invalid_response", wantInput: 40, wantOutput: 50},
		{name: "duplicate tool argument", finishReason: "tool_calls", arguments: `{"subtask_id":"1","subtask_id":"0"}`, includeTool: true,
			wantCode: "planner_gateway_invalid_response", wantInput: 40, wantOutput: 50},
		{name: "escaped duplicate tool argument", finishReason: "tool_calls", arguments: `{"subtask_id":"1","subtask_\u0069d":"0"}`, includeTool: true,
			wantCode: "planner_gateway_invalid_response", wantInput: 40, wantOutput: 50},
		{name: "nested duplicate tool argument", finishReason: "tool_calls", arguments: `{"subtask_id":"0","extra":{"x":1,"x":2}}`, includeTool: true,
			wantCode: "planner_gateway_invalid_response", wantInput: 40, wantOutput: 50},
		{name: "trailing tool argument", finishReason: "tool_calls", arguments: `{"subtask_id":"0"} {}`, includeTool: true,
			wantCode: "planner_gateway_invalid_response", wantInput: 40, wantOutput: 50},
		{name: "null tool arguments", finishReason: "tool_calls", arguments: "null", includeTool: true,
			wantCode: "planner_gateway_invalid_response", wantInput: 40, wantOutput: 50},
		{name: "missing usage", finishReason: "stop", omitUsage: true,
			wantCode: "planner_gateway_invalid_response", wantInput: 10, wantOutput: 10},
		{name: "inconsistent usage", finishReason: "stop", badUsage: true,
			wantCode: "planner_gateway_invalid_response", wantInput: 10, wantOutput: 10},
		{name: "rejected response exceeds token budget", finishReason: "length", arguments: "{", includeTool: true,
			maxTokens: 80, wantCode: "planner_token_limit", wantInput: 40, wantOutput: 50},
		{name: "HTTP status remains unavailable", status: http.StatusBadGateway,
			wantCode: "planner_gateway_unavailable", wantInput: 10, wantOutput: 10},
		{name: "rate limit remains unavailable", status: http.StatusTooManyRequests,
			wantCode: "planner_gateway_unavailable", wantInput: 10, wantOutput: 10},
		{name: "denied access is rejected", status: http.StatusUnauthorized,
			wantCode: "planner_gateway_rejected", wantInput: 10, wantOutput: 10},
		{name: "invalid request is rejected", status: http.StatusUnprocessableEntity,
			wantCode: "planner_gateway_rejected", wantInput: 10, wantOutput: 10},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				if calls.Add(1) == 1 {
					_ = json.NewEncoder(w).Encode(map[string]any{
						"model": "planner-model",
						"choices": []any{map[string]any{
							"finish_reason": "tool_calls",
							"message": map[string]any{"tool_calls": []any{map[string]any{
								"id": "call-add", "type": "function",
								"function": map[string]any{"name": "add_subtask", "arguments": `{"objective":"Build","instructions":"Produce the report"}`},
							}}},
						}},
						"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
					})
					return
				}
				if test.status != 0 {
					w.WriteHeader(test.status)
					return
				}
				message := map[string]any{}
				if test.includeTool {
					message["tool_calls"] = []any{map[string]any{
						"id": "call-execute", "type": "function",
						"function": map[string]any{"name": executeCurrentSubtaskToolName, "arguments": test.arguments},
					}}
				}
				response := map[string]any{
					"model":   "planner-model",
					"choices": []any{map[string]any{"finish_reason": test.finishReason, "message": message}},
				}
				if !test.omitUsage {
					total := 70
					if test.badUsage {
						total = 1
					}
					response["usage"] = map[string]any{"prompt_tokens": 30, "completion_tokens": 40, "total_tokens": total}
				}
				_ = json.NewEncoder(w).Encode(response)
			}))
			t.Cleanup(server.Close)
			llm, err := NewOpenAICompatibleModel(GatewaySettings{
				URL: server.URL + "/v1", Model: "planner-model", HTTPClient: server.Client(),
			})
			if err != nil {
				t.Fatal(err)
			}
			workers := &fakeWorkerInvoker{}
			invocation := testInvocation("builder")
			if test.maxTokens != 0 {
				invocation.ModelAccess.ModelPolicy.MaxTotalTokens = test.maxTokens
			}
			instance, err := mustFactory(t, newFakeSessions(), workers, &fakeInspector{}, llm, Limits{}).Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			_, err = instance.Run(t.Context())
			assertPlannerCode(t, err, test.wantCode)
			if retryable := planner.FailureFrom(err).Retryable; retryable == (test.wantCode == "planner_gateway_rejected") {
				t.Fatalf("%s retryable = %t", test.wantCode, retryable)
			}
			report, ok := instance.(*streamlinePlanner).ExecutionReport()
			if !ok || report.Metrics.ModelCalls == nil || *report.Metrics.ModelCalls != 2 ||
				report.Metrics.InputTokens == nil || *report.Metrics.InputTokens != test.wantInput ||
				report.Metrics.OutputTokens == nil || *report.Metrics.OutputTokens != test.wantOutput ||
				report.Metrics.TotalTokens == nil || *report.Metrics.TotalTokens != test.wantInput+test.wantOutput ||
				calls.Load() != 2 || len(workers.calls) != 0 {
				t.Fatalf("report=%+v Gateway calls=%d Worker calls=%d", report, calls.Load(), len(workers.calls))
			}
		})
	}
}

func TestConvertContentsGeneratesUniqueSyntheticToolCallIDs(t *testing.T) {
	firstCall := genai.NewContentFromFunctionCall("worker_first", map[string]any{}, genai.RoleModel)
	firstResult := genai.NewContentFromFunctionResponse("worker_first", map[string]any{"ok": true}, genai.RoleUser)
	secondCall := genai.NewContentFromFunctionCall("worker_second", map[string]any{}, genai.RoleModel)
	secondResult := genai.NewContentFromFunctionResponse("worker_second", map[string]any{"ok": true}, genai.RoleUser)

	messages, err := convertContents(&model.LLMRequest{Contents: []*genai.Content{
		firstCall, firstResult, secondCall, secondResult,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 4 || len(messages[0].ToolCalls) != 1 || len(messages[2].ToolCalls) != 1 ||
		messages[0].ToolCalls[0].ID != "call_1" || messages[1].ToolCallID != "call_1" ||
		messages[2].ToolCalls[0].ID != "call_2" || messages[3].ToolCallID != "call_2" {
		t.Fatalf("synthetic call correlation = %+v", messages)
	}
}

// The compactor path keeps the provider's explicit finish reason and the
// valid usage of every received reply; the owner classifies the reply.
func TestOpenAICompatibleCompactorKeepsFinishReasonAndUsage(t *testing.T) {
	for _, test := range []struct {
		name, finish string
		usage        bool
	}{
		{"stop", `"stop"`, true},
		{"length", `"length"`, true},
		{"content-filter", `"content_filter"`, true},
		{"missing", `null`, true},
		{"unknown", `"paused"`, true},
		{"missing-usage", `"stop"`, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				var payload map[string]any
				if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
					t.Error(err)
				}
				if payload["model"] != "compactor-model" || payload["tools"] != nil || payload["max_tokens"] != float64(256) {
					t.Errorf("compactor payload = %+v", payload)
				}
				usage := `,"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}`
				if !test.usage {
					usage = ""
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"finish_reason":` + test.finish +
					`,"message":{"content":"{\"summary\":\"memory\"}"}}]` + usage + `}`))
			}))
			defer server.Close()
			llm, err := NewOpenAICompatibleModel(GatewaySettings{
				URL: server.URL + "/v1", Model: "compactor-model", MaxOutputTokens: 256, HTTPClient: server.Client(),
			})
			if err != nil {
				t.Fatal(err)
			}
			compactor, ok := llm.(compactionModel)
			if !ok {
				t.Fatal("Gateway model has no compactor path")
			}
			reply, err := compactor.compact(t.Context(), &model.LLMRequest{
				Contents: []*genai.Content{genai.NewContentFromText("history", genai.RoleUser)},
			})
			if requests.Load() != 1 {
				t.Fatalf("requests = %d", requests.Load())
			}
			if !test.usage {
				var rejected *gatewayResponseRejected
				if !errors.As(err, &rejected) || rejected.usage != nil {
					t.Fatalf("missing usage error = %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			want := strings.Trim(test.finish, `"`)
			if test.finish == "null" {
				want = ""
			}
			if reply.finishReason != want || reply.text != `{"summary":"memory"}` || reply.toolCalls != 0 ||
				reply.usage == nil || reply.usage.TotalTokenCount != 18 {
				t.Fatalf("reply = %+v", reply)
			}
		})
	}
}

func TestOpenAICompatibleGatewayEnvelopeIsUnambiguous(t *testing.T) {
	const valid = `{"choices":[{"finish_reason":"stop","message":{"content":"done"}}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}`
	for _, test := range []struct {
		name, payload string
		valid         bool
	}{
		{"provider extensions", `{"id":"response-1","provider":{"extra":true},` + valid[1:], true},
		{"duplicate top-level key", `{"model":"first","model":"second",` + valid[1:], false},
		{"escaped duplicate key", `{"model":"first","mo\u0064el":"second",` + valid[1:], false},
		{"duplicate usage", strings.Replace(valid, `"prompt_tokens":11`, `"prompt_tokens":1,"prompt_tokens":11`, 1), false},
		{"duplicate choice field", strings.Replace(valid, `"content":"done"`, `"content":"wrong","content":"done"`, 1), false},
		{"duplicate extension field", `{"provider":{"extra":false,"extra":true},` + valid[1:], false},
		{"trailing object", valid + `{}`, false},
		{"trailing garbage", valid + `invalid`, false},
	} {
		for _, path := range []string{"main", "compactor"} {
			t.Run(test.name+"/"+path, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(test.payload))
				}))
				defer server.Close()
				llm, err := NewOpenAICompatibleModel(GatewaySettings{
					URL: server.URL, Model: "planner-model", HTTPClient: server.Client(),
				})
				if err != nil {
					t.Fatal(err)
				}
				request := &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("context", genai.RoleUser)}}
				if path == "compactor" {
					_, err = llm.(compactionModel).compact(t.Context(), request)
				} else {
					for _, currentErr := range llm.GenerateContent(t.Context(), request, false) {
						err = currentErr
					}
				}
				if test.valid {
					if err != nil {
						t.Fatalf("valid provider extension rejected: %v", err)
					}
					return
				}
				var rejected *gatewayResponseRejected
				if !errors.As(err, &rejected) || rejected.usage != nil {
					t.Fatalf("ambiguous envelope = %v, want rejection without trusted usage", err)
				}
			})
		}
	}
}
