package streamline

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/requestid"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

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
	if gotErr == nil || !strings.Contains(gotErr.Error(), "HTTP 307") || targetCalled {
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
