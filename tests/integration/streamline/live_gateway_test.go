package streamline_test

import (
	"os"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner/streamline"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// TestLiveGatewayToolCall is opt-in because it needs a running model. It is
// intentionally small: deterministic lifecycle semantics remain covered by
// fake-model tests, while this verifies the deployed LiteLLM/LM Studio tool
// calling dialect.
func TestLiveGatewayToolCall(t *testing.T) {
	url := os.Getenv("CONTRACTOR_STREAMLINE_LIVE_GATEWAY_URL")
	modelName := os.Getenv("CONTRACTOR_STREAMLINE_LIVE_MODEL")
	if url == "" || modelName == "" {
		t.Skip("live Streamline Gateway settings are not set")
	}
	token := os.Getenv("CONTRACTOR_STREAMLINE_LIVE_GATEWAY_TOKEN")
	if token == "" {
		token = "unused"
	}
	llm, err := streamline.NewOpenAICompatibleModel(streamline.GatewaySettings{
		URL: url, Token: contracts.NewSecretString(token), Model: modelName,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText(
			`Call finish now with {"summary":"live gateway ok","artifacts":{}}. Do not answer with prose.`,
			genai.RoleUser,
		)},
		Config: &genai.GenerateContentConfig{Tools: []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name: "finish", Description: "Finish the smoke test",
				ParametersJsonSchema: map[string]any{
					"type": "object", "additionalProperties": false,
					"properties": map[string]any{
						"summary":   map[string]any{"type": "string"},
						"artifacts": map[string]any{"type": "object"},
					},
					"required": []string{"summary", "artifacts"},
				},
			}},
		}}},
	}
	called := false
	for response, responseErr := range llm.GenerateContent(t.Context(), request, false) {
		if responseErr != nil {
			t.Fatal(responseErr)
		}
		if response != nil && response.Content != nil {
			for _, part := range response.Content.Parts {
				if part != nil && part.FunctionCall != nil && part.FunctionCall.Name == "finish" {
					called = true
				}
			}
		}
	}
	if !called {
		t.Fatal("live model did not call finish")
	}
}
