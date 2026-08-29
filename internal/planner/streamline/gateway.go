package streamline

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"

	"github.com/grauwolf32/contractor/internal/requestid"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

const (
	maxGatewayRequestBytes  = 4 * 1024 * 1024
	maxGatewayResponseBytes = 4 * 1024 * 1024
)

// NewOpenAICompatibleModel adapts an OpenAI-compatible LLM Gateway (including
// LiteLLM) to the Google ADK model.LLM interface. Provider bodies and the
// bearer token are deliberately absent from returned errors.
func NewOpenAICompatibleModel(settings GatewaySettings) (model.LLM, error) {
	normalized, err := normalizeGatewaySettings(settings)
	if err != nil {
		return nil, err
	}
	return &openAICompatibleModel{settings: normalized}, nil
}

type openAICompatibleModel struct {
	settings GatewaySettings
}

func (m *openAICompatibleModel) Name() string { return m.settings.Model }

func (m *openAICompatibleModel) GenerateContent(
	ctx context.Context, request *model.LLMRequest, stream bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if stream {
			yield(nil, fmt.Errorf("Planner LLM Gateway streaming is disabled"))
			return
		}
		payload, err := encodeChatRequest(request, m.settings.Model, m.settings.MaxOutputTokens)
		if err != nil || len(payload) > maxGatewayRequestBytes {
			if err == nil {
				err = fmt.Errorf("Planner Gateway request is oversized")
			}
			yield(nil, err)
			return
		}
		httpRequest, err := http.NewRequestWithContext(
			ctx, http.MethodPost, m.settings.URL+"/chat/completions", bytes.NewReader(payload),
		)
		if err != nil {
			yield(nil, fmt.Errorf("build Planner Gateway request"))
			return
		}
		httpRequest.Header.Set("Authorization", "Bearer "+m.settings.Token.Reveal())
		httpRequest.Header.Set("Content-Type", "application/json")
		httpRequest.Header.Set(requestid.Header, requestid.Ensure(ctx))
		response, err := m.settings.HTTPClient.Do(httpRequest)
		if err != nil {
			yield(nil, fmt.Errorf("Planner Gateway request failed"))
			return
		}
		defer response.Body.Close()
		body, readErr := io.ReadAll(io.LimitReader(response.Body, maxGatewayResponseBytes+1))
		if readErr != nil || len(body) > maxGatewayResponseBytes {
			yield(nil, fmt.Errorf("Planner Gateway response is unreadable or oversized"))
			return
		}
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			yield(nil, fmt.Errorf("Planner Gateway returned HTTP %d", response.StatusCode))
			return
		}
		result, err := decodeChatResponse(body)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(result, nil)
	}
}

type chatRequest struct {
	Model             string        `json:"model"`
	Messages          []chatMessage `json:"messages"`
	Tools             []chatTool    `json:"tools,omitempty"`
	ToolChoice        any           `json:"tool_choice,omitempty"`
	ParallelToolCalls bool          `json:"parallel_tool_calls"`
	MaxTokens         int           `json:"max_tokens,omitempty"`
	Temperature       *float32      `json:"temperature,omitempty"`
}

type chatMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	Name       string         `json:"name,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
}

type chatTool struct {
	Type     string       `json:"type"`
	Function chatFunction `json:"function"`
}

type chatFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type chatToolCall struct {
	ID       string               `json:"id"`
	Type     string               `json:"type"`
	Function chatToolCallFunction `json:"function"`
}

type chatToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func encodeChatRequest(request *model.LLMRequest, modelName string, maxOutputTokens int) ([]byte, error) {
	if request == nil {
		return nil, fmt.Errorf("Planner model request is required")
	}
	messages, err := convertContents(request)
	if err != nil {
		return nil, err
	}
	result := chatRequest{
		Model: modelName, Messages: messages, ParallelToolCalls: false,
		MaxTokens: maxOutputTokens,
	}
	if request.Config != nil {
		result.Temperature = request.Config.Temperature
		for _, group := range request.Config.Tools {
			if group == nil {
				continue
			}
			for _, declaration := range group.FunctionDeclarations {
				if declaration == nil || strings.TrimSpace(declaration.Name) == "" {
					return nil, fmt.Errorf("Planner model tool declaration is invalid")
				}
				parameters := declaration.ParametersJsonSchema
				if parameters == nil {
					parameters = declaration.Parameters
				}
				result.Tools = append(result.Tools, chatTool{Type: "function", Function: chatFunction{
					Name: declaration.Name, Description: declaration.Description, Parameters: parameters,
				}})
			}
		}
	}
	if len(result.Tools) > 0 {
		result.ToolChoice = "auto"
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return nil, fmt.Errorf("encode Planner Gateway request")
	}
	return encoded, nil
}

func convertContents(request *model.LLMRequest) ([]chatMessage, error) {
	result := make([]chatMessage, 0, len(request.Contents)+1)
	if request.Config != nil && request.Config.SystemInstruction != nil {
		text, err := textParts(request.Config.SystemInstruction)
		if err != nil {
			return nil, err
		}
		if text != "" {
			result = append(result, chatMessage{Role: "system", Content: text})
		}
	}
	pendingCalls := make(map[string][]string)
	nextSyntheticCall := 0
	for _, content := range request.Contents {
		if content == nil {
			continue
		}
		if content.Role == genai.RoleModel {
			message := chatMessage{Role: "assistant"}
			var text []string
			for _, part := range content.Parts {
				switch {
				case part == nil:
				case part.FunctionCall != nil:
					arguments, err := json.Marshal(part.FunctionCall.Args)
					if err != nil {
						return nil, fmt.Errorf("encode Planner tool arguments")
					}
					identifier := part.FunctionCall.ID
					if identifier == "" {
						nextSyntheticCall++
						identifier = "call_" + fmt.Sprint(nextSyntheticCall)
					}
					message.ToolCalls = append(message.ToolCalls, chatToolCall{
						ID: identifier, Type: "function", Function: chatToolCallFunction{
							Name: part.FunctionCall.Name, Arguments: string(arguments),
						},
					})
					pendingCalls[part.FunctionCall.Name] = append(pendingCalls[part.FunctionCall.Name], identifier)
				case part.Text != "":
					text = append(text, part.Text)
				default:
					return nil, fmt.Errorf("Planner Gateway does not support non-text model content")
				}
			}
			message.Content = strings.Join(text, "\n")
			result = append(result, message)
			continue
		}
		var ordinaryText []string
		for _, part := range content.Parts {
			switch {
			case part == nil:
			case part.FunctionResponse != nil:
				encoded, err := json.Marshal(part.FunctionResponse.Response)
				if err != nil {
					return nil, fmt.Errorf("encode Planner tool response")
				}
				identifier := part.FunctionResponse.ID
				if identifier == "" && len(pendingCalls[part.FunctionResponse.Name]) > 0 {
					identifier = pendingCalls[part.FunctionResponse.Name][0]
					pendingCalls[part.FunctionResponse.Name] = pendingCalls[part.FunctionResponse.Name][1:]
				}
				result = append(result, chatMessage{
					Role: "tool", Name: part.FunctionResponse.Name,
					ToolCallID: identifier, Content: string(encoded),
				})
			case part.Text != "":
				ordinaryText = append(ordinaryText, part.Text)
			default:
				return nil, fmt.Errorf("Planner Gateway does not support non-text user content")
			}
		}
		if len(ordinaryText) > 0 {
			result = append(result, chatMessage{Role: "user", Content: strings.Join(ordinaryText, "\n")})
		}
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("Planner Gateway request has no messages")
	}
	return result, nil
}

func textParts(content *genai.Content) (string, error) {
	var values []string
	for _, part := range content.Parts {
		if part == nil {
			continue
		}
		if part.Text == "" || part.FunctionCall != nil || part.FunctionResponse != nil {
			return "", fmt.Errorf("Planner system instruction must contain text only")
		}
		values = append(values, part.Text)
	}
	return strings.Join(values, "\n"), nil
}

type chatResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Content   string         `json:"content"`
			ToolCalls []chatToolCall `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int32 `json:"prompt_tokens"`
		CompletionTokens int32 `json:"completion_tokens"`
		TotalTokens      int32 `json:"total_tokens"`
	} `json:"usage"`
}

func decodeChatResponse(payload []byte) (*model.LLMResponse, error) {
	var response chatResponse
	decoder := json.NewDecoder(bytes.NewReader(payload))
	if err := decoder.Decode(&response); err != nil {
		return nil, fmt.Errorf("Planner Gateway returned invalid JSON")
	}
	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("Planner Gateway returned no choices")
	}
	if response.Usage == nil || response.Usage.PromptTokens < 0 ||
		response.Usage.CompletionTokens < 0 || response.Usage.TotalTokens <= 0 ||
		int64(response.Usage.TotalTokens) <
			int64(response.Usage.PromptTokens)+int64(response.Usage.CompletionTokens) {
		return nil, fmt.Errorf("Planner Gateway returned invalid token usage")
	}
	choice := response.Choices[0]
	parts := make([]*genai.Part, 0, len(choice.Message.ToolCalls)+1)
	if choice.Message.Content != "" {
		parts = append(parts, genai.NewPartFromText(choice.Message.Content))
	}
	for _, call := range choice.Message.ToolCalls {
		var arguments map[string]any
		if strings.TrimSpace(call.Function.Arguments) == "" {
			arguments = map[string]any{}
		} else if err := json.Unmarshal([]byte(call.Function.Arguments), &arguments); err != nil {
			return nil, fmt.Errorf("Planner Gateway returned invalid tool arguments")
		}
		part := genai.NewPartFromFunctionCall(call.Function.Name, arguments)
		part.FunctionCall.ID = call.ID
		parts = append(parts, part)
	}
	if len(parts) == 0 {
		return nil, fmt.Errorf("Planner Gateway returned an empty choice")
	}
	usage := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     response.Usage.PromptTokens,
		CandidatesTokenCount: response.Usage.CompletionTokens,
		TotalTokenCount:      response.Usage.TotalTokens,
	}
	return &model.LLMResponse{
		Content:       &genai.Content{Role: genai.RoleModel, Parts: parts},
		UsageMetadata: usage, ModelVersion: response.Model,
	}, nil
}

var _ model.LLM = (*openAICompatibleModel)(nil)
