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

	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	"google.golang.org/adk/v2/model"
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

// Format redacts the model as a whole for every verb (SEC-R88): fmt cannot call the Gateway token's own redaction through the unexported settings field.
func (openAICompatibleModel) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "streamline.openAICompatibleModel([REDACTED])")
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
		body, err := m.send(ctx, request)
		if err != nil {
			yield(nil, err)
			return
		}
		result, err := decodeChatResponse(body)
		yield(result, err)
	}
}

// send encodes one bounded chat request and returns the Gateway's response
// body. Both paths classify a failed model request with the Gateway's
// failure signatures; Recovery retries a transient one, and a reported
// failure is a *gatewayrecovery.FailureError.
func (m *openAICompatibleModel) send(ctx context.Context, request *model.LLMRequest) ([]byte, error) {
	payload, err := encodeChatRequest(request, m.settings.Model, m.settings.MaxOutputTokens)
	if err != nil || len(payload) > maxGatewayRequestBytes {
		if err == nil {
			err = fmt.Errorf("Planner Gateway request is oversized")
		}
		return nil, err
	}
	httpRequest, err := http.NewRequestWithContext(
		ctx, http.MethodPost, m.settings.URL+"/chat/completions", bytes.NewReader(payload),
	)
	if err != nil {
		return nil, fmt.Errorf("build Planner Gateway request")
	}
	if token := m.settings.Token.Reveal(); token != "" {
		httpRequest.Header.Set("Authorization", "Bearer "+token)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set(requestid.Header, requestid.Ensure(ctx))
	if m.settings.Recovery != nil {
		return m.settings.Recovery.Do(httpRequest, m.settings.HTTPClient, maxGatewayResponseBytes)
	}
	return gatewayrecovery.Send(
		httpRequest, m.settings.HTTPClient, maxGatewayResponseBytes, m.settings.FailureSignatures,
	)
}

// compactorReply is one received compactor response with its explicit
// provider finish reason. The compaction owner, not the adapter, decides
// whether it is a usable summary (PLN-R128).
type compactorReply struct {
	text         string
	toolCalls    int
	finishReason string
	usage        *genai.GenerateContentResponseUsageMetadata
}

// compactionModel is the Planner Gateway path of the tool-free compactor. It
// keeps the raw finish reason that the ordinary main-response decoding drops.
type compactionModel interface {
	compact(context.Context, *model.LLMRequest) (compactorReply, error)
}

func (m *openAICompatibleModel) compact(ctx context.Context, request *model.LLMRequest) (compactorReply, error) {
	body, err := m.send(ctx, request)
	if err != nil {
		return compactorReply{}, err
	}
	return decodeCompactorResponse(body)
}

// decodeCompactorResponse reads one compactor reply. Unparseable JSON or
// usage is the ordinary invalid Gateway response; every other defect is left
// to compactor validation with its known usage.
func decodeCompactorResponse(payload []byte) (compactorReply, error) {
	var response chatResponse
	if err := json.NewDecoder(bytes.NewReader(payload)).Decode(&response); err != nil {
		return compactorReply{}, &gatewayResponseRejected{reason: "invalid JSON"}
	}
	usage, err := chatResponseUsage(response.Usage)
	if err != nil {
		return compactorReply{}, err
	}
	reply := compactorReply{usage: usage}
	if len(response.Choices) > 0 {
		choice := response.Choices[0]
		reply.text = choice.Message.Content
		reply.toolCalls = len(choice.Message.ToolCalls)
		reply.finishReason = choice.FinishReason
	}
	return reply, nil
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
	Usage *chatUsage `json:"usage"`
}

type chatUsage struct {
	PromptTokens     int32 `json:"prompt_tokens"`
	CompletionTokens int32 `json:"completion_tokens"`
	TotalTokens      int32 `json:"total_tokens"`
}

// gatewayResponseRejected identifies a received but unusable HTTP 200 reply.
// Its reason is adapter-owned text and its usage is safe to charge when valid.
type gatewayResponseRejected struct {
	reason string
	usage  *genai.GenerateContentResponseUsageMetadata
}

func (e *gatewayResponseRejected) Error() string {
	return "Planner Gateway returned " + e.reason
}

func decodeChatResponse(payload []byte) (*model.LLMResponse, error) {
	var response chatResponse
	decoder := json.NewDecoder(bytes.NewReader(payload))
	if err := decoder.Decode(&response); err != nil {
		return nil, &gatewayResponseRejected{reason: "invalid JSON"}
	}
	usage, err := chatResponseUsage(response.Usage)
	if err != nil {
		return nil, err
	}
	reject := func(reason string) (*model.LLMResponse, error) {
		return nil, &gatewayResponseRejected{reason: reason, usage: usage}
	}
	if len(response.Choices) == 0 {
		return reject("no choices")
	}
	choice := response.Choices[0]
	if choice.FinishReason == "length" {
		return reject("a length-limited choice")
	}
	parts := make([]*genai.Part, 0, len(choice.Message.ToolCalls)+1)
	if choice.Message.Content != "" {
		parts = append(parts, genai.NewPartFromText(choice.Message.Content))
	}
	for _, call := range choice.Message.ToolCalls {
		var arguments map[string]any
		if strings.TrimSpace(call.Function.Arguments) == "" {
			arguments = map[string]any{}
		} else if err := json.Unmarshal([]byte(call.Function.Arguments), &arguments); err != nil || arguments == nil {
			return reject("invalid tool arguments")
		}
		part := genai.NewPartFromFunctionCall(call.Function.Name, arguments)
		part.FunctionCall.ID = call.ID
		parts = append(parts, part)
	}
	if len(parts) == 0 {
		return reject("an empty choice")
	}
	return &model.LLMResponse{
		Content:       &genai.Content{Role: genai.RoleModel, Parts: parts},
		UsageMetadata: usage, ModelVersion: response.Model,
	}, nil
}

// chatResponseUsage validates the token usage every Planner Gateway response
// must carry (PLN-R86).
func chatResponseUsage(usage *chatUsage) (*genai.GenerateContentResponseUsageMetadata, error) {
	if usage == nil || usage.PromptTokens < 0 || usage.CompletionTokens < 0 || usage.TotalTokens <= 0 ||
		int64(usage.TotalTokens) < int64(usage.PromptTokens)+int64(usage.CompletionTokens) {
		return nil, &gatewayResponseRejected{reason: "invalid token usage"}
	}
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     usage.PromptTokens,
		CandidatesTokenCount: usage.CompletionTokens,
		TotalTokenCount:      usage.TotalTokens,
	}, nil
}

var _ model.LLM = (*openAICompatibleModel)(nil)
var _ compactionModel = (*openAICompatibleModel)(nil)
