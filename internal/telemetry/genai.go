package telemetry

import (
	"bytes"
	"encoding/json"
	"strings"
)

// Content follows OpenTelemetry GenAI semantic conventions v1.41.1. Collector
// implementations own any conversion into their native UI representation.
func genAIContent(field, raw string, model bool) map[string]string {
	if !model {
		return map[string]string{"contractor." + field: raw}
	}
	var value any
	if json.Unmarshal([]byte(raw), &value) != nil || value == nil {
		return nil
	}
	values := map[string]any{}
	if field == "output" {
		message := genAIMessage(value, "assistant")
		message["finish_reason"] = "unknown"
		values["gen_ai.output.messages"] = []any{message}
	} else if request, ok := genAIRequest(value); ok {
		messages := []any{}
		contents, _ := request["contents"].([]any)
		if request["contents"] != nil && contents == nil {
			return nil
		}
		for _, item := range contents {
			messages = append(messages, genAIMessage(item, "user"))
		}
		values["gen_ai.input.messages"] = messages
		if system := request["systemInstruction"]; system != nil {
			values["gen_ai.system_instructions"] = genAIParts(system)
		}
		tools := []any{}
		definitions, _ := request["tools"].([]any)
		for _, definition := range definitions {
			tool, _ := definition.(map[string]any)
			declarations, _ := tool["functionDeclarations"].([]any)
			for _, entry := range declarations {
				declaration, ok := entry.(map[string]any)
				if !ok {
					continue
				}
				item := map[string]any{"type": "function", "name": declaration["name"]}
				if description := declaration["description"]; description != nil {
					item["description"] = description
				}
				parameters := declaration["parametersJsonSchema"]
				if parameters == nil {
					parameters = declaration["parameters"]
				}
				if parameters != nil {
					item["parameters"] = parameters
				}
				tools = append(tools, item)
			}
		}
		if len(tools) > 0 {
			values["gen_ai.tool.definitions"] = tools
		}
	} else {
		values["gen_ai.input.messages"] = []any{genAIMessage(value, "user")}
	}
	result := map[string]string{}
	for key, item := range values {
		result[key] = boundedGenAI(key, item)
	}
	return result
}

func genAIRequest(value any) (map[string]any, bool) {
	request, ok := value.(map[string]any)
	if !ok {
		return nil, false
	}
	_, found := request["contents"]
	return request, found
}

func genAIText(value any) string {
	if text, ok := value.(string); ok {
		return text
	}
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if encoder.Encode(value) != nil {
		return ""
	}
	return strings.TrimSuffix(buffer.String(), "\n")
}

func genAIParts(value any) []any {
	content, _ := value.(map[string]any)
	if content["truncated"] == true {
		return []any{map[string]any{"type": "text", "content": content["preview"], "truncated": true}}
	}
	parts, ok := content["parts"].([]any)
	if !ok {
		return []any{map[string]any{"type": "text", "content": genAIText(value)}}
	}
	result := []any{}
	for _, entry := range parts {
		part, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		if text, ok := part["text"].(string); ok {
			kind := "text"
			if part["thought"] == true {
				kind = "reasoning"
			}
			result = append(result, map[string]any{"type": kind, "content": text})
		} else if call, ok := part["functionCall"].(map[string]any); ok {
			item := map[string]any{"type": "tool_call", "name": call["name"], "arguments": call["args"]}
			if id, ok := call["id"].(string); ok {
				item["id"] = id
			}
			result = append(result, item)
		} else if response, ok := part["functionResponse"].(map[string]any); ok {
			item := map[string]any{"type": "tool_call_response", "response": response["response"]}
			if id, ok := response["id"].(string); ok {
				item["id"] = id
			}
			result = append(result, item)
		} else {
			result = append(result, map[string]any{"type": "provider_specific", "value": part})
		}
	}
	return result
}

func genAIMessage(value any, defaultRole string) map[string]any {
	role := defaultRole
	if content, ok := value.(map[string]any); ok {
		if selected, ok := content["role"].(string); ok {
			role = selected
		}
	}
	if role == "model" {
		role = "assistant"
	}
	parts := genAIParts(value)
	tool := len(parts) > 0
	for _, part := range parts {
		if part.(map[string]any)["type"] != "tool_call_response" {
			tool = false
		}
	}
	if tool {
		role = "tool"
	}
	return map[string]any{"role": role, "parts": parts}
}

func boundedGenAI(key string, value any) string {
	encoded := genAIText(value)
	if len(encoded) <= MaxContentBytes {
		return encoded
	}
	preview := strings.ToValidUTF8(encoded[:MaxContentBytes], "")
	for {
		part := map[string]any{"type": "text", "content": preview, "truncated": true}
		var bounded any
		switch key {
		case "gen_ai.system_instructions":
			bounded = []any{part}
		case "gen_ai.tool.definitions":
			bounded = []any{map[string]any{"type": "truncated", "name": "truncated", "preview": preview}}
		default:
			message := map[string]any{"role": "user", "parts": []any{part}}
			if key == "gen_ai.output.messages" {
				message["role"] = "assistant"
				message["finish_reason"] = "unknown"
			}
			bounded = []any{message}
		}
		encoded = genAIText(bounded)
		if len(encoded) <= MaxContentBytes {
			return encoded
		}
		retained := max(0, len(preview)-(len(encoded)-MaxContentBytes+16))
		preview = strings.ToValidUTF8(preview[:retained], "")
	}
}
