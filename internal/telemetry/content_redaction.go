package telemetry

import (
	"encoding/json"
	"errors"
)

// copyPlannerContent normalizes SDK values before interpreting their schema.
func copyPlannerContent(value any) (any, error) {
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var decoded any
	if err = json.Unmarshal(raw, &decoded); err != nil {
		return nil, err
	}
	return transformContentStrings(decoded, func(text string) string { return text })
}

// transformContentStrings keeps JSON keys intact. Protocol fields are excluded
// by the owning GenAI projection; only its data payloads reach this function.
func transformContentStrings(value any, redactText func(string) string) (any, error) {
	remaining := 100000
	var visit func(any, int) (any, error)
	visit = func(item any, depth int) (any, error) {
		remaining--
		if depth > 64 || remaining < 0 {
			return nil, errors.New("telemetry content exceeds traversal bound")
		}
		switch v := item.(type) {
		case string:
			return redactText(v), nil
		case []any:
			for i, child := range v {
				masked, e := visit(child, depth+1)
				if e != nil {
					return nil, e
				}
				v[i] = masked
			}
		case map[string]any:
			result := make(map[string]any, len(v))
			for key, child := range v {
				masked, e := visit(child, depth+1)
				if e != nil {
					return nil, e
				}
				result[key] = masked
			}
			return result, nil
		}
		return item, nil
	}
	return visit(value, 0)
}
