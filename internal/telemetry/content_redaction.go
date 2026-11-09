package telemetry

import (
	"encoding/json"
	"errors"
)

// encodeContent makes a JSON copy before masking, keeping escaping out of
// credential matching. The ordinary content byte bound is applied afterwards.
func (s *otlpPlannerSpan) encodeContent(value any) ([]byte, error) {
	s.owner.mu.Lock()
	redactText := s.owner.redactText
	s.owner.mu.Unlock()
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var decoded any
	if err = json.Unmarshal(raw, &decoded); err != nil {
		return nil, err
	}
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
				maskedKey := redactText(key)
				if _, exists := result[maskedKey]; exists {
					return nil, errors.New("telemetry content has colliding masked keys")
				}
				masked, e := visit(child, depth+1)
				if e != nil {
					return nil, e
				}
				result[maskedKey] = masked
			}
			return result, nil
		}
		return item, nil
	}
	masked, err := visit(decoded, 0)
	if err != nil {
		return nil, err
	}
	return json.Marshal(masked)
}
