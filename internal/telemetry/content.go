package telemetry

import (
	"encoding/json"
	"strings"
)

const MaxContentBytes = 256 * 1024

type contentSpan interface {
	encodeContent(any) ([]byte, error)
	setContent(string, string)
}

// CapturePlannerInput/Output evaluate content only for a configured exporter.
// Known credentials are replaced before truncation and enqueueing.
func CapturePlannerInput(span PlannerSpan, value func() any) {
	capturePlannerContent(span, "input", value)
}
func CapturePlannerOutput(span PlannerSpan, value func() any) {
	capturePlannerContent(span, "output", value)
}

func capturePlannerContent(span PlannerSpan, field string, value func() any) {
	c, ok := span.(contentSpan)
	if !ok || value == nil {
		return
	}
	// Optional serialization must not change execution outcomes.
	defer func() { _ = recover() }()
	raw, err := c.encodeContent(value())
	if err != nil {
		return
	}
	if len(raw) > MaxContentBytes {
		preview := strings.ToValidUTF8(string(raw[:MaxContentBytes]), "")
		for {
			raw, err = json.Marshal(map[string]any{"truncated": true, "preview": preview})
			if err != nil {
				return
			}
			if len(raw) <= MaxContentBytes {
				break
			}
			preview = strings.ToValidUTF8(preview[:len(preview)/2], "")
		}
	}
	c.setContent(field, string(raw))
}
