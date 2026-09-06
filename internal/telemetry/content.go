package telemetry

import (
	"encoding/json"
	"strings"
)

const MaxContentBytes = 256 * 1024

type contentSpan interface {
	capturesContent() bool
	setContent(string, string)
}

// CapturePlannerInput/Output evaluate their payload only for an opted-in sink.
// Content is deliberately unredacted. It never enters normal logs or metrics.
func CapturePlannerInput(span PlannerSpan, value func() any) {
	capturePlannerContent(span, "langfuse.observation.input", value)
}
func CapturePlannerOutput(span PlannerSpan, value func() any) {
	capturePlannerContent(span, "langfuse.observation.output", value)
}

func capturePlannerContent(span PlannerSpan, field string, value func() any) {
	c, ok := span.(contentSpan)
	if !ok || !c.capturesContent() || value == nil {
		return
	}
	// Optional serialization must not change execution outcomes.
	defer func() { _ = recover() }()
	raw, err := json.Marshal(value())
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
