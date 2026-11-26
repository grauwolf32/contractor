package telemetry

const MaxContentBytes = 256 * 1024

type contentSpan interface {
	captureContent(string, any)
}

// PlannerSpanCapturesContent lets a remote executor avoid assembling content
// when the selected instrumentation cannot consume it.
func PlannerSpanCapturesContent(span PlannerSpan) bool {
	_, ok := span.(contentSpan)
	return ok
}

// CapturePlannerInput/Output evaluate content only for a configured exporter.
// SDK projection precedes masking; byte bounds apply to sanitized content.
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
	c.captureContent(field, value())
}
