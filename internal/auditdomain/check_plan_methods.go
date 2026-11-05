package auditdomain

import "github.com/grauwolf32/contractor/internal/contracts"

const (
	FindingCheckPlanSchema      = "contractor.audit.finding-check-plan.v1"
	MaximumCheckCapabilities    = 32
	MaximumCheckCapabilityTools = 128
)

// CheckPlanKey is a registered semantic key, independent of Workflow role names.
type CheckPlanKey struct {
	Schema string `json:"schema"`
	Method string `json:"method"`
	Phase  string `json:"phase"`
}

// LookupCheckPlanMethod declares the only methods the finding source adapter
// can turn into typed plans. Ordinary untyped findings may use other methods.
func LookupCheckPlanMethod(schema, method string) (CheckPlanKey, bool) {
	if schema != FindingCheckPlanSchema {
		return CheckPlanKey{}, false
	}
	phase := ""
	switch method {
	case "verify", "static-trace", "http-probe":
		phase = "verify"
	case "replay", "applicability", "alternative":
		phase = method
	default:
		return CheckPlanKey{}, false
	}
	return CheckPlanKey{Schema: schema, Method: method, Phase: phase}, true
}

// CheckCapabilityRequirement names tools of one exact Worker in a route's
// pinned Workflow. Tools of another Worker or another item cannot satisfy it.
type CheckCapabilityRequirement struct {
	Stage string               `json:"stage"`
	Agent string               `json:"agent"`
	Ref   contracts.ToolsetRef `json:"ref"`
	Tools []string             `json:"tools"`
}
