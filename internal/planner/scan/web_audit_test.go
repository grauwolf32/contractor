package scan

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

func pinWebRulepack(t *testing.T, h *factoryHarness) {
	t.Helper()
	ref := h.artifacts.put(contracts.ArtifactRef{Namespace: "inputs", Name: "rulepack"}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("id: fixture-rule\n")})
	h.invocation.Context.Artifacts["rulepack"] = &ref

}

func TestWebAuditExecutorRetainsInstalledAndCustomRules(t *testing.T) {
	for _, custom := range []bool{false, true} {
		t.Run(map[bool]string{false: "installed", true: "custom"}[custom], func(t *testing.T) {
			h, _ := newAuditFactoryHarness(t, "web")
			if custom {
				pinWebRulepack(t, h)
			}
			result, err := h.instance(t).Run(t.Context())
			if err != nil || len(h.invoker.calls) != 1 {
				t.Fatalf("execution: %v", err)
			}
			call := h.invoker.calls[0].request
			if call.Parameters["target"] != "https://target.test/api?view=all" || call.Parameters["template_ids"] != "fixture-*" || call.Parameters["tags"] != "http" || call.Parameters["severity"] != "high,critical" || call.Parameters["rate_limit"] != "" {
				t.Fatalf("exact selection lost: %+v", call.Parameters)
			}
			_, pack := call.Artifacts["rulepack"]
			if pack != custom {
				t.Fatal("rule source changed")
			}
			plan, err := scanplan.DecodePlan(h.artifacts.payload(*h.sessions.state.Plan).Data)
			if err != nil || len(plan.Jobs) != 1 {
				t.Fatalf("retained plan: %v", err)
			}
			pkg := auditFactoryResult(t, h, result)
			if !pkg.ExecutionCompleted {
				t.Fatal("completed scanner lost execution status")
			}
		})
	}
}

// A completed session replays its original result before mutable source reads.
// Replacing current context cannot repeat scanner traffic or change that result.
func TestWebAuditCompletedReplayKeepsOriginalRuleResult(t *testing.T) {
	h, _ := newAuditFactoryHarness(t, "web")
	pinWebRulepack(t, h)
	h.invoker.observation = map[string]any{"status": "failed", "exitCode": 1}
	first, err := h.instance(t).Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	// Same Stage session recovery uses the immutable plan and its input bindings.
	ref := h.artifacts.put(contracts.ArtifactRef{Namespace: "inputs", Name: "rulepack"}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("id: another-rule\n")})
	h.invocation.Context.Artifacts["rulepack"] = &ref
	replay, err := h.instance(t).Run(t.Context())
	if err != nil || len(h.invoker.calls) != 1 || !first.Artifacts["report"].SameExact(replay.Artifacts["report"]) {
		t.Fatalf("completed receipt replay changed or retried traffic: error=%v calls=%d", err, len(h.invoker.calls))
	}
}
