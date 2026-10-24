package scan

import (
	"encoding/json"
	"sort"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

func pinWebRulepack(t *testing.T, h *factoryHarness) {
	t.Helper()
	ref := h.artifacts.put(contracts.ArtifactRef{Namespace: "inputs", Name: "rulepack"}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("id: fixture-rule\n")})
	h.invocation.Context.Artifacts["rulepack"] = &ref
	manifestRef := h.invocation.Context.Artifacts["execution_manifest"]
	var manifest auditdomain.ExecutionManifest
	if err := json.Unmarshal(h.artifacts.payload(*manifestRef).Data, &manifest); err != nil {
		t.Fatal(err)
	}
	manifest.Items[0].Inputs = append(manifest.Items[0].Inputs, auditdomain.ExactInput{Name: "rulepack", Ref: ref, Digest: auditdomain.DigestBytes([]byte("id: fixture-rule\n"))})
	sort.Slice(manifest.Items[0].Inputs, func(i, j int) bool { return manifest.Items[0].Inputs[i].Name < manifest.Items[0].Inputs[j].Name })
	data, err := auditdomain.EncodeExecutionManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	updated := h.artifacts.put(contracts.ArtifactRef{Namespace: "inputs", Name: "execution_manifest"}, artifacts.Payload{MediaType: "application/json", Data: data})
	h.invocation.Context.Artifacts["execution_manifest"] = &updated
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
			if call.Parameters["target"] != "https://target.test/api?view=all" || call.Parameters["template_ids"] != "fixture-*" || call.Parameters["tags"] != "http" || call.Parameters["severity"] != "high,critical" || call.Parameters["rate_limit"] != "7" {
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
			if pkg.Results.Results[0].Assessment != "inconclusive" {
				t.Fatal("scanner completion claimed a security verdict")
			}
		})
	}
}

func TestWebAuditRejectsMissingOrChangedCustomRulesBeforeTraffic(t *testing.T) {
	for _, mode := range []string{"missing", "changed"} {
		t.Run(mode, func(t *testing.T) {
			h, _ := newAuditFactoryHarness(t, "web")
			pinWebRulepack(t, h)
			if mode == "missing" {
				h.invocation.Context.Artifacts["rulepack"] = nil
			} else {
				ref := h.artifacts.put(contracts.ArtifactRef{Namespace: "inputs", Name: "rulepack"}, artifacts.Payload{MediaType: "application/yaml", Data: []byte("id: different-rule\n")})
				h.invocation.Context.Artifacts["rulepack"] = &ref
			}
			_, err := h.instance(t).Run(t.Context())
			if err == nil || len(h.invoker.calls) != 0 {
				t.Fatal("a custom-rule mismatch caused scanner traffic or installed-rule fallback")
			}
		})
	}
}
