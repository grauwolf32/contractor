package scan

import (
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"testing"
)

func TestScanPlanSelectsExactJobOutputsAtMaterialLimit(t *testing.T) {
	h := newFactoryHarness(t, 16)
	h.invocation.Stage.ScanPlan.MaxJobs = 16
	h.invocation.Stage.Result.Artifacts["materials"] = resolved.ArtifactSlot{Required: true, MediaTypes: []string{"application/json"}, From: &resolved.ArtifactBinding{Namespace: "scan-results", Name: "finding-materials"}}
	result := h.run(t)
	if result.Outcome != stageresult.StageSucceeded || len(h.invoker.calls) != 16 {
		t.Fatalf("result=%+v calls=%d", result, len(h.invoker.calls))
	}
	ref := result.Artifacts["materials"]
	if ref.Name != "finding-materials" {
		t.Fatal("declared Stage context cannot resolve the output")
	}
	selected, err := contracts.DecodeMaterialSet(h.artifacts.payload(ref).Data)
	if err != nil || len(selected.Materials) != 32 {
		t.Fatalf("selection=%+v %v", selected, err)
	}
	for _, material := range selected.Materials {
		if material.ScopeID != "run-scan" || material.Ref.ValidateExact() != nil {
			t.Fatal("foreign or unpinned job output")
		}
	}
	replay := h.run(t)
	if !replay.Artifacts["materials"].SameExact(ref) || len(h.invoker.calls) != 16 {
		t.Fatal("completion recovery repeated scanner work")
	}
	h = newFactoryHarness(t, 17)
	h.invocation.Stage.ScanPlan.MaxJobs = 17
	h.invocation.Stage.Result.Artifacts["materials"] = resolved.ArtifactSlot{Required: true, MediaTypes: []string{"application/json"}, From: &resolved.ArtifactBinding{Namespace: "scan-results", Name: "finding-materials"}}
	if _, err := h.factory.Create(h.invocation); err == nil {
		t.Fatal("one-over material selection allowed target dispatch")
	}
}

func TestScanPlanRetainsReportAndExactPairsAfterOutputAcknowledgementFailure(t *testing.T) {
	h := newFactoryHarness(t, 1)
	h.invocation.Stage.ScanPlan.MaxJobs = 1
	h.invocation.Stage.Result.Artifacts["materials"] = resolved.ArtifactSlot{Required: true, MediaTypes: []string{"application/json"}, From: &resolved.ArtifactBinding{Namespace: "scan-results", Name: "finding-materials"}}
	h.invoker.failureCode = "tool_report_failed"
	result := h.run(t)
	if result.Outcome != stageresult.StageFailed || h.sessions.state.Jobs[0].Report == nil || len(h.sessions.state.Jobs[0].Outputs) != 2 {
		t.Fatalf("publication failure lost original report/pair: %+v", h.sessions.state.Jobs)
	}
	selected, err := contracts.DecodeMaterialSet(h.artifacts.payload(result.Artifacts["materials"]).Data)
	if err != nil || len(selected.Materials) != 2 || len(h.invoker.calls) != 1 {
		t.Fatalf("recovery selection=%+v error=%v", selected, err)
	}
}
