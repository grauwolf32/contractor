package contracts

import (
	"encoding/json"
	"strings"
	"testing"
)

func testCompaction() WorkerContextManagement {
	return WorkerContextManagement{Strategy: ContextStrategyCompact, WorkerCompaction: &WorkerCompaction{
		ModelPolicy: ResolvedModelPolicy{
			Ref:   ModelPolicyRef{PolicyID: "compactor", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64)},
			Model: "compactor-model", ContextWindowTokens: 131072, MaxOutputTokens: 4096, MaxModelCalls: 1, MaxTotalTokens: 140000,
		},
		Instructions: ResolvedInstructions{
			Ref: "instructions/worker-compaction.md", Digest: "sha256:" + strings.Repeat("b", 64), Text: "Summarize.",
		},
		ContextWindowRatio: 0.75, KeepRecentGroups: 4, MaxCompactions: 8,
	}}
}

func TestCompactBranchRejectsUnsupportedWorkers(t *testing.T) {
	modeled := ResolvedAgentTemplate{Runtime: WorkerRuntimeRef{RuntimeID: "adk", Version: "1"}}
	worker := ResolvedModelPolicy{ContextWindowTokens: 131072, MaxOutputTokens: 4096}
	audit := modeled
	audit.Toolsets = []ToolsetSelection{{Ref: ToolsetRef{ToolsetID: AuditResultsToolsetID, Version: AuditResultsToolsetVersion}}}
	summarizerPolicy := worker
	for _, tc := range []struct {
		name    string
		context AllocationContext
	}{
		{"model-free", AllocationContext{Template: ResolvedAgentTemplate{Runtime: WorkerRuntimeRef{RuntimeID: "tool", Version: "1"}}, SessionMode: WorkerSessionIsolated, WorkerPolicy: worker}},
		{"shared", AllocationContext{Template: modeled, SessionMode: WorkerSessionShared, WorkerPolicy: worker}},
		{"summarizer selection", AllocationContext{Template: modeled, SessionMode: WorkerSessionIsolated, WorkerPolicy: worker, SummarizerModelPolicy: &summarizerPolicy}},
		{"no context window", AllocationContext{Template: modeled, SessionMode: WorkerSessionIsolated, WorkerPolicy: ResolvedModelPolicy{MaxOutputTokens: 4096}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := testCompaction().ValidateAllocation(tc.context); err == nil {
				t.Fatal("unsupported compact@1 allocation accepted")
			}
		})
	}
	if err := testCompaction().ValidateAllocation(AllocationContext{Template: modeled, SessionMode: WorkerSessionIsolated, WorkerPolicy: worker}); err != nil {
		t.Fatalf("isolated modeled Worker: %v", err)
	}
	if err := testCompaction().ValidateAllocation(AllocationContext{Template: audit, SessionMode: WorkerSessionIsolated, WorkerPolicy: worker}); err != nil {
		t.Fatalf("isolated Audit Worker: %v", err)
	}
}

func TestEffectiveContextManagementReplacesOnlyTheBranch(t *testing.T) {
	summarized := ResolvedAgentTemplate{Summarizer: &WorkerSummarizerConfig{}}
	template := WorkerContextManagement{Strategy: ContextStrategyTemplate}
	if got := EffectiveContextManagement(template, summarized); got.Strategy != ContextStrategyTerminal {
		t.Fatalf("summarized template = %+v", got)
	}
	if got := EffectiveContextManagement(template, ResolvedAgentTemplate{}); got.Strategy != ContextStrategyNone {
		t.Fatalf("plain template = %+v", got)
	}
	compact := testCompaction()
	got := EffectiveContextManagement(compact, summarized)
	if got.Strategy != ContextStrategyCompact || got.WorkerCompaction == compact.WorkerCompaction {
		t.Fatalf("compact branch was not cloned: %+v", got)
	}
}

func TestWorkerContextManagementJSONIsFlatAndClosed(t *testing.T) {
	encoded, err := json.Marshal(testCompaction())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(encoded), `{"strategy":"compact@1","modelPolicy":`) {
		t.Fatalf("encoding = %s", encoded)
	}
	var decoded WorkerContextManagement
	if err := json.Unmarshal(encoded, &decoded); err != nil || decoded.ValidateBinding() != nil {
		t.Fatalf("round trip: %v", err)
	}
	for _, raw := range []string{
		`{}`, `null`, `{"strategy":"none","keepRecentGroups":1}`,
		`{"strategy":"compact@1"}`, `{"strategy":"template@1","strategy":"template@1"}`,
	} {
		var value WorkerContextManagement
		if err := json.Unmarshal([]byte(raw), &value); err == nil && value.ValidateBinding() == nil {
			t.Fatalf("%s accepted", raw)
		}
	}
}
