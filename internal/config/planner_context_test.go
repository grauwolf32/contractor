package config

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
)

const compactingSelector = "compacting-review@1"

const compactBlock = `      plannerContextManagement:
        strategy: compact@1
        modelPolicy: test-compactor@1
        instructions:
          ref: instructions/planner-compaction.md
`

func compactionFixture(t *testing.T) string {
	t.Helper()
	return copyCoreFixture(t, "planner-compaction")
}

func compactingWorkflowPath(root string) string {
	return filepath.Join(root, "workflows/compacting_review.yaml")
}

func TestPlannerContextManagementResolvesExplicitCompactBranch(t *testing.T) {
	snapshot := mustLoad(t, compactionFixture(t))
	workflow, err := snapshot.Workflow(compactingSelector)
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["review"]
	selected := stage.PlannerContextManagement
	if selected == nil || selected.Strategy != resolved.PlannerContextCompact ||
		selected.ContextWindowRatio != 0.75 || selected.KeepRecentGroups != 4 || selected.MaxCompactions != 8 ||
		selected.Instructions.Ref != "instructions/planner-compaction.md" || selected.Instructions.Text == "" {
		t.Fatalf("resolved compaction = %+v", selected)
	}
	assertDigest(t, selected.Instructions.Digest)
	planner := stage.ExecutionConfig.Planner
	if planner == nil || planner.CompactorModelPolicy == nil ||
		planner.CompactorModelPolicy.Ref.PolicyID != "test-compactor" ||
		planner.Origins.CompactorModelPolicy != "workflow.stages.review.plannerContextManagement" ||
		planner.ModelPolicy.Ref.PolicyID != "test-window-planner" {
		t.Fatalf("Planner selection = %+v", planner)
	}
	// The resolved snapshot round-trips strictly with every default explicit.
	encoded, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	for _, member := range []string{`"contextWindowRatio":0.75`, `"keepRecentGroups":4`, `"maxCompactions":8`, `"compactorModelPolicy"`} {
		if !strings.Contains(string(encoded), member) {
			t.Fatalf("snapshot lacks %s", member)
		}
	}
	var decoded resolved.ResolvedStage
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if *decoded.PlannerContextManagement != *selected {
		t.Fatal("snapshot did not round-trip the compact@1 branch")
	}
}

func TestPlannerContextManagementNoneKeepsSnapshotUnchanged(t *testing.T) {
	for _, test := range []struct {
		name  string
		block string
	}{
		{"omitted", ""},
		{"explicit", "      plannerContextManagement:\n        strategy: none\n"},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := compactionFixture(t)
			replaceFile(t, compactingWorkflowPath(root), compactBlock, test.block)
			workflow, err := mustLoad(t, root).Workflow(compactingSelector)
			if err != nil {
				t.Fatal(err)
			}
			stage := workflow.Stages["review"]
			if stage.PlannerContextManagement != nil || stage.ExecutionConfig.Planner.CompactorModelPolicy != nil ||
				stage.ExecutionConfig.Planner.Origins.CompactorModelPolicy != "" {
				t.Fatalf("none resolved to %+v", stage.PlannerContextManagement)
			}
			encoded, err := json.Marshal(stage)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(encoded), "plannerContextManagement") || strings.Contains(string(encoded), "compactor") {
				t.Fatal("none added members to the Stage snapshot")
			}
		})
	}
}

func TestPlannerContextManagementRejectsInvalidAuthoring(t *testing.T) {
	for _, test := range []struct {
		name    string
		old     string
		new     string
		planner string
	}{
		{name: "null", old: compactBlock, new: "      plannerContextManagement: null\n"},
		{name: "none with empty field", old: compactBlock, new: "      plannerContextManagement: {strategy: none, modelPolicy: \"\"}\n"},
		{name: "null through merge", old: compactBlock, new: "      plannerContextManagement:\n        <<: {strategy: compact@1, modelPolicy: test-compactor@1, instructions: {ref: instructions/planner-compaction.md}, contextWindowRatio: null}\n"},
		{name: "null-member", old: "        strategy: compact@1\n", new: "        strategy: compact@1\n        contextWindowRatio: null\n"},
		{name: "unknown", old: "        strategy: compact@1\n", new: "        strategy: compact@1\n        modelURL: http://model\n"},
		{name: "duplicate", old: "        strategy: compact@1\n", new: "        strategy: compact@1\n        strategy: none\n"},
		{name: "none-mixed", old: "        strategy: compact@1\n", new: "        strategy: none\n"},
		{name: "terminal", old: "        strategy: compact@1\n", new: "        strategy: terminal@1\n"},
		{name: "summarizer", old: "        strategy: compact@1\n", new: "        strategy: compact@1\n        summarizerModelPolicy: test-compactor@1\n"},
		{name: "missing-policy", old: "        modelPolicy: test-compactor@1\n", new: ""},
		{name: "missing-instructions", old: "        instructions:\n          ref: instructions/planner-compaction.md\n", new: ""},
		{name: "unknown-policy", old: "modelPolicy: test-compactor@1", new: "modelPolicy: absent-compactor@1"},
		{name: "tool-policy", old: "modelPolicy: test-compactor@1", new: "modelPolicy: test-worker@1"},
		{name: "ratio-zero", old: "        strategy: compact@1\n", new: "        strategy: compact@1\n        contextWindowRatio: 0\n"},
		{name: "ratio-one", old: "        strategy: compact@1\n", new: "        strategy: compact@1\n        contextWindowRatio: 1\n"},
		{name: "recent-17", old: "        strategy: compact@1\n", new: "        strategy: compact@1\n        keepRecentGroups: 17\n"},
		{name: "cycles-0", old: "        strategy: compact@1\n", new: "        strategy: compact@1\n        maxCompactions: 0\n"},
		{name: "cycles-17", old: "        strategy: compact@1\n", new: "        strategy: compact@1\n        maxCompactions: 17\n"},
		{name: "passthrough", old: "      planner: streamline@1\n", new: "      planner: passthrough@1\n", planner: "remove"},
		{name: "missing-window", old: "modelPolicy: test-window-planner@1", new: "modelPolicy: test-planner@1"},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := compactionFixture(t)
			replaceFile(t, compactingWorkflowPath(root), test.old, test.new)
			if test.planner == "remove" {
				replaceFile(t, compactingWorkflowPath(root),
					"    planner:\n      modelPolicy: test-window-planner@1\n      llmGateway: local-litellm@1\n", "")
			}
			_, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
			if err == nil {
				t.Fatal("invalid plannerContextManagement loaded")
			}
			if !strings.Contains(err.Error(), "plannerContextManagement") && !strings.Contains(err.Error(), "compactor") {
				t.Fatalf("rejection is not attributable to plannerContextManagement: %v", err)
			}
		})
	}
}

func TestPlannerContextManagementAcceptsFrozenBounds(t *testing.T) {
	root := compactionFixture(t)
	replaceFile(t, compactingWorkflowPath(root), "        strategy: compact@1\n",
		"        strategy: compact@1\n        contextWindowRatio: 0.5\n        keepRecentGroups: 16\n        maxCompactions: 16\n")
	workflow, err := mustLoad(t, root).Workflow(compactingSelector)
	if err != nil {
		t.Fatal(err)
	}
	selected := workflow.Stages["review"].PlannerContextManagement
	if selected.ContextWindowRatio != 0.5 || selected.KeepRecentGroups != 16 || selected.MaxCompactions != 16 {
		t.Fatalf("bounds = %+v", selected)
	}
}

func TestCompactorModelPolicySelectionPrecedence(t *testing.T) {
	root := compactionFixture(t)
	writeFile(t, filepath.Join(root, "model-policies/test-compactor-2.yaml"), []byte(strings.ReplaceAll(
		string(readFile(t, filepath.Join(root, "model-policies/test-compactor.yaml"))), "name: test-compactor", "name: test-compactor-two",
	)))
	replaceFile(t, compactingWorkflowPath(root), "      modelPolicy: test-window-planner@1\n",
		"      modelPolicy: test-window-planner@1\n      compactorModelPolicy: test-compactor-two@1\n")
	snapshot := mustLoad(t, root)
	workflow, err := snapshot.Workflow(compactingSelector)
	if err != nil {
		t.Fatal(err)
	}
	planner := workflow.Stages["review"].ExecutionConfig.Planner
	if planner.CompactorModelPolicy.Ref.PolicyID != "test-compactor-two" || planner.Origins.CompactorModelPolicy != "workflow.executionConfig.planner" {
		t.Fatalf("Workflow compactor = %s from %s", planner.CompactorModelPolicy.Ref.PolicyID, planner.Origins.CompactorModelPolicy)
	}
	run, err := snapshot.ResolveRunWorkflow(t.Context(), compactingSelector,
		decodeExecutionConfigPatch(t, `{"planner":{"compactorModelPolicy":"test-compactor@1"}}`), developmentCredentialLookup(t, snapshot))
	if err != nil {
		t.Fatal(err)
	}
	planner = run.Stages["review"].ExecutionConfig.Planner
	if planner.CompactorModelPolicy.Ref.PolicyID != "test-compactor" || !strings.HasPrefix(planner.Origins.CompactorModelPolicy, "run.") {
		t.Fatalf("Run compactor = %s from %s", planner.CompactorModelPolicy.Ref.PolicyID, planner.Origins.CompactorModelPolicy)
	}
	for _, patch := range []string{
		`{"workers":{"compactorModelPolicy":"test-compactor@1"}}`,
		`{"planner":{"compactorModelPolicy":"test-worker@1"}}`,
		`{"stages":{"review":{"agents":{"reviewer":{"compactorModelPolicy":"test-compactor@1"}}}}}`,
	} {
		if _, err := snapshot.ResolveRunWorkflow(t.Context(), compactingSelector,
			decodeExecutionConfigPatch(t, patch), developmentCredentialLookup(t, snapshot)); err == nil {
			t.Fatalf("inapplicable compactor selection %s accepted", patch)
		}
	}
}

func TestCompactorSelectionRejectedWithoutCompactingStage(t *testing.T) {
	root := compactionFixture(t)
	replaceFile(t, compactingWorkflowPath(root), compactBlock, "")
	replaceFile(t, compactingWorkflowPath(root), "      modelPolicy: test-window-planner@1\n",
		"      modelPolicy: test-window-planner@1\n      compactorModelPolicy: test-compactor@1\n")
	if _, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed")); err == nil {
		t.Fatal("ignored compactor selection on a none Workflow loaded")
	}
	root = compactionFixture(t)
	replaceFile(t, compactingWorkflowPath(root), compactBlock, "")
	snapshot := mustLoad(t, root)
	if _, err := snapshot.ResolveRunWorkflow(t.Context(), compactingSelector,
		decodeExecutionConfigPatch(t, `{"planner":{"compactorModelPolicy":"test-compactor@1"}}`), developmentCredentialLookup(t, snapshot)); err == nil {
		t.Fatal("Run compactor selection on a none Stage accepted")
	}
}

func TestPlannerContextManagementSnapshotRejectsNoneAndPartialBranches(t *testing.T) {
	for _, value := range []string{
		`null`,
		`{"strategy":"none"}`,
		`{"strategy":"compact@1"}`,
		`{"strategy":"compact@1","instructions":{"ref":"i.md","digest":"sha256:` + strings.Repeat("a", 64) + `","text":"t"},"contextWindowRatio":0.75,"keepRecentGroups":4,"maxCompactions":8,"extra":1}`,
		`{"strategy":"compact@1","instructions":{"ref":"i.md","digest":"sha256:` + strings.Repeat("a", 64) + `","text":"t"},"contextWindowRatio":1,"keepRecentGroups":4,"maxCompactions":8}`,
	} {
		var decoded resolved.PlannerContextManagement
		if err := json.Unmarshal([]byte(value), &decoded); err == nil {
			t.Fatalf("snapshot branch %s accepted", value)
		}
	}
}

func TestPinnedWorkflowRejectsChangedPlannerContextManagement(t *testing.T) {
	root := compactionFixture(t)
	snapshot := mustLoad(t, root)
	pinned, err := snapshot.Workflow(compactingSelector)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := snapshot.ResolvePinnedRunWorkflow(t.Context(), pinned, developmentCredentialLookup(t, snapshot)); err != nil {
		t.Fatal(err)
	}
	changed := pinned.Clone()
	stage := changed.Stages["review"]
	stage.PlannerContextManagement.MaxCompactions = 2
	changed.Stages["review"] = stage
	if _, err := snapshot.ResolvePinnedRunWorkflow(t.Context(), changed, developmentCredentialLookup(t, snapshot)); err == nil {
		t.Fatal("pinned Workflow with changed compaction thresholds accepted")
	}
}

func TestInlineEscalationReplacesCompactor(t *testing.T) {
	root := compactionFixture(t)
	writeFile(t, filepath.Join(root, "model-policies/test-compactor-2.yaml"), []byte(strings.ReplaceAll(
		string(readFile(t, filepath.Join(root, "model-policies/test-compactor.yaml"))), "name: test-compactor", "name: test-compactor-two",
	)))
	replaceFile(t, compactingWorkflowPath(root), "        failed:\n          fail: {}\n",
		"        failed:\n          escalate:\n            maxAttempts: 1\n            executionConfig:\n              planner: {compactorModelPolicy: test-compactor-two@1}\n            then:\n              fail: {}\n")
	snapshot := mustLoad(t, root)
	workflow, err := snapshot.Workflow(compactingSelector)
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["review"]
	escalated := stage.On.Failed.Escalate.ExecutionConfig.Effective.Planner
	if escalated.CompactorModelPolicy.Ref.PolicyID != "test-compactor-two" ||
		escalated.Origins.CompactorModelPolicy != "workflow.stages.review.on.failed.escalate.executionConfig.planner" ||
		stage.ExecutionConfig.Planner.CompactorModelPolicy.Ref.PolicyID != "test-compactor" {
		t.Fatalf("escalated compactor = %s from %s", escalated.CompactorModelPolicy.Ref.PolicyID, escalated.Origins.CompactorModelPolicy)
	}

	// The same override on a Stage without compaction is an ignored selection.
	replaceFile(t, compactingWorkflowPath(root), compactBlock, "")
	if _, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed")); err == nil {
		t.Fatal("compactor override applied to a none Stage loaded")
	}
}

func TestPlannerContextYAMLAliasesAndMergesResolveLikeInlineConfiguration(t *testing.T) {
	for _, merge := range []bool{false, true} {
		t.Run(map[bool]string{false: "alias", true: "merge"}[merge], func(t *testing.T) {
			root := compactionFixture(t)
			path := compactingWorkflowPath(root)
			source := string(readFile(t, path))
			prefix, body, ok := strings.Cut(source, "    review:\n")
			if !ok {
				t.Fatal("fixture has no review Stage")
			}
			first := strings.Replace(body, "      plannerContextManagement:\n", "      plannerContextManagement: &planner_compaction\n", 1)
			first = strings.Replace(first, "          succeed: {}", "          next: again", 1)
			first = strings.Replace(first, "      workflowOutputs:\n        report: report\n", "", 1)
			selection := "      plannerContextManagement: *planner_compaction\n"
			inlineBlock := compactBlock
			if merge {
				selection = "      plannerContextManagement:\n        <<: *planner_compaction\n        keepRecentGroups: 2\n"
				inlineBlock += "        keepRecentGroups: 2\n"
			}
			second := strings.Replace(body, compactBlock, selection, 1)
			writeFile(t, path, []byte(prefix+"    review:\n"+first+"\n    again:\n"+second))
			aliased, err := mustLoad(t, root).Workflow(compactingSelector)
			if err != nil {
				t.Fatal(err)
			}
			second = strings.Replace(body, compactBlock, inlineBlock, 1)
			writeFile(t, path, []byte(prefix+"    review:\n"+first+"\n    again:\n"+second))
			inline, err := mustLoad(t, root).Workflow(compactingSelector)
			if err != nil {
				t.Fatal(err)
			}
			left, err := resolved.EncodeResolvedWorkflowSnapshot(aliased)
			if err != nil {
				t.Fatal(err)
			}
			right, err := resolved.EncodeResolvedWorkflowSnapshot(inline)
			if err != nil || string(left) != string(right) {
				t.Fatalf("YAML reference changed a resolved policy or exact pin: %v", err)
			}
			if aliased.Stages["review"].PlannerContextManagement == aliased.Stages["again"].PlannerContextManagement {
				t.Fatal("independent Stage context policies share mutable state")
			}
		})
	}
}
