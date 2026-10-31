//go:build e2e

package e2e

import (
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
)

func TestCodeAnalysisE2EConfigurationLoads(t *testing.T) {
	root := stageE2EConfiguration(t, filepath.Join(repoRoot(t), "configs"), filepath.Join(t.TempDir(), "configs"), "http://127.0.0.1:1/v1")
	writeCodeAnalysisE2EConfiguration(t, root)
	snapshot, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatalf("generated Code Analysis process configuration must load: %v", err)
	}
	production, err := snapshot.ModelPolicy("worker@2")
	if err != nil {
		t.Fatal(err)
	}
	for _, fixture := range []struct {
		name     string
		scenario codeAnalysisScenario
	}{
		{"shallow", codeAnalysisShallowScenario("configuration")},
		{"graph", codeAnalysisGraphScenario("configuration", true)},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			workflow, err := snapshot.Workflow("code-analysis-" + fixture.name + "-e2e@1")
			if err != nil {
				t.Fatal(err)
			}
			template := workflow.Stages["analyze"].Agents["analyst"].Template
			policy := workflow.Stages["analyze"].ExecutionConfig.Agents["analyst"].ModelPolicy
			if !reflect.DeepEqual(policy, production) {
				t.Fatal("generated analyst must use the current production Worker policy")
			}
			var tools []string
			for _, selection := range template.Toolsets {
				tools = append(tools, selection.Tools...)
			}
			slices.Sort(tools)
			if !slices.Equal(tools, fixture.scenario.tools) {
				t.Fatalf("generated tool surface = %v, want %v", tools, fixture.scenario.tools)
			}
			// Each scripted candidate also requires one mandatory result finalizer.
			calls, toolCalls := len(fixture.scenario.steps)+1, 0
			for _, step := range fixture.scenario.steps {
				if !step.final {
					toolCalls++
				}
			}
			if policy.MaxModelCalls < calls || policy.MaxToolCalls < toolCalls {
				t.Fatalf("policy cannot execute the unchanged script: need %d model and %d tool calls", calls, toolCalls)
			}
			t.Logf("resolved %s@%s model=%s total_model_calls_including_finalizer=%d script_tool_calls=%d", policy.Ref.PolicyID, policy.Ref.Version, policy.Model, calls, toolCalls)
		})
	}
}
