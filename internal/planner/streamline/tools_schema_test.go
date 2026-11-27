package streamline

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	plannermemory "github.com/grauwolf32/contractor/internal/memory"
)

func TestRuntimeToolDeclarationsKeepExactWorkerScopes(t *testing.T) {
	for _, count := range []int{1, 2, 3, 64} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			names := make([]string, count)
			for i := range names {
				names[i] = fmt.Sprintf("worker_%02d", i)
			}
			invocation := testInvocation(names...)
			invocation.Stage.Planner = resolved.PlannerRef{PlannerID: "router", Version: "1"}
			binding := invocation.Stage.Agents[names[0]]
			binding.Template.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "memory-tools", Version: "1"}, Tools: []string{"read_memory"}}}
			invocation.Stage.Agents[names[0]] = binding
			factory, err := NewRuntimeFactoryWithMemory(newFakeSessions(), &fakeWorkerInvoker{}, &fakeInspector{}, unavailableWorkerStateReader{}, newPlannerMemoryArtifactStore(), Limits{})
			if err != nil {
				t.Fatal(err)
			}
			current, err := factory.CreateRouter(invocation)
			if err != nil {
				t.Fatal(err)
			}
			p := current.(*streamlinePlanner)
			defer p.stateViews.Close()
			tools, _, err := p.buildTools()
			if err != nil {
				t.Fatal(err)
			}
			seen := map[string]bool{}
			for _, tool := range tools {
				seen[tool.Name] = true
				var shape struct {
					Properties map[string]struct{ Enum []string }
					Required   []string
				}
				if err := json.Unmarshal(tool.Parameters, &shape); err != nil {
					t.Fatal(err)
				}
				switch tool.Name {
				case executeCurrentSubtaskToolName, getWorkerToolUsageToolName:
					if got := shape.Properties["worker_name"].Enum; len(got) != count {
						t.Fatalf("%s: enum %v", tool.Name, got)
					}
				case "read_memory":
					if got := shape.Properties["worker_name"].Enum; len(got) != 1 || got[0] != names[0] {
						t.Fatal("Memory granted to an ineligible Worker", got)
					}
				}
			}
			for _, name := range plannermemory.OperationNames() {
				if name != "read_memory" && seen[name] {
					t.Fatal("unselected Memory operation exposed", name)
				}
			}
			if seen[readContextHistoryToolName] {
				t.Fatal("history exposed for none")
			}
		})
	}
}
