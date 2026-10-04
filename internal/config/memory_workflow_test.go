package config

import (
	"reflect"
	"testing"
)

// sharedMemoryFixtureRoot is a frozen catalog with one Streamline and one
// Router Workflow whose Worker bindings select explicit Memory namespaces.
const sharedMemoryFixtureRoot = "testdata/shared-memory"

func TestWorkflowMemoryNamespaceBindingsResolve(t *testing.T) {
	t.Parallel()
	snapshot := mustLoad(t, sharedMemoryFixtureRoot, MVPDescriptors())

	streamline, err := snapshot.Workflow("shared-memory-streamline@1")
	if err != nil {
		t.Fatalf("resolve shared-memory Streamline Workflow: %v", err)
	}
	coordinate := streamline.Stages["coordinate"]
	confirm := streamline.Stages["confirm"]
	if streamline.EntryStage != "coordinate" ||
		coordinate.Planner != (PlannerRef{PlannerID: "streamline", Version: "1"}) ||
		coordinate.Agents["builder"].Namespace != "shared" ||
		confirm.Agents["builder"].Namespace != "shared" ||
		confirm.On.Succeeded.Kind != TransitionSucceed {
		t.Fatalf("unexpected shared-memory Streamline graph: %+v", streamline)
	}
	assertNext(t, coordinate.On.Succeeded, "confirm")
	assertBoundedRetry(t, coordinate.On.Failed, 2)
	// The fixture lists the Memory operations out of order; resolution sorts them.
	wantWorkerTools := []string{"append_memory", "list_memories", "read_memory", "write_memory"}
	if got := coordinate.Agents["builder"].Template.Toolsets[0].Tools; !reflect.DeepEqual(got, wantWorkerTools) {
		t.Fatalf("shared-memory Worker tools = %v, want %v", got, wantWorkerTools)
	}
	if coordinate.Agents["builder"].Template.Ref != confirm.Agents["builder"].Template.Ref {
		t.Fatal("later Stage did not pin the same immutable AgentTemplate")
	}

	router, err := snapshot.Workflow("shared-memory-router@1")
	if err != nil {
		t.Fatalf("resolve shared-memory Router Workflow: %v", err)
	}
	route := router.Stages[router.EntryStage]
	if route.Planner != (PlannerRef{PlannerID: "router", Version: "1"}) ||
		route.Agents["builder"].Namespace != "builder_notes" ||
		route.Agents["reviewer"].Namespace != "reviewer_notes" {
		t.Fatalf("unexpected shared-memory Router bindings: %+v", route)
	}
	if got, want := route.Agents["reviewer"].Template.Toolsets[0].Tools,
		[]string{"read_memory", "write_memory"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("reviewer Memory tools = %v, want %v", got, want)
	}
	if reflect.DeepEqual(
		route.Agents["builder"].Template.Toolsets[0].Tools,
		route.Agents["reviewer"].Template.Toolsets[0].Tools,
	) {
		t.Fatal("Router fixture does not exercise distinct per-operation allowlists")
	}
}
