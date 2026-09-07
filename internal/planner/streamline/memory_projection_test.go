package streamline

import (
	"testing"

	plannermemory "github.com/grauwolf32/contractor/internal/memory"
)

func TestPlannerMemoryUntaggedResultsRemainArrays(t *testing.T) {
	namespace, err := plannermemory.NewNamespace(newPlannerMemoryArtifactStore(), plannermemory.Binding{RunID: "run", StageExecutionID: "stage", Namespace: "builder"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	created, err := namespace.WriteMemory(ctx, "untagged", "body", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := namespace.WriteMemory(ctx, "previously_tagged", "body", "", []string{"old"}); err != nil {
		t.Fatal(err)
	}
	replaced, err := namespace.WriteMemory(ctx, "previously_tagged", "new body", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	appended, err := namespace.AppendMemory(ctx, "previously_tagged", "fragment")
	if err != nil {
		t.Fatal(err)
	}
	read, err := namespace.ReadMemory(ctx, "previously_tagged")
	if err != nil {
		t.Fatal(err)
	}
	for _, note := range []plannermemory.Note{created, replaced, appended, read} {
		response := memoryNoteResult(note)
		tags, ok := response["tags"].([]any)
		if !ok || len(tags) != 0 {
			t.Fatalf("full result tags=%#v", response["tags"])
		}
	}
	previews, err := namespace.ListMemories(ctx)
	if err != nil {
		t.Fatal(err)
	}
	response := memoryListResult(previews)
	for _, raw := range response["result"].([]any) {
		note := raw.(map[string]any)
		tags, ok := note["tags"].([]any)
		if !ok || len(tags) != 0 {
			t.Fatalf("preview tags=%#v", note["tags"])
		}
	}
}
