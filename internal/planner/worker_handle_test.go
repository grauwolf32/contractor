package planner

import (
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestCloneWorkerHandleDetachesAgentCard(t *testing.T) {
	original := contracts.WorkerHandle{AgentCard: map[string]any{
		"name":                "worker",
		"supportedInterfaces": []any{map[string]any{"url": "https://original.example/a2a"}},
	}}
	cloned := CloneWorkerHandle(original)
	cloned.AgentCard["name"] = "changed"
	cloned.AgentCard["supportedInterfaces"].([]any)[0].(map[string]any)["url"] = "https://changed.example/a2a"
	if original.AgentCard["name"] != "worker" {
		t.Error("changing the cloned card changed the original name")
	}
	if original.AgentCard["supportedInterfaces"].([]any)[0].(map[string]any)["url"] != "https://original.example/a2a" {
		t.Error("changing a nested cloned value changed the original endpoint")
	}
	delete(cloned.AgentCard, "name")
	if _, ok := original.AgentCard["name"]; !ok {
		t.Error("deleting a cloned key deleted the original key")
	}
}

func TestCloneWorkerHandleConcurrentReads(t *testing.T) {
	original := contracts.WorkerHandle{AgentCard: map[string]any{"name": "worker"}}
	var group sync.WaitGroup
	start := make(chan struct{})
	for range 2 {
		group.Add(1)
		go func() {
			defer group.Done()
			<-start
			for range 20 {
				if copy := CloneWorkerHandle(original); copy.AgentCard["name"] != "worker" {
					t.Error("cloned card differs")
				}
			}
		}()
	}
	close(start)
	group.Wait()
}

func TestCloneWorkerHandleRejectsNonJSONCardWithoutSharingIt(t *testing.T) {
	original := contracts.WorkerHandle{
		AllocationID: "allocation-1",
		AgentCard:    map[string]any{"unsupported": make(chan struct{})},
	}
	cloned := CloneWorkerHandle(original)
	if cloned.AgentCard != nil || cloned.AllocationID != original.AllocationID {
		t.Fatalf("invalid card must be omitted while retaining handle identity: %+v", cloned)
	}
	if _, present := original.AgentCard["unsupported"]; !present {
		t.Error("invalid original card was mutated")
	}
}
