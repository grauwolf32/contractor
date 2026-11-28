package controlplane

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedWorkerProgressProjectsValidatedStateAndRechecksAuthority(t *testing.T) {
	fixture := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "valid", "agent-state-compaction-applied.json")
	registry := &InMemoryRegistry{allocations: map[string]storedReservation{}}
	calls := 0
	loseDuringRead := false
	server := newRuntimeFixture(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if !strings.HasSuffix(r.URL.Path, "/allocations/allocation-1/agent-state") {
			t.Errorf("unexpected State path %s", r.URL.Path)
		}
		if loseDuringRead {
			registry.mu.Lock()
			delete(registry.allocations, "allocation-1")
			registry.mu.Unlock()
		}
		w.Header().Set("Cache-Control", "private, no-cache")
		w.Header().Set("ETag", `"contractor-agent-state-v1-7"`)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(fixture)
	}))
	defer server.Close()
	registry.allocations["allocation-1"] = storedReservation{reservation: Reservation{ControlURL: server.URL, Grant: AllocationGrant{AllocationID: "allocation-1", RunID: "run", StageExecutionID: "stage", LogicalAgentName: "worker", RuntimeInstanceID: "instance", RuntimeAgentID: fixturePrincipal(server.URL)}}}
	client, _ := NewRuntimeControlClient(server.Client())
	reader := NewWorkerProgressReader(registry, client)
	for _, run := range []string{"foreign", "run"} {
		progress, phases, err := reader.ReadWorkerProgress(t.Context(), run, "stage", "allocation-1", "worker")
		if err != nil {
			t.Fatal(err)
		}
		if run == "foreign" {
			if calls != 0 || progress != nil {
				t.Fatal("foreign selector caused I/O")
			}
			continue
		}
		if progress == nil || progress.Phase != "applied" || progress.AfterPrompt.PromptTokens != nil || phases == nil {
			t.Fatalf("measured facts lost: %+v", progress)
		}
	}
	loseDuringRead = true
	progress, _, err := reader.ReadWorkerProgress(context.Background(), "run", "stage", "allocation-1", "worker")
	if err != nil || progress != nil {
		t.Fatalf("released observation survived: %+v %v", progress, err)
	}
}
