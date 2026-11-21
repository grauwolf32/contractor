//go:build e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestMemorySnapshotAcrossProcesses(t *testing.T) {
	tools := withMemoryTools([]string{"read_artifact", "write_text_artifact"})
	writer := func(label string) domainGatewayStage {
		return domainGatewayStage{name: label, tools: tools, steps: []domainGatewayStep{
			toolGatewayStep("write_memory", fixedArguments(map[string]any{"name": "selected", "content": "Selected source reports nginx"})),
			toolGatewayStep("write_memory", fixedArguments(map[string]any{"name": "ignored", "content": "Ignored private marker"})),
			toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": "report", "text": "Notes written", "media_type": "text/markdown"})),
			finalGatewayStep("Notes written", map[string]domainArtifactBinding{"report": {namespace: "writer", name: "report"}}),
		}}
	}
	reader := domainGatewayStage{name: "snapshot/reader", tools: tools, steps: []domainGatewayStep{
		toolGatewayStep("read_artifact", fixedArguments(map[string]any{"namespace": "writer", "name": "memory.selected"})),
		toolGatewayStep("read_artifact", stageRefArguments("snapshot", nil)),
		toolGatewayStep("write_text_artifact", func(request map[string]any) (map[string]any, error) {
			reads := findingsToolResponses(request, "read_artifact")
			if len(reads) != 2 {
				return nil, fmt.Errorf("expected hidden-ref rejection and ordinary snapshot")
			}
			rejected, _ := json.Marshal(reads[0])
			if !strings.Contains(string(rejected), "artifact_access_denied") {
				return nil, fmt.Errorf("Memory binding was exposed: %s", rejected)
			}
			data, err := artifactTextBytes(reads[1]["text"])
			if err != nil || !strings.Contains(string(data), "Selected source reports nginx") || strings.Contains(string(data), "Ignored private marker") || strings.Contains(string(data), "revision") || strings.Contains(string(data), "ordinal") {
				return nil, fmt.Errorf("unexpected snapshot contents")
			}
			return map[string]any{"name": "report", "text": string(data), "media_type": "text/markdown"}, nil
		}),
		finalGatewayStep("Selected note read", map[string]domainArtifactBinding{"report": {namespace: "reader", name: "report"}}),
	}}
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{writer("snapshot/writer"), reader, writer("without-snapshot/writer")}, func(target string) {
		source := filepath.Join(repoRoot(t), "tests/e2e/testdata/memory-snapshot")
		for _, entry := range []struct{ from, to string }{
			{"instructions/worker.md", "instructions/snapshot-fixture-worker.md"},
			{"agent-templates/worker.yaml", "agent-templates/snapshot_fixture_worker.yaml"},
			{"workflows/snapshot.yaml", "workflows/snapshot_fixture.yaml"},
			{"workflows/without.yaml", "workflows/without_snapshot_fixture.yaml"},
		} {
			data, err := os.ReadFile(filepath.Join(source, entry.from))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(target, entry.to), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
	})
	project := createProjectResource(t, h.client, h.baseURL)
	pool, err := pgxpool.New(h.ctx, h.databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	for _, scenario := range []struct {
		workflow string
		stages   int
	}{{"snapshot-fixture@1", 3}, {"without-snapshot-fixture@1", 1}} {
		body, _ := json.Marshal(map[string]any{"workflow": scenario.workflow, "artifacts": map[string]any{}})
		run := postProjectRun(t, h.client, h.baseURL, project.ProjectID, strings.TrimSuffix(scenario.workflow, "@1"), body, false)
		status := waitForFindingsRun(t, h, run, "succeeded")
		if len(status.Attempts) != scenario.stages {
			t.Fatalf("attempts=%d", len(status.Attempts))
		}
		stages, err := store.ListStageExecutions(h.ctx, run)
		if err != nil {
			t.Fatal(err)
		}
		for _, stage := range stages {
			if stage.StageName != "snapshot" {
				continue
			}
			if stage.PlannerSessionID != nil || stage.PlannerInvocationID != nil {
				t.Fatal("snapshot created Planner")
			}
			allocations, err := store.ListStageAllocations(h.ctx, stage.StageExecutionID)
			if err != nil || len(allocations) != 0 {
				t.Fatalf("snapshot allocations=%d error=%v", len(allocations), err)
			}
		}
		for _, attempt := range status.Attempts {
			if attempt.Stage == "snapshot" && (attempt.PlannerStartedAt != nil || len(attempt.Resources) != 0) {
				t.Fatal("snapshot reported model resources")
			}
		}
		if scenario.stages == 1 {
			request, _ := http.NewRequest(http.MethodGet, h.baseURL+"/v1/runs/"+run+"/artifacts/snapshot/selected-notes", nil)
			request.Header.Set("Authorization", "Bearer "+publicToken)
			response := do(t, h.client, request, http.StatusNotFound)
			response.Body.Close()
		} else {
			data, _ := download(t, h.client, h.baseURL+"/v1/runs/"+run+"/outputs/report")
			if !strings.Contains(string(data), "nginx") {
				t.Fatal("snapshot was not read")
			}
		}
	}
	// Each ordinary Worker adds its existing finalizer call; the operation adds none.
	if h.gateway.CompletedStages() != 3 || h.gateway.Calls() != 15 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("unexpected model work: stages=%d calls=%d errors=%v", h.gateway.CompletedStages(), h.gateway.Calls(), h.gateway.Failures())
	}
}
