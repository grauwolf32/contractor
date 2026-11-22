//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestFactExtractionAcrossProcesses(t *testing.T) {
	writer := func() domainGatewayStage {
		return domainGatewayStage{name: "facts/writer", tools: withMemoryTools([]string{"read_artifact", "write_text_artifact"}), steps: []domainGatewayStep{
			toolGatewayStep("write_memory", fixedArguments(map[string]any{"name": "selected", "content": "Source reports nginx. Version may be 1.20. No exhaustive probe was performed."})),
			toolGatewayStep("write_memory", fixedArguments(map[string]any{"name": "ignored", "content": "Working chatter; do not extract."})),
			toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": "report", "text": "const engine = \"nginx\";", "media_type": "text/markdown"})),
			toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": "http", "text": "HTTP/1.1 200 OK\r\nServer: nginx\r\n", "media_type": "text/plain"})),
			finalGatewayStep("Notes written", map[string]domainArtifactBinding{"report": {namespace: "writer", name: "report"}}),
		}}
	}
	tools := []string{"read_artifact", "write_text_artifact", "record_fact"}
	fact := func(title, kind, memoryName string) domainGatewayStep {
		return toolGatewayStep("record_fact", func(request map[string]any) (map[string]any, error) {
			args := map[string]any{"title": title, "description": "Free-form reported observation; version and probe coverage remain uncertain."}
			if kind == "memory" {
				args["memory_name"] = memoryName
				return args, nil
			}
			for _, read := range findingsToolResponses(request, "read_artifact") {
				candidate, _ := read["artifact"].(map[string]any)
				if candidate["name"] == kind {
					args["evidence_refs"] = []any{materialSelector(candidate)}
					return args, nil
				}
			}
			return nil, fmt.Errorf("missing exact %s source", kind)
		})
	}
	finish := func(expected int) []domainGatewayStep {
		return []domainGatewayStep{
			toolGatewayStep("write_text_artifact", func(request map[string]any) (map[string]any, error) {
				accepted := 0
				for _, result := range findingsToolResponses(request, "record_fact") {
					if result["finding_id"] != nil {
						accepted++
					}
				}
				if accepted != expected {
					return nil, fmt.Errorf("accepted %d facts, want %d", accepted, expected)
				}
				return map[string]any{"name": "report", "text": fmt.Sprintf("Accepted %d facts. Rejections and uncertainty retained; zero is valid.", accepted), "media_type": "text/markdown"}, nil
			}),
			finalGatewayStep("Extraction complete", map[string]domainArtifactBinding{"report": {namespace: "facts", name: "report"}}),
		}
	}
	extractor := domainGatewayStage{name: "facts/extract", tools: tools, steps: []domainGatewayStep{
		toolGatewayStep("read_artifact", stageReadArguments("memories", nil)),
		toolGatewayStep("read_artifact", fixedArguments(map[string]any{"namespace": "writer", "name": "report"})),
		toolGatewayStep("read_artifact", fixedArguments(map[string]any{"namespace": "writer", "name": "http"})),
		fact("Absent source", "memory", "absent"),
		fact("nginx reported", "memory", "selected"),
		fact("Different title, same evidence", "memory", "selected"),
		fact("nginx in code", "report", ""), fact("nginx response", "http", ""),
	}}
	extractor.steps = append(extractor.steps, domainGatewayStep{modelFail: true})
	retry := domainGatewayStage{name: "facts/retry", tools: tools, steps: []domainGatewayStep{
		toolGatewayStep("read_artifact", stageReadArguments("memories", nil)),
		fact("New attempt, same memory", "memory", "selected"),
	}}
	retry.steps = append(retry.steps, finish(1)...)
	zero := domainGatewayStage{name: "facts/zero", tools: tools, steps: []domainGatewayStep{toolGatewayStep("read_artifact", stageReadArguments("memories", nil))}}
	zero.steps = append(zero.steps, finish(0)...)
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{writer(), extractor, retry, writer(), zero}, func(target string) {
		for _, entry := range []struct{ from, to string }{
			{"memory-snapshot/instructions/worker.md", "instructions/snapshot-fixture-worker.md"},
			{"memory-snapshot/agent-templates/worker.yaml", "agent-templates/snapshot_fixture_worker.yaml"},
			{"fact-extraction/workflows/extract.yaml", "workflows/fact_extraction_fixture.yaml"},
		} {
			data, err := os.ReadFile(filepath.Join(repoRoot(t), "tests/e2e/testdata", entry.from))
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(filepath.Join(target, entry.to), data, 0600); err != nil {
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
	for i := range 2 {
		body, _ := json.Marshal(map[string]any{"workflow": "fact-extraction-fixture@1", "artifacts": map[string]any{}})
		run := postProjectRun(t, h.client, h.baseURL, project.ProjectID, fmt.Sprintf("extract-%d", i), body, false)
		waitForFindingsRun(t, h, run, "succeeded")
		receipts := findingsReceipts(t, h, run)
		if i == 0 {
			if len(receipts) != 5 {
				t.Fatalf("receipts=%d", len(receipts))
			}
			ids := map[string]string{}
			var selectedReceipt string
			var originalMemory []byte
			for _, receipt := range receipts {
				if receipt.FindingID == nil || receipt.Origin.Audit != nil {
					t.Fatalf("invalid Run-level fact: %+v", receipt)
				}
				ids[receipt.Document.Title] = *receipt.FindingID
				if receipt.Document.Title == "nginx reported" {
					selectedReceipt = receipt.ReceiptID
					ref := receipt.Evidence[0].Ref
					data, _ := download(t, h.client, h.baseURL+"/v1/runs/"+run+"/artifacts/"+ref.Namespace+"/"+ref.Name+"?revision="+url.QueryEscape(*ref.Revision))
					originalMemory = data
					var original struct {
						Notes []struct {
							Name, Content, Description string
							Tags                       []string
						} `json:"notes"`
					}
					if json.Unmarshal(data, &original) != nil || len(original.Notes) != 1 || original.Notes[0].Name != "selected" || original.Notes[0].Content != "Source reports nginx. Version may be 1.20. No exhaustive probe was performed." || original.Notes[0].Tags == nil {
						t.Fatal("original selected memory was not attached exactly")
					}
				}
				if !strings.Contains(receipt.Document.Description, "Free-form reported observation") {
					t.Fatal("the original free-form statement changed")
				}
			}
			if ids["nginx reported"] != ids["Different title, same evidence"] || ids["nginx reported"] != ids["New attempt, same memory"] || ids["nginx reported"] == ids["nginx in code"] {
				t.Fatal("evidence correlation changed")
			}
			selection := publishFindingsCollection(t, h, "extracted-fact-ground", []findingintake.CollectionSelection{{Kind: "project", ID: project.ProjectID, ReceiptIDs: []string{selectedReceipt}, Findings: []findingintake.CollectionFindingSelection{}}}, false)
			ref := selection.Artifact.Ref
			body, _ := download(t, h.client, h.baseURL+"/v1/artifacts/"+ref.Namespace+"/"+ref.Name+"?revision="+url.QueryEscape(*ref.Revision))
			collection, pkg, err := auditdomain.DecodeFindingCollectionPackage(body)
			if err != nil || len(collection.Entries) != 1 || collection.Entries[0].ReceiptID != selectedReceipt || len(collection.Entries[0].Evidence) != 1 {
				t.Fatal("exact ground selection expanded to other source receipts", err)
			}
			material, ok := pkg.MemberByID(collection.Entries[0].Evidence[0].DocumentID)
			if !ok || !bytes.Equal(material.Data(), originalMemory) {
				t.Fatal("selected ground lost the original memory")
			}
		} else if len(receipts) != 0 {
			t.Fatal("zero extraction wrote facts")
		}
		executions, err := store.ListStageExecutions(h.ctx, run)
		if err != nil {
			t.Fatal(err)
		}
		expectedExecutions := 3
		if i == 0 {
			expectedExecutions++
		}
		if len(executions) != expectedExecutions {
			t.Fatalf("stages=%d", len(executions))
		}
		for _, execution := range executions {
			allocations, err := store.ListStageAllocations(h.ctx, execution.StageExecutionID)
			if err != nil {
				t.Fatal(err)
			}
			if execution.StageName == "snapshot" {
				if execution.PlannerSessionID != nil || execution.PlannerInvocationID != nil {
					t.Fatal("snapshot created a Planner session")
				}
				if len(allocations) != 0 {
					t.Fatal("snapshot allocated Worker")
				}
				continue
			}
			if len(allocations) != 1 {
				t.Fatalf("%s allocations=%d", execution.StageName, len(allocations))
			}
			if execution.StageName == "extract" {
				stage, err := resolved.DecodeResolvedStageSnapshot(execution.StageSpecSnapshot)
				if err != nil {
					t.Fatal(err)
				}
				if stage.Planner.PlannerID != "passthrough" || stage.ExecutionConfig.Planner != nil {
					t.Fatal("extractor Planner changed")
				}
				allocation := allocations[0]
				if allocation.AgentTemplateRef.TemplateID != "fact_extractor" || allocation.RuntimeConfiguration == nil || allocation.RuntimeConfiguration.ModelPolicy != stage.ExecutionConfig.Agents["extractor"].ModelPolicy.Ref {
					t.Fatalf("wrong Worker/model: %+v", allocation)
				}
			}
		}
	}
	if h.gateway.CompletedStages() != 5 || len(h.gateway.Failures()) != 0 {
		t.Fatal(h.gateway.Failures())
	}
}
