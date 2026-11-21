//go:build e2e

package e2e

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalservice"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestEvalFindingProjectNavigationAcrossProcesses(t *testing.T) {
	stage := func(name string) domainGatewayStage {
		return domainGatewayStage{name: name, tools: []string{"read_artifact", "record_fact", "read_audit_task", "submit_check_result", "submit_check_results"}, steps: []domainGatewayStep{
			toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
			toolGatewayStep("read_artifact", stageRefArguments("memories", nil)),
			toolGatewayStep("record_fact", func(request map[string]any) (map[string]any, error) {
				reads := findingsToolResponses(request, "read_artifact")
				tasks := findingsToolResponses(request, "read_audit_task")
				if len(reads) != 1 || len(tasks) != 1 {
					return nil, fmt.Errorf("missing selected source/task")
				}
				members, ok := tasks[0]["tasks"].([]any)
				if !ok || len(members) != 1 {
					return nil, fmt.Errorf("missing assigned member")
				}
				member := members[0].(map[string]any)
				return map[string]any{"title": "Reported nginx", "description": "Reported observation with selected exact material", "evidence_refs": []any{reads[0]["artifact"]}, "item_key": member["item_key"]}, nil
			}),
			toolGatewayStep("submit_check_result", auditResultArguments(map[string]any{"rationale": "Selected the reported observation", "tested_conditions": []string{"Read exact material"}, "limitations": []string{"No independent probe"}, "evidence": []string{}})),
			finalGatewayStep("Recorded exploration", nil),
		}}
	}
	root := filepath.Join(repoRoot(t), "tests", "e2e", "testdata", "eval-finding-navigation")
	runFindingEvalNavigation(t, stage, root, []byte(`{"reported":"nginx"}`), filepath.Join(root, "experiment.json"))
}

func runFindingEvalNavigation(t *testing.T, stage func(string) domainGatewayStage, root string, sourceBytes []byte, experimentPath string) {
	t.Helper()
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{stage("eval/a"), stage("eval/b")}, func(target string) {
		for _, directory := range []string{"agent-templates", "workflows", "audit-profiles", "instructions"} {
			entries, err := os.ReadDir(filepath.Join(root, directory))
			if err != nil {
				t.Fatal(err)
			}
			for _, entry := range entries {
				data, err := os.ReadFile(filepath.Join(root, directory, entry.Name()))
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(target, directory, entry.Name()), data, 0600); err != nil {
					t.Fatal(err)
				}
			}
		}
	})
	h.gateway.forbidden = append(h.gateway.forbidden, "PRIVATE_MEMORY_QUALITY_SENTINEL")
	project, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects", map[string]any{"kind": "evaluation", "name": "Memory extraction evaluation"}, "facts-evaluation-project", 0, http.StatusCreated)
	projectID := project["projectId"].(string)
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, projectID, "fact-eval", "technology", "application/json", sourceBytes)
	dataset := map[string]any{
		"datasetId": "project-facts-v1", "name": "Memory extraction mechanism",
		"cases":         []any{map[string]any{"id": "technology", "task": map[string]any{"kind": "memory-extraction", "objective": "Select useful reported observations", "parameters": map[string]string{}}, "inputs": map[string]any{"memories": map[string]any{"scope": "project", "scopeId": projectID, "namespace": source.Namespace, "name": source.Name, "revision": *source.Revision, "sha256": evaldomain.Digest(sourceBytes), "mediaType": "application/json", "sizeBytes": len(sourceBytes)}}, "requires": []string{}, "outputs": map[string]any{}}},
		"privateChecks": []any{map[string]any{"id": "memory-quality", "revision": "v1", "rubric": "Review original memories without asserting independent verification", "expected": map[string]string{"private": "PRIVATE_MEMORY_QUALITY_SENTINEL"}}},
	}
	imported, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+projectID+"/eval-datasets", dataset, "facts-dataset", 0, http.StatusCreated)
	raw, err := os.ReadFile(experimentPath)
	if err != nil {
		t.Fatal(err)
	}
	var input evaldomain.CreateExperiment
	if err := json.Unmarshal(raw, &input); err != nil {
		t.Fatal(err)
	}
	input.Draft.Dataset.Revision = imported["revision"].(string)
	input.Draft.CaseIDs = []string{"technology"}
	input.Draft.Repetitions = 1
	input.Draft.Budgets.MaxMembers = 2
	created, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+projectID+"/eval-experiments", input, "facts-eval", 0, http.StatusCreated)
	id := created["experimentId"].(string)
	path := "/v1/eval-experiments/" + id
	current, _ := basicAuditRequest(t, h, http.MethodGet, path, nil, "", 0, http.StatusOK)
	basicAuditRequest(t, h, http.MethodPost, path+"/commands", map[string]any{"kind": "prepare"}, "facts-prepare", uint64(current["revision"].(float64)), http.StatusAccepted)
	ready := waitFactEvalState(t, h, id, "ready")
	basicAuditRequest(t, h, http.MethodPost, path+"/commands", map[string]any{"kind": "start", "planSha256": ready["planSha256"]}, "facts-start", uint64(ready["revision"].(float64)), http.StatusAccepted)
	waitFactEvalState(t, h, id, "finished")
	var members evalservice.MemberPage
	auditProgramGET(t, h.client, h.baseURL+path+"/members?limit=10", &members)
	if len(members.Items) != 2 || members.HasMore {
		t.Fatalf("Eval members=%d, more=%t", len(members.Items), members.HasMore)
	}
	pool, err := pgxpool.New(h.ctx, h.databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	var owner string
	if err := pool.QueryRow(h.ctx, `SELECT owner_id FROM eval_experiments WHERE experiment_id=$1`, id).Scan(&owner); err != nil {
		t.Fatal(err)
	}
	store := evalstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	projects := map[string]bool{}
	for _, member := range members.Items {
		submission, err := store.Submission(h.ctx, owner, id, member.Member.ID)
		if err != nil {
			t.Fatal(err)
		}
		if submission.ExecutionProjectID == nil || *submission.ExecutionProjectID == projectID || projects[*submission.ExecutionProjectID] {
			t.Fatal("Audit Eval members did not get isolated ordinary Projects")
		}
		projects[*submission.ExecutionProjectID] = true
		record, err := store.LatestNativeRecord(h.ctx, owner, id, member.Member.ID, evaldomain.RecordKindResult)
		if err != nil || record == nil {
			t.Fatal("missing collected native result", err)
		}
		assessment, err := store.LatestNativeRecord(h.ctx, owner, id, member.Member.ID, evaldomain.RecordKindAssessment)
		if err != nil || assessment == nil {
			t.Fatal("missing native review requirement", err)
		}
		var quality evaldomain.AssessmentInput
		if json.Unmarshal(assessment.Document.Bytes(), &quality) != nil || len(quality.Checks) != 1 || quality.Checks[0].Status != "incomplete" {
			t.Fatal("scripted execution claimed semantic quality")
		}
		var result evaldomain.ResultInput
		if err := json.Unmarshal(record.Document.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if result.Collection.Status != "complete" || result.Execution.Ref == nil || result.Execution.Ref.Kind != "audit" || result.Execution.Ref.ID != *submission.ExecutionID {
			t.Fatalf("incomplete native Audit execution: %+v", result.Collection)
		}
		audit := basicAuditGet(t, h, *submission.ExecutionID)
		if audit["projectId"] != *submission.ExecutionProjectID {
			t.Fatal("Eval execution navigation selected a different Project")
		}
		facts, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/projects/"+*submission.ExecutionProjectID+"/findings", nil, "", 0, http.StatusOK)
		rows, ok := facts["items"].([]any)
		if !ok || len(rows) != 1 || !strings.Contains(fmt.Sprint(rows[0]), "Reported nginx") {
			t.Fatal("record_fact is unavailable through the Eval member's actual Audit/Project sources")
		}
	}
	ownerWide, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/findings", nil, "", 0, http.StatusOK)
	if rows, ok := ownerWide["items"].([]any); !ok || len(rows) != 0 {
		t.Fatal("owner-wide Finding discovery exposed Eval workspaces")
	}
	if len(h.gateway.Failures()) != 0 || h.gateway.CompletedStages() != 2 {
		t.Fatalf("Eval Gateway: %v", h.gateway.Failures())
	}
}

func waitFactEvalState(t *testing.T, h *findingsHarness, id, wanted string) map[string]any {
	t.Helper()
	ctx, cancel := context.WithTimeout(h.ctx, 3*time.Minute)
	defer cancel()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		value, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/eval-experiments/"+id, nil, "", 0, http.StatusOK)
		if value["state"] == wanted {
			return value
		}
		if value["state"] == "cancelled" || (wanted == "ready" && value["state"] == "draft" && value["diagnostics"] != nil && len(value["diagnostics"].([]any)) > 0) {
			t.Fatalf("Eval preparation failed: %v", value["diagnostics"])
		}
		select {
		case <-ctx.Done():
			t.Fatalf("Eval did not reach %s; last state=%v diagnostics=%v", wanted, value["state"], value["diagnostics"])
		case <-ticker.C:
		}
	}
}
