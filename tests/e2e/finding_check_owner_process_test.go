//go:build e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The fixture supplies trusted Run admission until the public launch caller is
// added. Scheduler, private RPC, Runtime collection and durable receipts are real.
func TestFindingCheckOwnerAcrossProcesses(t *testing.T) {
	stage := domainGatewayStage{name: "finding-check-owner", tools: []string{"read_audit_task", "evidence", "submit_result", "submit_results", "write_text_artifact"}, steps: []domainGatewayStep{
		toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
		toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": "proof", "text": "Observed the selected fixture property", "media_type": "text/plain"})),
		toolGatewayStep("evidence", auditResultArguments(map[string]any{"sources": []any{map[string]any{"kind": "artifact", "namespace": "audit-check", "name": "proof"}}, "interpretation": "A bounded observation", "tested_conditions": []string{"fixture deployment, reader A"}, "limitations": []string{}})),
		toolGatewayStep("submit_result", auditResultArguments(map[string]any{"verdict": "supported", "rationale": "Evidence supports the bounded assertion", "limitations": []string{}})),
		finalGatewayStep("Check recorded", nil),
	}}
	h := startFindingsHarness(t, []domainGatewayStage{ordinaryFindingsProducerStage("seed"), stage})
	project := createProjectResource(t, h.client, h.baseURL)
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "sources", "check-source", "application/zip", auditProgramZip(t, map[string][]byte{"app.py": []byte(findingsSource)}))
	body, _ := json.Marshal(map[string]any{"workflow": "fixture-ordinary-findings@1", "artifacts": map[string]artifactRef{"source": source}})
	seedID := postProjectRun(t, h.client, h.baseURL, project.ProjectID, "check-owner-source", body, false)
	waitForFindingsRun(t, h, seedID, "succeeded")
	receipts := findingsReceipts(t, h, seedID)
	if len(receipts) != 1 || receipts[0].FindingID == nil {
		t.Fatal("source finding missing")
	}
	selection := contracts.FindingCheck{FindingID: *receipts[0].FindingID, Revision: 1, ReceiptID: receipts[0].ReceiptID, Assertion: "Reader A observes the selected fixture property"}
	pool, err := pgxpool.New(h.ctx, h.databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	defer func() {
		if t.Failed() {
			rows, queryErr := pool.Query(h.ctx, `SELECT run_id,state,state_reason_code,stage_termination FROM stage_executions`)
			if queryErr == nil {
				defer rows.Close()
				for rows.Next() {
					values, _ := rows.Values()
					t.Logf("persisted Run failure: %v", values)
				}
			}
		}
	}()
	storage := artifactpgtest.NewStorage()
	runs := runstore.NewPostgresStore(pool, storage)
	seed, err := runs.GetRun(h.ctx, seedID)
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := resolved.DecodeResolvedWorkflowSnapshot(seed.WorkflowSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	workflow.Ref.Name = "finding-check-fixture"
	workflow.AuditTask = nil
	current := workflow.Stages[workflow.EntryStage]
	var name string
	var binding resolved.ResolvedAgentBinding
	for name, binding = range current.Agents {
		break
	}
	binding.Template = binding.Template.Clone()
	binding.Template.Skills = nil
	binding.Template.Summarizer = nil
	binding.ContextManagement = contracts.WorkerContextManagement{Strategy: contracts.ContextStrategyTemplate}
	binding.Template.Ref.TemplateID = "finding-check-fixture"
	instruction := "Check one trusted assigned Finding; retain evidence and submit the scoped result."
	binding.Template.Instructions = contracts.ResolvedInstructions{Ref: "fixture/check.md", Digest: contentdigest.Bytes([]byte(instruction)), Text: instruction}
	binding.Template.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "audit-results", Version: "2"}, Tools: []string{"evidence", "read_audit_task", "submit_result", "submit_results"}}, {Ref: contracts.ToolsetRef{ToolsetID: "text-artifacts", Version: "1"}, Tools: []string{"write_text_artifact"}}}
	binding.Template.Ref.Digest, err = binding.Template.ContentDigest()
	if err != nil {
		t.Fatal(err)
	}
	current.Agents = map[string]resolved.ResolvedAgentBinding{name: binding}
	execution := current.ExecutionConfig.Agents[name]
	execution.SummarizerModelPolicy = nil
	execution.CompactorModelPolicy = nil
	execution.Origins.SummarizerModelPolicy = ""
	execution.Origins.CompactorModelPolicy = ""
	current.ExecutionConfig.Agents[name] = execution
	current.Context = resolved.StageContext{Artifacts: map[string]resolved.ContextArtifact{"context": {Namespace: "inputs", Name: "context", Required: true}}}
	current.Result.Artifacts = map[string]resolved.ArtifactSlot{"result": {Required: true, MediaTypes: []string{"application/json"}, From: &resolved.ArtifactBinding{Namespace: binding.Namespace, Name: "result"}}}
	current.WorkflowOutputs = map[string]string{"result": "result"}
	workflow.Stages = map[string]resolved.ResolvedStage{workflow.EntryStage: current}
	workflow.Inputs = map[string]resolved.ArtifactSlot{"context": {Required: true, MediaTypes: []string{"application/json"}}}
	workflow.Outputs = map[string]resolved.ArtifactSlot{"result": {Required: true, Primary: true, MediaTypes: []string{"application/json"}}}
	raw, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil {
		t.Fatal(err)
	}
	const checkID = "finding-check-owner-run"
	run, err := runstoretest.CreateRun(h.ctx, runs, runstore.CreateRunParams{RunID: checkID, OwnerID: seed.OwnerID, ProjectID: &project.ProjectID, WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: raw, Parameters: map[string]string{}, RuntimeConfig: seed.RuntimeConfig, LaneClass: resolved.EntryLaneClass(workflow)})
	if err != nil {
		t.Fatal(err)
	}
	checkDigest, _ := contentdigest.CanonicalJSON(json.RawMessage(raw))
	contextData, _ := json.Marshal(map[string]any{"schema": "contractor.audit.context.v1", "project_id": project.ProjectID, "finding": selection, "run_id": run.RunID, "workflow_ref": workflow.Ref.Name + "@" + workflow.Ref.Version, "workflow_digest": checkDigest, "items": []any{map[string]any{"item_key": auditcontract.ExecutionItemKey(0), "finding_id": selection.FindingID, "objective": selection.Assertion, "evidence": []any{}}}})
	scope, _ := artifacts.RunScope(checkID)
	artifactService := artifacts.NewService(storage.Repository(pool))
	input, err := artifactService.CreateFrozen(h.ctx, scope, contracts.ArtifactRef{Namespace: "inputs", Name: "context"}, artifacts.Payload{MediaType: "application/json", Data: contextData})
	if err != nil {
		t.Fatal(err)
	}
	if err = runs.SetAuditCompletion(h.ctx, checkID, runstore.AuditCompletionSnapshot{Finding: &selection, Stage: workflow.EntryStage, Agent: name, Contract: contracts.WorkerCompletionContract{Kind: contracts.AuditCheckResultsV1, Context: input.Ref, ResultArtifact: contracts.ArtifactRef{Namespace: binding.Namespace, Name: "result"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err = runs.TransitionRun(h.ctx, checkID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "fixture_admitted"}); err != nil {
		t.Fatal(err)
	}
	waitForFindingsRun(t, h, checkID, "succeeded")
	if len(findingsReceipts(t, h, checkID)) != 0 {
		t.Fatal("validation emitted another finding")
	}
	history, _ := basicAuditRequest(t, h, "GET", fmt.Sprintf("/v1/projects/%s/finding-sources/%s/history?limit=64", project.ProjectID, selection.ReceiptID), nil, "", 0, 200)
	if len(history["items"].([]any)) != 2 {
		t.Fatalf("history missing evidence/result: %v", history)
	}
	if h.gateway.CompletedStages() != 2 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("gateway: %v", h.gateway.Failures())
	}
	deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, checkID)
	after, _ := basicAuditRequest(t, h, "GET", fmt.Sprintf("/v1/projects/%s/finding-sources/%s/history?limit=64", project.ProjectID, selection.ReceiptID), nil, "", 0, 200)
	if len(after["items"].([]any)) != 2 {
		t.Fatal("producer purge lost history")
	}
}
