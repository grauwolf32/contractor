package auditservice

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
)

func TestSharedPostgresFindingCheckEvidenceJudgmentAndPurge(t *testing.T) {
	f := newJudgmentFixture(t, 1, true)
	f.service.findings, _ = findingintake.New(f.pool, artifactpgtest.NewStorage())
	ctx := t.Context()
	storage := artifactpgtest.NewStorage()
	runs := runstore.NewPostgresStore(f.pool, storage)
	findingID := seedAuditFinding(t, ctx, f.pool, f.audit.ProjectID, "owner", f.audit.AuditID, "check-source")
	finding, err := f.service.GetProjectFinding(ctx, "owner", f.audit.ProjectID, findingID)
	if err != nil {
		t.Fatal(err)
	}
	selection := contracts.FindingCheck{FindingID: findingID, ReceiptID: "receipt-check-source", Revision: finding.Revision, Assertion: "Reader can read another account's document on the selected target"}
	workflow := f.pinned.Workflows["inspect"]
	raw, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil {
		t.Fatal(err)
	}
	const runID = "ordinary-finding-check"
	run, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{RunID: runID, OwnerID: "owner", ProjectID: &f.audit.ProjectID, WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: raw, Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(), LaneClass: resolved.EntryLaneClass(workflow)})
	if err != nil {
		t.Fatal(err)
	}
	art := artifacts.NewService(storage.Repository(f.pool))
	local, _ := art.Run(runID)
	checkDigest, _ := contentdigest.CanonicalJSON(json.RawMessage(raw))
	contextData, _ := json.Marshal(map[string]any{"schema": "contractor.audit.context.v1", "project_id": f.audit.ProjectID, "finding": selection, "run_id": runID, "workflow_ref": workflow.Ref.Name + "@" + workflow.Ref.Version, "workflow_digest": checkDigest, "items": []any{map[string]any{"item_key": auditcontract.ExecutionItemKey(0), "finding_id": findingID, "objective": selection.Assertion, "evidence": []any{}}}})
	scope, _ := artifacts.RunScope(runID)
	input, err := art.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: "inputs", Name: "context"}, artifacts.Payload{MediaType: "application/json", Data: contextData})
	if err != nil {
		t.Fatal(err)
	}
	spec := workflow.Stages[workflow.EntryStage]
	agent := spec.Agents["builder"]
	pin := runstore.AuditCompletionSnapshot{Finding: &selection, Stage: workflow.EntryStage, Agent: "builder", Contract: contracts.WorkerCompletionContract{Kind: contracts.AuditCheckResultsV1, Context: input.Ref, ResultArtifact: contracts.ArtifactRef{Namespace: agent.Namespace, Name: "report"}}}
	foreign := selection
	foreign.ReceiptID = "another-source"
	badPin := pin
	badPin.Finding = &foreign
	if err = runs.SetAuditCompletion(ctx, runID, badPin); !errors.Is(err, runstore.ErrInvalid) {
		t.Fatalf("pin/context mismatch admitted: %v", err)
	}
	if err = runs.SetAuditCompletion(ctx, runID, pin); err != nil {
		t.Fatal(err)
	}
	for _, states := range [][2]runstore.WorkflowRunState{{runstore.RunInitializing, runstore.RunPending}, {runstore.RunPending, runstore.RunRunning}} {
		if _, err = runs.TransitionRun(ctx, runID, states[0], states[1], runstore.Reason{Code: "fixture"}); err != nil {
			t.Fatal(err)
		}
	}
	stageRaw, _ := json.Marshal(spec)
	stage, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{LaneClass: resolved.LaneClassModel, StageExecutionID: "ordinary-check-stage", RunID: runID, StageName: workflow.EntryStage, Attempt: 1, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageRaw, StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{}})
	if err != nil {
		t.Fatal(err)
	}
	grant := controlplane.AllocationGrant{AllocationID: "ordinary-check-allocation", RunID: runID, StageExecutionID: stage.StageExecutionID, LogicalAgentName: "builder", Namespace: agent.Namespace, ReadPolicy: controlplane.ReadCurrentRun, WritePolicy: controlplane.WriteInputsAndIntermediates}
	observation, err := local.Write(ctx, contracts.ArtifactRef{Namespace: agent.Namespace, Name: "observation"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("Actual controlled response for reader A includes document B")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	evidenceInput := auditcontract.EvidenceSubmission{APIVersion: contracts.APIVersion, InvocationID: "invocation", SubmissionID: "observation", ItemKey: auditcontract.ExecutionItemKey(0), Sources: []contracts.ArtifactRef{observation.Ref}, Interpretation: "Observed the selected object access", TestedConditions: []string{"reader A, fixture deployment"}, Limitations: []string{}}
	evidence, err := f.service.RecordEvidence(ctx, grant, evidenceInput)
	if err != nil {
		t.Fatal(err)
	}
	if evidence.Material.ScopeKind != "project" || evidence.Material.ScopeID != f.audit.ProjectID || len(evidence.Sources) != 1 {
		t.Fatalf("wrong owner: %+v", evidence)
	}
	initialHistory, err := f.service.findings.ListProjectSourceHistory(ctx, "owner", f.audit.ProjectID, selection.ReceiptID, "", 64, nil)
	if err != nil || len(initialHistory.Items) != 1 {
		t.Fatalf("initial acknowledgement: %+v %v", initialHistory, err)
	}
	// Reuse the existing history-scale fixture. New acceptance must read only
	// this Run's receipts, independently of the source's lifetime fanout.
	prefix := strings.Split(initialHistory.Items[0].Ref.Name, "check-")[0]
	if _, err = f.pool.Exec(ctx, projectFindingScaleHistorySQL, f.audit.ProjectID, initialHistory.Items[0].Ref.Name, prefix); err != nil {
		t.Fatal(err)
	}
	bad := evidenceInput
	bad.ItemKey = auditcontract.ExecutionItemKey(1)
	bad.SubmissionID = "foreign-member"
	if _, err = f.service.RecordEvidence(ctx, grant, bad); err == nil {
		t.Fatal("unassigned member admitted")
	}
	verdict := "supported"
	judgment := judgmentInput(evidenceInput.ItemKey, "Control and probe support the scoped assertion", "check", 0)
	judgment.Items[0].Conclusion.Verdict = &verdict
	judgment.Items[0].Conclusion.TestedConditions = evidenceInput.TestedConditions
	judgment.Items[0].Conclusion.Evidence = evidence.Sources
	accepted, err := f.service.RecordJudgments(ctx, grant, judgment)
	if err != nil {
		t.Fatal(err)
	}
	if len(accepted.Generations) != 1 || accepted.Generations[0].FindingID != findingID || accepted.Generations[0].TaskID != "" {
		t.Fatalf("invented Task: %+v", accepted)
	}
	sameKey := judgment
	sameKey.Items = append([]auditcontract.JudgmentUpdate{}, judgment.Items...)
	sameKey.Items[0].Conclusion.Rationale = "Changed bytes under an existing receipt"
	if _, err = f.service.RecordJudgments(ctx, grant, sameKey); !errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("changed receipt bytes: %v", err)
	}
	changed := judgment
	changed.SubmissionID = "changed"
	changed.Items = append([]auditcontract.JudgmentUpdate{}, judgment.Items...)
	changed.Items[0].Conclusion.Rationale = "A different conclusion on stale base"
	if _, err = f.service.RecordJudgments(ctx, grant, changed); !errors.As(err, new(*JudgmentStaleError)) {
		t.Fatalf("stale correction: %v", err)
	}
	changed.Items[0].BaseGeneration = 1
	corrected, err := f.service.RecordJudgments(ctx, grant, changed)
	if err != nil || corrected.Generations[0].Generation != 2 || corrected.Generations[0].FindingID != findingID {
		t.Fatalf("Finding correction: %+v %v", corrected, err)
	}
	grant.WriteFenced = true
	if replay, err := f.service.RecordEvidence(ctx, grant, evidenceInput); err != nil || !replay.Replayed {
		t.Fatalf("fenced evidence reconciliation: %+v %v", replay, err)
	}
	if replay, err := f.service.RecordJudgments(ctx, grant, judgment); err != nil || !replay.Replayed || !replay.Material.Ref.SameExact(accepted.Material.Ref) {
		t.Fatalf("judgment reconciliation: %+v %v", replay, err)
	}
	changed.SubmissionID = "closed"
	changed.Items[0].BaseGeneration = 1
	if _, err = f.service.RecordJudgments(ctx, grant, changed); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("late write: %v", err)
	}
	if _, err = runs.TransitionRun(ctx, runID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "fixture_failure"}); err != nil {
		t.Fatal(err)
	}
	if err = rundeletion.New(f.pool, storage).DeleteReleasedTerminalRun(ctx, "owner", runID); err != nil {
		t.Fatal(err)
	}
	project, _ := art.Project(f.audit.ProjectID)
	if _, err = project.ReadExact(ctx, accepted.Material.Exact()); err != nil {
		t.Fatalf("accepted result lost on failed Run purge: %v", err)
	}
	if _, err = project.ReadExact(ctx, evidence.Sources[0].Exact()); err != nil {
		t.Fatalf("partial evidence lost on purge: %v", err)
	}
	current, err := f.service.GetProjectFinding(ctx, "owner", f.audit.ProjectID, findingID)
	if err != nil || current.State != nil {
		t.Fatalf("assessment changed: %+v %v", current, err)
	}
	history, err := f.service.findings.ListProjectSourceHistory(ctx, "owner", f.audit.ProjectID, selection.ReceiptID, "", 64, nil)
	if err != nil || len(history.Items) != 64 {
		t.Fatalf("history: %d %v", len(history.Items), err)
	}
	for _, meta := range history.Items[:3] {
		if !strings.Contains(meta.Ref.Name, ".check-") {
			t.Fatal("new acknowledgements missing from the 10000-record history")
		}
	}
	if run.AuditExecutionID != nil || run.PublicationMode != runstore.PublicationOrdinary {
		t.Fatal("ordinary Run acquired Audit identity")
	}
}

const projectFindingScaleHistorySQL = `WITH copied AS (
 INSERT INTO artifact_bindings
 SELECT (jsonb_populate_record(NULL::artifact_bindings,to_jsonb(seed)||jsonb_build_object(
 'name',$3::text||'scale.'||lpad(n::text,6,'0')))).*
 FROM artifact_bindings AS seed CROSS JOIN generate_series(1,10000) AS n
 WHERE scope_kind='project' AND scope_id=$1 AND namespace='finding-history' AND name=$2
 RETURNING name
)
 INSERT INTO artifact_binding_revisions
 SELECT (jsonb_populate_record(NULL::artifact_binding_revisions,to_jsonb(seed)||jsonb_build_object('name',copied.name))).*
 FROM artifact_binding_revisions AS seed CROSS JOIN copied
 WHERE seed.scope_kind='project' AND seed.scope_id=$1 AND seed.namespace='finding-history' AND seed.name=$2`
