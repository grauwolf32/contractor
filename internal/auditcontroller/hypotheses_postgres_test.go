//go:build integration

package auditcontroller

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresCommonHypothesisChecksRetainPerItemRevisions(t *testing.T) {
	h := newCommonControllerHarness(t, func(p *auditcontract.Program) {
		p.Hypotheses = map[string]auditcontract.HypothesisSeed{"bug": {Statement: "A reader can access another owner's record in the selected fixture", Scope: map[string]string{"actor": "reader"}, Grounds: "An explicit suspected violation"}, "property": {Statement: "Only the owner can read a record in the selected fixture", Scope: map[string]string{"actor": "owner"}, Grounds: "A selected positive property"}}
		p.Workflows["check"] = auditcontract.ProgramWorkflow{Ref: "audit-contract-check@1", Inputs: map[string]auditcontract.InputMapping{"source": {Source: "audit-input", Name: "source"}, "context": {Source: "task-context"}}, Parameters: map[string]string{}, Outputs: map[string]string{"result": "result"}}
		p.InitialWork = []auditcontract.InitialWork{{Key: "bug-check", Workflow: "check", Objective: "Check the bug claim", Hypothesis: "bug"}, {Key: "property-check", Workflow: "check", Objective: "Check the positive property", Hypothesis: "property"}}
		p.Limits.BatchSize = 2
		p.Limits.MaxHypotheses = 8
		p.Limits.MaxHypothesisRevisions = 16
		p.Limits.MaxHypothesisBytes = 1 << 20
	})
	ctx := t.Context()
	controller := h.commonController(t)
	step := func() {
		t.Helper()
		changed, err := controller.RunOnce(ctx)
		if err != nil || !changed {
			t.Fatalf("step=%v error=%v", changed, err)
		}
	}
	step() // deterministic seed admission; no generation Run
	step() // common Task acceptance
	step() // a single ordinary Run over two exact members
	executions, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatalf("executions=%+v error=%v", executions, err)
	}
	runID := *executions[0].RunID
	run, _ := h.artifacts.Run(runID)
	trusted, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "context"})
	if err != nil {
		t.Fatal(err)
	}
	var context struct {
		Items []struct {
			ItemKey    string                   `json:"item_key"`
			TaskID     string                   `json:"task_id"`
			Hypothesis auditcontract.Hypothesis `json:"hypothesis"`
			Evidence   []auditcontract.Material `json:"evidence"`
		}
	}
	if json.Unmarshal(trusted.Payload.Data, &context) != nil || len(context.Items) != 2 || context.Items[0].Hypothesis.ID == context.Items[1].Hypothesis.ID {
		t.Fatal("lost exact independent context membership")
	}
	audit, err := h.service.Get(ctx, "owner", h.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	paused, err := h.service.Pause(ctx, auditservice.MutationParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "pause-hypotheses", RequestDigest: contentdigest.Bytes([]byte("pause-hypotheses"))})
	if err != nil {
		t.Fatal(err)
	}
	source, err := h.service.Hypotheses(ctx, auditstore.ListHypothesesParams{OwnerID: "owner", AuditID: audit.AuditID, Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	original := source.Items[0]
	if _, _, err := h.service.WriteHypothesis(ctx, auditservice.HypothesisMutationParams{MutationParams: auditservice.MutationParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: paused.Audit.Revision, IdempotencyKey: "edit-during-run", RequestDigest: contentdigest.Bytes([]byte("edit-during-run"))}, HypothesisID: original.Document.ID, Content: auditservice.HypothesisContent{Key: original.SourceKey, Statement: "A different scoped assertion", Scope: original.Document.Scope, Grounds: "An explicit later edit"}}); err != nil {
		t.Fatal(err)
	}
	items := []any{}
	for index, member := range context.Items {
		verdict := "supported"
		if index == 1 {
			verdict = "refuted"
		}
		items = append(items, map[string]any{"item_key": member.ItemKey, "verdict": verdict, "rationale": "Inspected exact fixture evidence", "tested_conditions": []string{"Only the selected fixture and actor"}, "limitations": []string{}, "evidence": member.Evidence})
	}
	data, _ := json.Marshal(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": items})
	written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: "result"}, artifacts.Payload{MediaType: "application/json", Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifactstest.BindOutput(ctx, h.artifacts, runID, "result", written.Ref); err != nil {
		t.Fatal(err)
	}
	if err := artifactstest.FreezeOutputs(ctx, h.artifacts, runID); err != nil {
		t.Fatal(err)
	}
	for _, transition := range [][2]runstore.WorkflowRunState{{runstore.RunPending, runstore.RunRunning}, {runstore.RunRunning, runstore.RunSucceeded}} {
		if _, err := h.runs.TransitionRun(ctx, runID, transition[0], transition[1], runstore.Reason{Code: "fixture"}); err != nil {
			t.Fatal(err)
		}
	}
	step()
	step() // terminal observation and collection while paused
	for _, member := range context.Items {
		task, err := h.service.Task(ctx, "owner", audit.AuditID, member.TaskID)
		if err != nil || task.State != auditstore.TaskSettled || task.Document.Hypothesis.Revision != 1 || task.Hypothesis.Statement != member.Hypothesis.Statement {
			t.Fatalf("accepted Task rebound: %+v error=%v", task, err)
		}
		read, err := h.service.ReadArtifact(ctx, "owner", audit.AuditID, "task/"+member.TaskID+"/result", *task.Result.Ref.Revision)
		if err != nil {
			t.Fatal(err)
		}
		var result struct {
			Items []struct {
				Verdict string `json:"verdict"`
				Origin  struct {
					Hypothesis auditcontract.HypothesisRef `json:"hypothesis"`
					TaskID     string                      `json:"task_id"`
				} `json:"origin"`
			}
		}
		if json.Unmarshal(read.Payload.Data, &result) != nil || len(result.Items) != 2 {
			t.Fatalf("bad exact result: %s", read.Payload.Data)
		}
		matches := 0
		for _, item := range result.Items {
			if item.Origin.TaskID == member.TaskID {
				matches++
				if item.Origin.Hypothesis.ID != member.Hypothesis.ID || item.Origin.Hypothesis.Revision != 1 {
					t.Fatal("result rebound to another hypothesis")
				}
			}
		}
		if matches != 1 {
			t.Fatalf("expected one matching conclusion, got %d", matches)
		}
		history, err := h.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: audit.AuditID, HypothesisID: member.Hypothesis.ID, Limit: 10})
		if err != nil || len(history.Tasks) != 1 || history.Tasks[0].Document.TaskID != member.TaskID {
			t.Fatal("result fanned out to another hypothesis", err)
		}
	}
	var findings int
	if err := h.pool.QueryRow(ctx, `SELECT count(*) FROM project_findings`).Scan(&findings); err != nil || findings != 0 {
		t.Fatalf("verdict polarity published a Finding: %d error=%v", findings, err)
	}
	page, err := h.service.Hypotheses(ctx, auditstore.ListHypothesesParams{OwnerID: "owner", AuditID: audit.AuditID, Limit: 10})
	if err != nil || page.FixedDenominator != 2 || page.Items[0].Document.Revision != 2 {
		t.Fatalf("coverage or history changed: %+v error=%v", page, err)
	}
}
