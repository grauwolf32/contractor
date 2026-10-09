//go:build integration

package auditcontroller

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/randomid"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func preparationControllerConfig(t *testing.T, dependent bool, customize ...func(map[string]string)) *config.Snapshot {
	t.Helper()
	return loadControllerConfigWithFindings(t, 1, 10, false, func(files map[string]string) {
		files["workflows/prepare.yaml"] = `apiVersion: contractor/v1alpha1
kind: Workflow
metadata: {name: prepare-checklist, version: "1"}
spec:
  parameters: {}
  inputs:
    source: {required: true, mediaTypes: [application/json]}
    previous: {required: false, mediaTypes: [application/json]}
  outputs:
    checklist: {required: true, mediaTypes: [application/json]}
    diagnostics: {required: true, mediaTypes: [application/json]}
  entryStage: prepare
  stages:
    prepare:
      objective: Prepare a checklist
      instructions: {ref: instructions/planner.md}
      planner: passthrough@1
      agents:
        worker: {template: audit-worker@1}
      context:
        artifacts:
          source: {namespace: inputs, name: source, required: true}
      result:
        artifacts:
          checklist: {required: true, mediaTypes: [application/json], from: {namespace: worker, name: checklist}}
          diagnostics: {required: true, mediaTypes: [application/json], from: {namespace: worker, name: diagnostics}}
      workflowOutputs: {checklist: checklist, diagnostics: diagnostics}
      on:
        succeeded: {succeed: {}}
        failed: {fail: {}}
        interrupted: {fail: {}}
`
		roles := `    z-seed:
      kind: prepare
      ref: prepare-checklist@1
      maxRunAttempts: 2
      inputs:
        source: {source: audit-input, name: checklist}
      parameters: {}
      outputs: {list: checklist, diagnostics: diagnostics}
`
		if dependent {
			// Alphabetical order deliberately differs from dependency order.
			roles += `    a-refine:
      kind: prepare
      ref: prepare-checklist@1
      maxRunAttempts: 2
      inputs:
        source: {source: audit-input, name: checklist}
        previous: {source: prepare-output, role: z-seed, name: list}
      parameters: {}
      outputs: {list: checklist, diagnostics: diagnostics}
`
		}
		files["audit-profiles/checklist.yaml"] = strings.Replace(files["audit-profiles/checklist.yaml"], "  workflows:\n", "  workflows:\n"+roles, 1)
		for _, apply := range customize {
			apply(files)
		}
	})
}

func preparationHarness(t *testing.T, dependent bool) (context.Context, *postgresControllerHarness, *Controller) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	t.Cleanup(cancel)
	h := newPostgresControllerHarnessWithConfig(t, ctx, 0, 1, preparationControllerConfig(t, dependent))
	if h.started.Audit.Phase != auditdomain.AuditPhasePreparing || h.started.Audit.CurrentRoundID != nil || len(h.started.Items) != 0 {
		t.Fatalf("start invented a Round: %+v", h.started)
	}
	baseline, err := auditservice.DecodeBaseline(h.started.Audit.BaselineSnapshot)
	if err != nil || baseline.Inventory != nil || len(baseline.Inputs) != 1 {
		t.Fatalf("original baseline: %+v %v", baseline, err)
	}
	return ctx, h, preparationController(t, h)
}

func preparationController(t *testing.T, h *postgresControllerHarness) *Controller {
	t.Helper()
	c := h.controllerWithCollector(t)
	// Independent controller processes use fresh Run IDs. Durable execution
	// identity and submission keys still come from the deterministic builder.
	c.options.NewID = randomid.New
	return c
}

func preparationStep(t *testing.T, ctx context.Context, c *Controller) {
	t.Helper()
	if worked, err := c.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("preparation reconciliation worked=%t: %v", worked, err)
	}
}

func preparationExecutions(t *testing.T, ctx context.Context, h *postgresControllerHarness, count int) []auditstore.Execution {
	t.Helper()
	executions, err := h.audits.ListPreparationExecutions(ctx, h.started.Audit.AuditID)
	if err != nil || len(executions) != count {
		t.Fatalf("preparation executions=%+v want=%d: %v", executions, count, err)
	}
	return executions
}

func finishPreparationRun(t *testing.T, ctx context.Context, h *postgresControllerHarness, execution auditstore.Execution, outcome runstore.WorkflowRunState, omitDiagnostics bool, outputs ...map[string]artifacts.Payload) {
	t.Helper()
	if execution.RunID == nil {
		t.Fatalf("execution has no Run: %+v", execution)
	}
	runID := *execution.RunID
	if outcome == runstore.RunSucceeded {
		runArtifacts, _ := h.artifacts.Run(runID)
		baseline, _ := auditservice.DecodeBaseline(h.started.Audit.BaselineSnapshot)
		projectArtifacts, _ := h.artifacts.Project(h.started.Audit.ProjectID)
		source, err := projectArtifacts.Read(ctx, baseline.Inputs["checklist"].Ref)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range []string{"checklist", "diagnostics"} {
			if name == "diagnostics" && omitDiagnostics {
				continue
			}
			payload := source.Payload
			if name == "diagnostics" {
				payload = artifacts.Payload{MediaType: "application/json", Data: []byte(`{"valid":true}`)}
			}
			if len(outputs) == 1 {
				if override, ok := outputs[0][name]; ok {
					payload = override
				}
			}
			written, err := runArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "worker", Name: name}, payload, nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := h.artifacts.BindOutputExact(ctx, runID, name, written.Ref, nil); err != nil {
				t.Fatal(err)
			}
		}
		if err := h.artifacts.FreezeRunOutputs(ctx, runID); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := h.runs.TransitionRun(ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "test-admitted"}); err != nil {
		t.Fatal(err)
	}
	if _, err := h.runs.TransitionRun(ctx, runID, runstore.RunRunning, outcome, runstore.Reason{Code: "test-terminal"}); err != nil {
		t.Fatal(err)
	}
}

func TestPostgresPreparationDependentRunsRetainExactOutputs(t *testing.T) {
	ctx, h, c := preparationHarness(t, true)
	originalBaseline := string(h.started.Audit.BaselineSnapshot)
	preparationStep(t, ctx, c)
	seed := preparationExecutions(t, ctx, h, 1)[0]
	if seed.WorkflowRole != "z-seed" || seed.Role != auditstore.ExecutionPrepare || seed.RoundID != nil || seed.RunID == nil || seed.Preparation == nil {
		t.Fatalf("first preparation execution: %+v", seed)
	}
	run, err := h.runs.GetRun(ctx, *seed.RunID)
	if err != nil || run.PublicationMode != runstore.PublicationAuditManaged || run.AuditExecutionID == nil || *run.AuditExecutionID != seed.ExecutionID {
		t.Fatalf("trusted ordinary Run: %+v %v", run, err)
	}
	finishPreparationRun(t, ctx, h, seed, runstore.RunSucceeded, false)
	preparationStep(t, ctx, c) // terminal observation
	preparationStep(t, ctx, c) // atomic output acceptance
	output, err := h.audits.GetPreparationOutput(ctx, h.started.Audit.AuditID, "z-seed", "list")
	if err != nil || output.Output.Source.Ref.Revision == nil || output.Output.Retained.Ref.Revision == nil {
		t.Fatalf("accepted exact output: %+v %v", output, err)
	}
	preparationStep(t, ctx, c)
	refine := preparationExecutions(t, ctx, h, 2)[0] // sorted by authored role
	if refine.WorkflowRole != "a-refine" || !refine.Preparation.Inputs["previous"].Ref.SameExact(output.Output.Retained.Ref) {
		t.Fatalf("dependent input is not retained exact output: %+v", refine)
	}
	refineArtifacts, _ := h.artifacts.Run(*refine.RunID)
	forked, err := refineArtifacts.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "previous"})
	if err != nil || auditdomain.DigestBytes(forked.Payload.Data) != output.Output.Retained.Digest {
		t.Fatalf("dependent Run did not receive the exact bytes: %+v %v", forked, err)
	}
	lineage, err := refineArtifacts.ListLineage(ctx, forked.Ref, artifacts.LineagePageQuery{Limit: 2})
	if err != nil || len(lineage) != 1 || !lineage[0].Source.SameExact(output.Output.Retained.Ref) || lineage[0].SourceScope != artifacts.ScopeProject {
		t.Fatalf("dependent Run fork lost exact Project lineage: %+v %v", lineage, err)
	}
	finishPreparationRun(t, ctx, h, refine, runstore.RunSucceeded, false)
	preparationStep(t, ctx, c)
	preparationStep(t, ctx, c)
	preparationStep(t, ctx, c)
	audit, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
	if err != nil || audit.Phase != auditdomain.AuditPhaseInventory || audit.State != auditstore.AuditActive || audit.CurrentRoundID != nil || audit.ReservedRunCount != 2 || audit.OutstandingRunCount != 0 || audit.RetainedEvidenceBytes <= 0 || string(audit.BaselineSnapshot) != originalBaseline {
		t.Fatalf("preparation boundary: %+v %v", audit, err)
	}
	projection, err := h.service.Preparation(ctx, audit.OwnerID, audit.AuditID)
	if err != nil || len(projection) != 2 || projection["a-refine"].Status != auditdomain.PreparationAccepted || len(projection["z-seed"].Outputs) != 2 {
		t.Fatalf("public preparation projection: %+v %v", projection, err)
	}
	for _, execution := range preparationExecutions(t, ctx, h, 2) {
		if execution.State != auditstore.ExecutionCollected || len(execution.PreparationOutputs) != 2 {
			t.Fatalf("incomplete receipt: %+v", execution)
		}
		if err := h.runs.DeleteReleasedTerminalRun(ctx, audit.OwnerID, *execution.RunID); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := h.audits.GetPreparationOutput(ctx, audit.AuditID, "z-seed", "diagnostics"); err != nil {
		t.Fatalf("source Run deletion lost diagnostics: %v", err)
	}
	projection, err = h.service.Preparation(ctx, audit.OwnerID, audit.AuditID)
	if err != nil || len(projection["z-seed"].Outputs) != 2 || projection["z-seed"].RunID == nil {
		t.Fatalf("source Run deletion lost public provenance: %+v %v", projection, err)
	}
	if _, err := h.service.Preparation(ctx, "another-owner", audit.AuditID); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("preparation disclosure to another owner: %v", err)
	}
	var roundCount, itemCount, receiptCount int
	if err := h.pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM audit_rounds), (SELECT count(*) FROM audit_items), (SELECT count(*) FROM audit_executions WHERE collection_disposition=$1)`, auditstore.CollectionAccepted).Scan(&roundCount, &itemCount, &receiptCount); err != nil || roundCount != 0 || itemCount != 0 || receiptCount != 2 {
		t.Fatalf("durable counts rounds=%d items=%d receipts=%d: %v", roundCount, itemCount, receiptCount, err)
	}
	// Restart at the accepted boundary must never create another prepare Run.
	if worked, err := preparationController(t, h).RunOnce(ctx); err != nil || worked {
		t.Fatalf("accepted boundary replay worked=%t: %v", worked, err)
	}
}

type lostPreparationSubmission struct {
	creator RunCreator
}

type preparationRunCreatorFunc func(context.Context, runservice.AuditCreateParams) (runservice.CreateResult, error)

func (f preparationRunCreatorFunc) CreateAudit(ctx context.Context, params runservice.AuditCreateParams) (runservice.CreateResult, error) {
	return f(ctx, params)
}

func TestPostgresPreparationStaleClaimBeforeRunBindingRecoversSameIntent(t *testing.T) {
	ctx, h, c := preparationHarness(t, false)
	var staleClaim auditstore.ControllerClaim
	c.creator = preparationRunCreatorFunc(func(ctx context.Context, params runservice.AuditCreateParams) (runservice.CreateResult, error) {
		staleClaim = params.Claim
		if _, err := h.pool.Exec(ctx, `UPDATE audit_controller_claims SET claimed_at=clock_timestamp()-interval '2 seconds', expires_at=clock_timestamp()-interval '1 second' WHERE audit_id=$1`, params.Claim.AuditID); err != nil {
			return runservice.CreateResult{}, err
		}
		return h.runService.CreateAudit(ctx, params)
	})
	if _, err := c.RunOnce(ctx); !errors.Is(err, auditstore.ErrClaimLost) {
		t.Fatalf("stale controller bound a Run: %v", err)
	}
	before := preparationExecutions(t, ctx, h, 1)[0]
	var count int
	if err := h.pool.QueryRow(ctx, `SELECT count(*) FROM workflow_runs`).Scan(&count); err != nil || count != 0 || before.RunID != nil || before.State != auditstore.ExecutionIntent {
		t.Fatalf("stale submission leaked Run: count=%d execution=%+v %v", count, before, err)
	}
	preparationStep(t, ctx, preparationController(t, h))
	after := preparationExecutions(t, ctx, h, 1)[0]
	if after.ExecutionID != before.ExecutionID || after.SubmissionKey != before.SubmissionKey || after.RunID == nil {
		t.Fatalf("new controller replaced pending intent: %+v", after)
	}
	if _, err := h.audits.GetReconcileSnapshot(ctx, staleClaim); !errors.Is(err, auditstore.ErrClaimLost) {
		t.Fatalf("old controller retained authority: %v", err)
	}
	audit, _ := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
	if audit.ReservedRunCount != 1 || audit.SubmittedRunCount != 1 || audit.OutstandingRunCount != 1 || audit.CurrentRoundID != nil {
		t.Fatalf("stale claim recovery budgets: %+v", audit)
	}
}

func (l lostPreparationSubmission) CreateAudit(ctx context.Context, params runservice.AuditCreateParams) (runservice.CreateResult, error) {
	_, err := l.creator.CreateAudit(ctx, params)
	if err != nil {
		return runservice.CreateResult{}, err
	}
	return runservice.CreateResult{}, context.DeadlineExceeded
}

type lostPreparationIntent struct{ Store }

func (l lostPreparationIntent) CreateExecutionIntent(ctx context.Context, params auditstore.CreateExecutionIntentParams) (auditstore.Execution, bool, error) {
	_, _, err := l.Store.CreateExecutionIntent(ctx, params)
	if err != nil {
		return auditstore.Execution{}, false, err
	}
	return auditstore.Execution{}, false, context.DeadlineExceeded
}

type lostPreparationCollection struct{ *auditstore.PostgresStore }

func (l lostPreparationCollection) Collect(ctx context.Context, params auditstore.CollectParams) (auditstore.CollectionReceipt, bool, error) {
	_, _, err := l.PostgresStore.Collect(ctx, params)
	if err != nil {
		return auditstore.CollectionReceipt{}, false, err
	}
	return auditstore.CollectionReceipt{}, false, context.DeadlineExceeded
}

func TestPostgresPreparationRecoversLostIntentAndRunResponses(t *testing.T) {
	for _, boundary := range []string{"intent", "run"} {
		t.Run(boundary, func(t *testing.T) {
			ctx, h, c := preparationHarness(t, false)
			if boundary == "intent" {
				c.store = lostPreparationIntent{h.audits}
			} else {
				c.creator = lostPreparationSubmission{h.runService}
			}
			if _, err := c.RunOnce(ctx); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("lost %s response: %v", boundary, err)
			}
			execution := preparationExecutions(t, ctx, h, 1)[0]
			identity := execution.ExecutionID
			if boundary == "intent" {
				if execution.State != auditstore.ExecutionIntent || execution.RunID != nil {
					t.Fatalf("lost intent response bound a Run: %+v", execution)
				}
				preparationStep(t, ctx, preparationController(t, h))
			}
			execution = preparationExecutions(t, ctx, h, 1)[0]
			if execution.ExecutionID != identity || execution.RunID == nil {
				t.Fatalf("recovery changed execution identity: %+v", execution)
			}
			finishPreparationRun(t, ctx, h, execution, runstore.RunSucceeded, false)
			restarted := preparationController(t, h)
			preparationStep(t, ctx, restarted)
			preparationStep(t, ctx, restarted)
			preparationStep(t, ctx, restarted)
			var runs, receipts int
			if err := h.pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM workflow_runs), (SELECT count(*) FROM audit_executions WHERE collection_disposition=$1)`, auditstore.CollectionAccepted).Scan(&runs, &receipts); err != nil || runs != 1 || receipts != 1 {
				t.Fatalf("recovery duplicated Run/receipt: runs=%d receipts=%d %v", runs, receipts, err)
			}
			audit, _ := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
			if audit.ReservedRunCount != 1 || audit.SubmittedRunCount != 1 || audit.OutstandingRunCount != 0 || audit.Phase != auditdomain.AuditPhaseInventory {
				t.Fatalf("recovery budgets/phase: %+v", audit)
			}
		})
	}
}

func TestPostgresPreparationRecoversLostCollectionResponse(t *testing.T) {
	ctx, h, c := preparationHarness(t, false)
	preparationStep(t, ctx, c)
	finishPreparationRun(t, ctx, h, preparationExecutions(t, ctx, h, 1)[0], runstore.RunSucceeded, false)
	preparationStep(t, ctx, c)
	access, _ := auditimport.NewArtifactAccess(h.artifacts)
	c.collector, _ = auditimport.New(lostPreparationCollection{h.audits}, h.runs, access)
	if _, err := c.RunOnce(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("lost collection response: %v", err)
	}
	before, _ := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
	execution := preparationExecutions(t, ctx, h, 1)[0]
	if execution.State != auditstore.ExecutionCollected || len(execution.PreparationOutputs) != 2 || before.OutstandingRunCount != 0 {
		t.Fatalf("collection did not commit atomically: %+v %+v", before, execution)
	}
	restarted := preparationController(t, h)
	preparationStep(t, ctx, restarted)
	if worked, err := restarted.RunOnce(ctx); err != nil || worked {
		t.Fatalf("collection replay advanced twice: %t %v", worked, err)
	}
	after, _ := h.audits.Get(ctx, before.OwnerID, before.AuditID)
	if after.ReservedRunCount != 1 || after.RetainedEvidenceBytes != before.RetainedEvidenceBytes || after.Phase != auditdomain.AuditPhaseInventory {
		t.Fatalf("collection replay charged twice: %+v", after)
	}
	if err := h.runs.DeleteReleasedTerminalRun(ctx, after.OwnerID, *execution.RunID); err != nil {
		t.Fatalf("committed collection did not release Run: %v", err)
	}
}

func preparationMutation(audit auditstore.Audit, key string) auditservice.MutationParams {
	return auditservice.MutationParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID,
		ExpectedRevision: audit.Revision, IdempotencyKey: key, RequestDigest: postgresDigest(key)}
}

func TestPostgresPreparationPauseAndDeadlineCollectWithoutAdvancing(t *testing.T) {
	for _, boundary := range []string{"pause", "deadline"} {
		t.Run(boundary, func(t *testing.T) {
			ctx, h, c := preparationHarness(t, true)
			preparationStep(t, ctx, c)
			execution := preparationExecutions(t, ctx, h, 1)[0]
			audit, _ := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
			if boundary == "pause" {
				if _, err := h.service.Pause(ctx, preparationMutation(audit, "pause")); err != nil {
					t.Fatal(err)
				}
			} else {
				if _, err := h.pool.Exec(ctx, `UPDATE audits SET deadline_at=clock_timestamp()-interval '1 second' WHERE audit_id=$1`, audit.AuditID); err != nil {
					t.Fatal(err)
				}
				preparationStep(t, ctx, c)
			}
			finishPreparationRun(t, ctx, h, execution, runstore.RunSucceeded, false)
			preparationStep(t, ctx, c)
			preparationStep(t, ctx, c)
			audit, _ = h.audits.Get(ctx, audit.OwnerID, audit.AuditID)
			if audit.State != auditstore.AuditPaused || audit.Phase != auditdomain.AuditPhasePreparing || audit.CurrentRoundID != nil || audit.OutstandingRunCount != 0 || audit.ReservedRunCount != 1 {
				t.Fatalf("paused preparation advanced: %+v", audit)
			}
			preparationExecutions(t, ctx, h, 1)
			resume := preparationMutation(audit, "resume")
			unlimited := 0
			resume.DeadlineSeconds = &unlimited
			resumed, err := h.service.Resume(ctx, resume)
			if err != nil || resumed.Audit.Phase != auditdomain.AuditPhasePreparing || resumed.Audit.CurrentRoundID != nil {
				t.Fatalf("resume without Round: %+v %v", resumed, err)
			}
			preparationStep(t, ctx, preparationController(t, h))
			preparationExecutions(t, ctx, h, 2)
		})
	}
}

func TestPostgresPreparationMissingOutputRetriesThenFailsWithoutRound(t *testing.T) {
	ctx, h, c := preparationHarness(t, false)
	for attempt := 1; attempt <= 2; attempt++ {
		preparationStep(t, ctx, c)
		executions := preparationExecutions(t, ctx, h, attempt)
		execution := executions[len(executions)-1]
		finishPreparationRun(t, ctx, h, execution, runstore.RunSucceeded, true)
		preparationStep(t, ctx, c)
		preparationStep(t, ctx, c)
		if _, err := h.audits.GetPreparationOutput(ctx, h.started.Audit.AuditID, "z-seed", "list"); !errors.Is(err, auditstore.ErrNotFound) {
			t.Fatalf("partial required output set was accepted: %v", err)
		}
	}
	for step := 0; step < 5; step++ {
		audit, _ := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
		if audit.State == auditstore.AuditFailed {
			if audit.StopReason == nil || audit.StopReason.Code != "preparation_failed" || audit.Hold != auditstore.HoldReleased || audit.OutstandingRunCount != 0 || audit.ReservedRunCount != 2 || audit.CurrentRoundID != nil || audit.RetainedEvidenceBytes <= 0 {
				t.Fatalf("exhausted preparation: %+v", audit)
			}
			for _, execution := range preparationExecutions(t, ctx, h, 2) {
				if execution.State != auditstore.ExecutionCollected || len(execution.PreparationOutputs) != 0 {
					t.Fatalf("partial output receipt: %+v", execution)
				}
				if err := h.runs.DeleteReleasedTerminalRun(ctx, audit.OwnerID, *execution.RunID); err != nil {
					t.Fatal(err)
				}
				if _, err := h.audits.GetArtifactLink(ctx, audit.AuditID, "prepare-attempt:"+execution.ExecutionID+":list"); err != nil {
					t.Fatalf("source Run deletion lost failed-attempt evidence: %v", err)
				}
			}
			report, err := h.service.GetReport(ctx, audit.OwnerID, audit.AuditID)
			if err != nil || report.Status != auditservice.ReportUnavailable {
				t.Fatalf("failed preparation invented report: %+v %v", report, err)
			}
			return
		}
		preparationStep(t, ctx, c)
	}
	t.Fatal("exhausted preparation did not fail")
}

func TestPostgresPreparationCancelAndDeleteDrainBeforeDependencies(t *testing.T) {
	for _, action := range []string{"cancel", "delete"} {
		t.Run(action, func(t *testing.T) {
			ctx, h, c := preparationHarness(t, true)
			preparationStep(t, ctx, c)
			execution := preparationExecutions(t, ctx, h, 1)[0]
			audit, _ := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
			var err error
			if action == "cancel" {
				_, err = h.service.Cancel(ctx, preparationMutation(audit, action))
			} else {
				_, err = h.service.Delete(ctx, preparationMutation(audit, action))
			}
			if err != nil {
				t.Fatal(err)
			}
			preparationStep(t, ctx, c) // request ordinary Run cancellation
			run, err := h.runs.GetRun(ctx, *execution.RunID)
			if err != nil || run.State != runstore.RunCancelling {
				t.Fatalf("child cancellation: %+v %v", run, err)
			}
			if _, err := h.runs.TransitionRun(ctx, run.RunID, runstore.RunCancelling, runstore.RunCancelled, runstore.Reason{Code: "test-cancelled"}); err != nil {
				t.Fatal(err)
			}
			for step := 0; step < 10; step++ {
				audit, err = h.audits.Get(ctx, audit.OwnerID, audit.AuditID)
				if action == "delete" && errors.Is(err, auditstore.ErrNotFound) {
					if _, err := h.runs.GetRun(ctx, run.RunID); !errors.Is(err, runstore.ErrNotFound) {
						t.Fatalf("deleted Audit left a Run: %v", err)
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				if audit.CurrentRoundID != nil || audit.ReservedRunCount != 1 {
					t.Fatalf("closed preparation started dependent work: %+v", audit)
				}
				if action == "cancel" && audit.State == auditstore.AuditCancelled {
					if audit.OutstandingRunCount != 0 || audit.Hold != auditstore.HoldReleased {
						t.Fatalf("cancelled Audit did not drain: %+v", audit)
					}
					if err := h.runs.DeleteReleasedTerminalRun(ctx, audit.OwnerID, run.RunID); err != nil {
						t.Fatal(err)
					}
					return
				}
				preparationStep(t, ctx, c)
			}
			t.Fatal("preparation did not settle")
		})
	}
}

func TestPostgresPreparationSharedSubmissionAndEvidenceBudgets(t *testing.T) {
	for _, budget := range []string{"submission", "evidence"} {
		t.Run(budget, func(t *testing.T) {
			ctx, h, c := preparationHarness(t, true)
			if budget == "evidence" {
				if _, err := h.pool.Exec(ctx, `UPDATE audits SET max_evidence_bytes=1 WHERE audit_id=$1`, h.started.Audit.AuditID); err != nil {
					t.Fatal(err)
				}
			}
			preparationStep(t, ctx, c)
			execution := preparationExecutions(t, ctx, h, 1)[0]
			finishPreparationRun(t, ctx, h, execution, runstore.RunSucceeded, false)
			preparationStep(t, ctx, c)
			preparationStep(t, ctx, c)
			if budget == "submission" {
				// An operator may lower the remaining window to the consumed count.
				if _, err := h.pool.Exec(ctx, `UPDATE audits SET max_submitted_runs=1, max_items_per_round=1, max_items_total=1 WHERE audit_id=$1`, h.started.Audit.AuditID); err != nil {
					t.Fatal(err)
				}
			}
			for step := 0; step < 5; step++ {
				audit, _ := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
				if audit.State == auditstore.AuditFailed {
					wantCode := "evidence_budget_exhausted"
					if budget == "submission" {
						wantCode = "submission_budget_exhausted"
					}
					if audit.StopReason == nil || audit.StopReason.Code != wantCode || audit.ReservedRunCount != 1 || audit.SubmittedRunCount != 1 || audit.OutstandingRunCount != 0 || audit.Hold != auditstore.HoldReleased || audit.CurrentRoundID != nil {
						t.Fatalf("shared %s budget: %+v", budget, audit)
					}
					if budget == "evidence" && audit.RetainedEvidenceBytes != 0 {
						t.Fatalf("over-budget outputs were retained: %+v", audit)
					}
					preparationExecutions(t, ctx, h, 1)
					return
				}
				preparationStep(t, ctx, c)
			}
			t.Fatal("budget did not close preparation")
		})
	}
}
