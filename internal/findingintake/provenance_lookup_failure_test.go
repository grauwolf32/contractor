//go:build integration

package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/controlplane"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// A storage failure while reading the submitting Run, its StageExecution or
// the Audit provenance of an Audit-managed Run is not a denied grant or a
// deselected finding tool: the submission keeps the database cause, commits
// nothing and succeeds once the store answers again. Each case breaks exactly
// one read; the reads before it still succeed.
func TestPostgresFindingSubmissionKeepsProvenanceLookupFailure(t *testing.T) {
	for _, test := range []struct {
		name, breakRead, restoreRead, sqlState string
		auditManaged                           bool
	}{
		{
			name:        "Run",
			breakRead:   `ALTER TABLE workflow_runs RENAME TO lookup_unavailable_runs`,
			restoreRead: `ALTER TABLE lookup_unavailable_runs RENAME TO workflow_runs`,
			sqlState:    persistencepostgres.SQLStateUndefinedTable,
		},
		{
			// The allocation check joins stage_executions on its keys only;
			// the StageExecution read is the first to select the snapshot.
			name:        "StageExecution",
			breakRead:   `ALTER TABLE stage_executions RENAME COLUMN stage_spec_snapshot TO lookup_unavailable_snapshot`,
			restoreRead: `ALTER TABLE stage_executions RENAME COLUMN lookup_unavailable_snapshot TO stage_spec_snapshot`,
			sqlState:    persistencepostgres.SQLStateUndefinedColumn,
		},
		{
			// Only an Audit-managed Run reads its Audit execution, last.
			name:         "Audit",
			breakRead:    `ALTER TABLE audit_executions RENAME TO lookup_unavailable_audit_executions`,
			restoreRead:  `ALTER TABLE lookup_unavailable_audit_executions RENAME TO audit_executions`,
			sqlState:     persistencepostgres.SQLStateUndefinedTable,
			auditManaged: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			pool := isolatedFindingPool(t, ctx)
			const ownerID, projectID, runID = "lookup-owner", "lookup-project", "lookup-run"
			if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
				ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject,
				Name: "Lookup project", IdempotencyKey: "lookup-project",
				RequestDigest: auditdomain.DigestBytes([]byte("lookup-project")),
			}); err != nil {
				t.Fatal(err)
			}
			var grant controlplane.AllocationGrant
			var auditOrigin *AuditOrigin
			if test.auditManaged {
				grant, auditOrigin = auditSubmissionGrant(t, ctx, pool, ownerID, projectID, runID)
			} else {
				grant = quotaSubmissionGrant(t, ctx, pool, ownerID, projectID, runID)
			}
			service, err := New(pool, artifactpgtest.NewStorage())
			if err != nil {
				t.Fatal(err)
			}
			input := testSubmission("lookup-invocation", "lookup-candidate", fixtureEvidence(t, ctx, pool, grant))

			if _, err := pool.Exec(ctx, test.breakRead); err != nil {
				t.Fatal(err)
			}
			_, _, err = service.Submit(ctx, grant, input)
			if errors.Is(err, ErrAccessDenied) || errors.Is(err, ErrToolNotSelected) ||
				persistencepostgres.SQLState(err) != test.sqlState {
				t.Fatalf("%s lookup failure = %v, want the SQLSTATE %s cause", test.name, err, test.sqlState)
			}
			var receipts int
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM finding_proposal_receipts`).Scan(&receipts); err != nil || receipts != 0 {
				t.Fatalf("receipts after a failed lookup = (%d, %v), want none", receipts, err)
			}
			if _, err := pool.Exec(ctx, test.restoreRead); err != nil {
				t.Fatal(err)
			}

			receipt, replayed, err := service.Submit(ctx, grant, input)
			if err != nil || replayed || receipt.Origin.RunID != runID ||
				receipt.Origin.StageExecutionID != grant.StageExecutionID {
				t.Fatalf("submission after recovery = (%+v, %t, %v)", receipt, replayed, err)
			}
			if (auditOrigin == nil) != (receipt.Origin.Audit == nil) ||
				auditOrigin != nil && *receipt.Origin.Audit != *auditOrigin {
				t.Fatalf("Audit origin after recovery = %+v, want %+v", receipt.Origin.Audit, auditOrigin)
			}
		})
	}
}

// auditSubmissionGrant creates runID as an ordinary Audit-managed Run bound to a
// single accepted common Task, with a running StageExecution whose builder
// selects the finding tool, and returns the builder's allocation grant and the
// Audit origin intake must record for it.
func auditSubmissionGrant(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, ownerID, projectID, runID string, memberCounts ...int,
) (controlplane.AllocationGrant, *AuditOrigin) {
	return auditSubmissionGrantWithBaseline(t, ctx, pool, ownerID, projectID, runID, nil, memberCounts...)
}

func auditSubmissionGrantWithBaseline(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, ownerID, projectID, runID string, configureBaseline func(map[string]any), memberCounts ...int,
) (controlplane.AllocationGrant, *AuditOrigin) {
	return auditSubmissionGrantWithWorkflow(t, ctx, pool, ownerID, projectID, runID, configureBaseline, nil, memberCounts...)
}
func auditSubmissionGrantWithWorkflow(t *testing.T, ctx context.Context, pool *pgxpool.Pool, ownerID, projectID, runID string, configureBaseline func(map[string]any), configureWorkflow func(*resolved.ResolvedWorkflow), memberCounts ...int) (controlplane.AllocationGrant, *AuditOrigin) {
	t.Helper()
	snapshot, _, err := configload.LoadUnionReadOnly("../config/testdata/valid", filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	binding := stage.Agents["builder"]
	binding.Template.Toolsets = append(binding.Template.Toolsets, contracts.ToolsetSelection{
		Ref:   contracts.ToolsetRef{ToolsetID: "security-findings", Version: "1"},
		Tools: []string{"finding"},
	})
	stage.Agents["builder"] = binding
	workflow.Stages[workflow.EntryStage] = stage
	count := 1
	if len(memberCounts) > 0 {
		count = memberCounts[0]
	}
	if count > 1 {
		workflow.Inputs["context"] = resolved.ArtifactSlot{Required: true, MediaTypes: []string{"application/json"}}
		workflow.Outputs["result"] = resolved.ArtifactSlot{Required: true, MediaTypes: []string{"application/json"}}
		for key, result := range stage.Result.Artifacts {
			result.MediaTypes = []string{"application/json"}
			stage.Result.Artifacts[key] = result
		}
		workflow.Stages[workflow.EntryStage] = stage
	}
	if configureWorkflow != nil {
		configureWorkflow(&workflow)
	}
	workflowJSON, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	stageJSON, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}

	digest := func(value string) string { return auditdomain.DigestBytes([]byte(runID + "-" + value)) }
	data, err := os.ReadFile("../config/testdata/audit-common-catalog/audit-profiles/common_tasks.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct{ Spec json.RawMessage }
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	program, err := auditcontract.DecodeProgram(fixture.Spec)
	if err != nil {
		t.Fatal(err)
	}
	if count > 1 {
		binding := program.Workflows["inspect"]
		binding.Inputs["context"] = auditcontract.InputMapping{Source: "task-context"}
		binding.Outputs = map[string]string{"result": "result"}
		program.Workflows["inspect"] = binding
	}
	programBytes, _ := json.Marshal(program)
	profile, err := auditprofile.PinProgram(resolved.Selector{ID: "submission", Version: "1"}, programBytes, func(ref string) (resolved.ResolvedWorkflow, error) { return workflow, nil })
	if err != nil {
		t.Fatal(err)
	}
	pin, _ := json.Marshal(profile)
	limits := profile.Program.Limits
	audits := auditstoretest.NewStore(pool, artifactpgtest.NewStorage())
	audit, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: runID + "-audit", OwnerID: ownerID, ProjectID: projectID,
		Profile: auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest}, ProfileSnapshot: pin, InputSelection: json.RawMessage(`{}`),
		Limits:         auditstore.Limits{MaxRounds: limits.MaxIterations, BatchSize: limits.BatchSize, MaxItemsPerRound: limits.MaxTasksPerIteration, MaxItemsTotal: limits.MaxTasks, MaxSubmittedRuns: limits.MaxSubmittedRuns, MaxItemRunAttempts: limits.MaxAttempts, MaxEvidenceBytes: limits.MaxEvidenceBytes},
		IdempotencyKey: runID + "-audit", RequestDigest: digest("audit"),
	})
	if err != nil {
		t.Fatal(err)
	}
	artifactService := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	projectArtifacts, _ := artifactService.Project(projectID)
	sourceName := "source"
	if configureBaseline != nil {
		sourceName = runID + "-source"
	}
	source, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: sourceName}, artifacts.Payload{MediaType: "text/plain", Data: []byte("selected source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	sourceScope, _ := artifacts.ProjectScope(projectID)
	targetScope, _ := artifacts.AuditScope(audit.AuditID)
	retained, err := artifactService.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageAuditInput, Source: sourceScope, SourceRef: source.Ref, Target: targetScope, TargetRef: contracts.ArtifactRef{Namespace: "inputs", Name: "selected"}, Frozen: true})
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := artifactpgtest.NewStorage().Repository(pool).Metadata(ctx, targetScope, retained.TargetRef)
	if err != nil {
		t.Fatal(err)
	}
	baselineDocument := map[string]any{"schema": "contractor.audit.baseline.v1", "inputs": map[string]auditstore.ExactArtifact{"source": {Ref: source.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size}}, "skills": []any{}}
	if configureBaseline != nil {
		configureBaseline(baselineDocument)
	}
	baseline, _ := json.Marshal(baselineDocument)
	audit, _, err = audits.Activate(ctx, auditstore.ActivateParams{OwnerID: ownerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision, BaselineSnapshot: baseline, DeadlineAt: time.Now().Add(time.Hour), IdempotencyKey: runID + "-start", RequestDigest: digest("start")})
	if err != nil {
		t.Fatal(err)
	}
	material := auditcontract.Material{ScopeKind: "audit", ScopeID: audit.AuditID, Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size}
	var assertion *auditcontract.Hypothesis
	if count > 1 {
		h := auditcontract.Hypothesis{Schema: auditcontract.HypothesisSchema, ID: "system-selected-hypothesis", Revision: 1, Statement: "A pinned assertion in this fixture", Scope: map[string]string{}, Grounds: "Explicit fixture content", Origin: auditcontract.HypothesisOrigin{Kind: "manual", AuditID: audit.AuditID}}
		for revision := uint64(1); revision <= 7; revision++ {
			h.Revision = revision
			result, _, err := audits.WriteHypotheses(ctx, auditstore.WriteHypothesesParams{OwnerID: ownerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: fmt.Sprintf("hypothesis-%d", revision), RequestDigest: digest(fmt.Sprintf("hypothesis-%d", revision)), Writes: []auditstore.HypothesisWrite{{Document: h, SourceKey: "manual"}}})
			if err != nil {
				t.Fatal(err)
			}
			audit.Revision = result.AuditRevision
		}
		assertion = &h
	}
	taskID := runID + "-task"
	taskIDs := []string{}
	tasks := []auditstore.AcceptedTask{}
	for index := range count {
		id := taskID
		if index > 0 {
			id = fmt.Sprintf("%s-%d", taskID, index)
		}
		task := auditcontract.Task{Schema: auditcontract.TaskSchema, AuditID: audit.AuditID, TaskID: id, Workflow: "inspect", Inputs: map[string]auditcontract.Material{"source": material}, Parameters: map[string]string{}, Objective: "Inspect selected source"}
		if index > 0 {
			task.Hypothesis = &auditcontract.HypothesisRef{ID: "system-selected-hypothesis", Revision: 7}
		}
		taskIDs = append(taskIDs, id)
		var frozen *auditcontract.Hypothesis
		if index > 0 {
			frozen = assertion
		}
		tasks = append(tasks, auditstore.AcceptedTask{Hypothesis: frozen, Document: task, Source: auditstore.TaskSource{Kind: "manual", Key: id}})
	}
	_, _, err = audits.AcceptTasks(ctx, auditstore.AcceptTasksParams{OwnerID: ownerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: runID + "-accept", RequestDigest: digest("accept"), Tasks: tasks})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := audits.Claim(ctx, auditstore.ClaimParams{HolderID: runID + "-controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim: %+v %v", claims, err)
	}
	claim := claims[0]
	execution, _, err := audits.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claim, TaskIDs: taskIDs})
	if err != nil {
		t.Fatal(err)
	}
	project := projectID
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		runs := runstore.NewPostgresStore(tx, artifactpgtest.NewStorage())
		if _, err := runs.CreateAuditRun(ctx, runstore.CreateAuditRunParams{
			CreateRunParams: runstore.CreateRunParams{
				LaneClass: resolved.LaneClassModel,
				RunID:     runID, OwnerID: ownerID, ProjectID: &project,
				WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
				WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: workflowJSON,
				Parameters: map[string]string{}, MetadataLabels: runstore.RunMetadataLabels{},
				RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
			},
			AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey,
		}); err != nil {
			return err
		}
		if _, err := auditstoretest.NewStore(tx, artifactpgtest.NewStorage()).BindRun(ctx, auditstore.BindRunParams{
			Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID,
		}); err != nil {
			return err
		}
		if _, err := runs.TransitionRun(
			ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"},
		); err != nil {
			return err
		}
		_, err := runs.TransitionRun(
			ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"},
		)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	stageID, allocationID := runID+"-stage", runID+"-allocation"
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		LaneClass:        resolved.LaneClassModel,
		StageExecutionID: stageID, RunID: runID, StageName: workflow.EntryStage, Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{},
		},
	}); err != nil {
		t.Fatal(err)
	}
	runtimeAgentID := strings.Repeat("a", 64)
	if err := runs.RecordStageAllocation(ctx, runstore.StageAllocation{
		AllocationID: allocationID, StageExecutionID: stageID,
		LogicalAgentName: "builder", Namespace: binding.Namespace,
		AgentTemplateRef: binding.Template.Ref, WorkerRuntimeRef: binding.Template.Runtime,
		RuntimeAgentID: runtimeAgentID, RuntimeAgentInstanceID: "audit-instance",
		RuntimeAgentLabelRevision:         1,
		RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion,
		RuntimeConfiguration:              findingRuntimeConfiguration(),
		PerformanceCollectionPolicy:       reporting.PerformanceCollectionDisabled,
	}); err != nil {
		t.Fatal(err)
	}
	return controlplane.AllocationGrant{
			AllocationID: allocationID, RuntimeAgentID: runtimeAgentID,
			RuntimeInstanceID: "audit-instance", RunID: runID, StageExecutionID: stageID,
			LogicalAgentName: "builder", Namespace: binding.Namespace,
		}, &AuditOrigin{
			AuditID: audit.AuditID, ExecutionID: execution.ExecutionID, TaskID: taskID,
		}
}

func TestPostgresFindingBatchTaskOriginAndBoundedResultReferences(t *testing.T) {
	ctx := t.Context()
	pool := isolatedFindingPool(t, ctx)
	const owner, project, run = "batch-owner", "batch-project", "batch-run"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Batch origins", IdempotencyKey: "project", RequestDigest: auditdomain.DigestBytes([]byte("project"))}); err != nil {
		t.Fatal(err)
	}
	grant, firstOrigin := auditSubmissionGrant(t, ctx, pool, owner, project, run, 2)
	service, err := New(pool, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"", "item-63"} {
		input := testSubmission("denied-invocation", "denied", []contracts.ArtifactRef{})
		if key != "" {
			input.ItemKey = &key
		}
		if _, _, err := service.Submit(ctx, grant, input); !errors.Is(err, ErrAccessDenied) {
			t.Fatalf("unattributable batch key %q: %v", key, err)
		}
	}
	proof := fixtureEvidence(t, ctx, pool, grant)
	requests := []ImportRequest{}
	var selected Receipt
	for index := range auditcontract.MaxFindingReferences + 2 {
		key := auditcontract.ExecutionItemKey(0)
		if index == auditcontract.MaxFindingReferences+1 {
			key = auditcontract.ExecutionItemKey(1)
		}
		input := testSubmission(fmt.Sprintf("invocation-%03d", index), fmt.Sprintf("candidate-%03d", index), proof)
		input.ItemKey = &key
		receipt, replayed, err := service.Submit(ctx, grant, input)
		if err != nil || replayed || receipt.Origin.Audit == nil {
			t.Fatalf("submit %d: %+v %v %v", index, receipt, replayed, err)
		}
		if key == "item-00" {
			if receipt.Origin.Audit.TaskID != firstOrigin.TaskID || receipt.Origin.Audit.Hypothesis != nil {
				t.Fatalf("first Task origin: %+v", receipt.Origin.Audit)
			}
		} else {
			selected = receipt
			if receipt.Origin.Audit.TaskID != run+"-task-1" || receipt.Origin.Audit.Hypothesis == nil || receipt.Origin.Audit.Hypothesis.ID != "system-selected-hypothesis" || receipt.Origin.Audit.Hypothesis.Revision != 7 {
				t.Fatalf("selected Task hypothesis origin: %+v", receipt.Origin.Audit)
			}
		}
		requests = append(requests, ImportRequest{OwnerID: owner, AuditID: firstOrigin.AuditID, RunID: run, Proposal: receipt.Proposal.Ref})
	}
	if err := service.RetainAuditCollectionBatch(ctx, requests); err != nil {
		t.Fatal(err)
	}
	references, err := service.AuditExecutionReceiptReferences(ctx, owner, firstOrigin.AuditID, firstOrigin.ExecutionID, run)
	if err != nil || len(references) != 2 {
		t.Fatalf("source association: %+v %v", references, err)
	}
	first := references[firstOrigin.TaskID]
	second := references[run+"-task-1"]
	if len(first.IDs) != auditcontract.MaxFindingReferences || !first.Truncated || len(second.IDs) != 1 || second.IDs[0] != selected.ReceiptID || second.Truncated {
		t.Fatalf("bounded per-Task references: %+v", references)
	}
	source, err := service.GetProjectReceipt(ctx, owner, project, selected.ReceiptID)
	if err != nil || source.Origin.Audit == nil || source.Origin.Audit.TaskID != selected.Origin.Audit.TaskID || source.Origin.Audit.Hypothesis == nil || *source.Origin.Audit.Hypothesis != *selected.Origin.Audit.Hypothesis {
		t.Fatalf("retained source origin: %+v %v", source.Origin, err)
	}
}
