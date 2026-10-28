package auditservice

import (
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/pentest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

// The baseline is created by ordinary Audit start, then consumed through the
// actual execution/Run/Stage ownership chain. Mutable inputs and Project
// policy cannot broaden the private task's still-empty target assignment.
func TestEngagementStartToPrivateAllocationPins(t *testing.T) {
	f := newEngagementFixture(t, 1)
	// Dispatch admission compares its deadline to the database clock. No test
	// assertion depends on elapsed time; the authorized timestamp is compared
	// only to the exact baseline returned by start.
	f.service.now = time.Now
	document := f.document("http://127.0.0.1")
	draft, err := f.create(t, "allocation-engagement", document)
	if err != nil {
		t.Fatal(err)
	}
	started, err := f.service.Start(f.ctx, f.start(draft))
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := DecodeBaseline(started.Audit.BaselineSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	profile, err := auditprofile.DecodeResolvedAuditProfileSnapshot(started.Audit.ProfileSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	store := auditstoretest.NewStore(f.pool)
	claim := claimAuditForPurge(t, f.ctx, store, draft.AuditID)
	roundID := *started.Audit.CurrentRoundID
	if _, err = store.TransitionRound(f.ctx, auditstore.RoundTransitionParams{Claim: claim, RoundID: roundID, ExpectedRevision: 1, ExpectedState: auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting}); err != nil {
		t.Fatal(err)
	}
	items, err := store.ListItemsPage(f.ctx, auditstore.ListItemsParams{OwnerID: f.project.OwnerID, AuditID: draft.AuditID, Limit: 2})
	if err != nil || len(items) != 1 {
		t.Fatalf("items=%+v %v", items, err)
	}
	execution, _, err := store.CreateExecutionIntent(f.ctx, auditstore.CreateExecutionIntentParams{Claim: claim, ExecutionID: "engagement-execution", RoundID: &roundID, Role: auditstore.ExecutionCheck, WorkflowRole: "check", LaneClass: resolved.LaneClassModel, Manifest: items[0].Task, SubmissionKey: "engagement-submission", RequestDigest: serviceTestDigest("engagement-submission"), Members: []auditstore.ExecutionMemberIntent{{ExecutionItemID: "engagement-member", ItemID: items[0].ItemID, ItemAttempt: 1, Task: items[0].Task, Inputs: []auditstore.ExactArtifact{}}}})
	if err != nil {
		t.Fatal(err)
	}
	workflow := profile.Workflows["check"].Workflow
	workflowBytes, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	const runID = "engagement-child-run"
	err = persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, createErr := runstore.NewPostgresStore(tx).CreateAuditRun(f.ctx, runstore.CreateAuditRunParams{CreateRunParams: runstore.CreateRunParams{RunID: runID, OwnerID: f.project.OwnerID, ProjectID: &f.project.ProjectID, WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: "contractor/v1alpha1", WorkflowSnapshot: workflowBytes, Parameters: map[string]string{}, RuntimeConfig: baseline.RuntimeConfig, LaneClass: resolved.LaneClassModel}, AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey})
		if createErr != nil {
			return createErr
		}
		_, bindErr := auditstoretest.NewStore(tx).BindRun(f.ctx, auditstore.BindRunParams{Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID})
		return bindErr
	})
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(f.pool)
	if _, err = runs.TransitionRun(f.ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	if _, err = runs.TransitionRun(f.ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	stageBytes, err := json.Marshal(workflow.Stages[workflow.EntryStage])
	if err != nil {
		t.Fatal(err)
	}
	stage, err := runs.CreateStageExecution(f.ctx, runstore.CreateStageExecutionParams{LaneClass: resolved.LaneClassModel, StageExecutionID: "engagement-stage", RunID: runID, StageName: workflow.EntryStage, Attempt: 1, StageSpecSchemaVersion: "contractor/v1alpha1", StageSpecSnapshot: stageBytes, StageContextSchemaVersion: "contractor/v1alpha1", StageContext: runstore.StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{}}})
	if err != nil {
		t.Fatal(err)
	}
	first, err := ResolveAllocationEngagement(f.ctx, f.pool, runID, stage.StageExecutionID)
	if err != nil || first == nil {
		t.Fatalf("private projection=%+v %v", first, err)
	}
	pin := baseline.Engagement
	if first.EngagementDigest != pin.Artifact.Digest || first.PolicyDigest != pin.PolicyDigest || first.AuthorizedBy != f.project.OwnerID || !first.AuthorizedAt.Equal(pin.AuthorizedAt) || first.Budgets != document.Budgets || !first.Engagement.SameExact(pin.Artifact.Ref) || len(first.Routes)+len(first.Exclusions)+len(first.Identities)+len(first.Inputs) != 0 {
		t.Fatalf("private projection drifted or exposed unassigned authority: %+v", first)
	}
	document.Target.Deployment = &pentest.Deployment{Marker: "new mutable document"}
	replacement, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.store.Write(f.ctx, contracts.ArtifactRef{Namespace: pin.Artifact.Ref.Namespace, Name: pin.Artifact.Ref.Name}, artifacts.Payload{MediaType: pentest.MediaType, Data: replacement}, pin.Artifact.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	prohibited := contracts.ActiveChecksProhibited
	if _, err = projectstore.NewPostgresStore(f.pool).Update(f.ctx, projectstore.UpdateParams{ProjectID: f.project.ProjectID, OwnerID: f.project.OwnerID, ExpectedRevision: f.project.Revision, Name: f.project.Name, ActiveChecks: &prohibited}); err != nil {
		t.Fatal(err)
	}
	second, err := ResolveAllocationEngagement(f.ctx, f.pool, runID, stage.StageExecutionID)
	if err != nil {
		t.Fatal(err)
	}
	firstBytes, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	secondBytes, err := json.Marshal(second)
	if err != nil {
		t.Fatal(err)
	}
	if string(firstBytes) != string(secondBytes) {
		t.Fatal("private metadata reread mutable Project or engagement")
	}
	if _, err = ResolveAllocationEngagement(f.ctx, f.pool, "foreign-run", stage.StageExecutionID); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("foreign Run reached baseline: %v", err)
	}
	if _, err = ResolveAllocationEngagement(f.ctx, f.pool, runID, "foreign-stage"); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("foreign Stage reached baseline: %v", err)
	}
}
