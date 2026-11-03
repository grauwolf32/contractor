//go:build integration

package findingintake

import (
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestProjectFindingRetainsExactHypothesisAfterAuditPurge(t *testing.T) {
	ctx := t.Context()
	pool := isolatedFindingPool(t, ctx)
	const owner, project, run = "hypothesis-owner", "hypothesis-project", "hypothesis-run"
	digest := auditdomain.DigestBytes
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Exact origin", IdempotencyKey: "project", RequestDigest: digest([]byte("project"))}); err != nil {
		t.Fatal(err)
	}
	grant, origin := auditSubmissionGrant(t, ctx, pool, owner, project, run, 2)
	service, err := New(pool, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	art := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	runArt, _ := art.Run(run)
	proof, err := runArt.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: "proof"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("Evidence of an observed problem under the selected conditions")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	key := auditcontract.ExecutionItemKey(1)
	input := testSubmission("hypothesis-invocation", "problem", []contracts.ArtifactRef{proof.Ref})
	input.ItemKey = &key
	input.PublishProblem = true
	receipt, _, err := service.Submit(ctx, grant, input)
	if err != nil || receipt.FindingID == nil || receipt.Origin.Audit == nil || receipt.Origin.Audit.HypothesisSnapshot == nil {
		t.Fatalf("publication=%+v error=%v", receipt, err)
	}
	for _, mutate := range []func(*Submission){func(s *Submission) { s.Incidental = true }, func(s *Submission) { k := auditcontract.ExecutionItemKey(0); s.ItemKey = &k }} {
		changed := input
		mutate(&changed)
		if _, _, err := service.Submit(ctx, grant, changed); !errors.Is(err, ErrConflict) {
			t.Fatalf("receipt rebound to changed membership: %v", err)
		}
	}
	incidental := input
	incidental.Incidental = true
	incidental.Proposal.ClientKey = "incidental"
	incidental.SubmissionID = stableSubmissionID(incidental.InvocationID, incidental.Proposal.ClientKey)
	other, _, err := service.Submit(ctx, grant, incidental)
	if err != nil || other.Origin.Audit == nil || other.Origin.Audit.TaskID != run+"-task-1" || other.Origin.Audit.Hypothesis != nil || other.Origin.Audit.HypothesisSnapshot != nil {
		t.Fatalf("incidental=%+v error=%v", other, err)
	}
	audits := auditstoretest.NewStore(pool, artifactpgtest.NewStorage()).WithPurgeDependencies(DeleteAuditHoldsBatch)
	claim := auditstore.ControllerClaim{AuditID: origin.AuditID, HolderID: run + "-controller", Epoch: 1}
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	if err := runs.MarkStageAllocationReleased(ctx, grant.AllocationID); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, run, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "fixture_finished"}); err != nil {
		t.Fatal(err)
	}
	cursor, err := runs.GetRunEventCursor(ctx, run)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := audits.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: claim, ExecutionID: origin.ExecutionID, RunID: run, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)}); err != nil {
		t.Fatal(err)
	}
	tasks, err := audits.ListExecutionTasks(ctx, origin.AuditID, origin.ExecutionID)
	if err != nil {
		t.Fatal(err)
	}
	collected := auditstore.CollectParams{Claim: claim, ExecutionID: origin.ExecutionID, ReceiptID: "hypothesis-collection", RequestDigest: digest([]byte("collection")), Disposition: auditstore.CollectionExecutionFailed}
	for _, task := range tasks {
		gap := "execution-failed"
		collected.Tasks = append(collected.Tasks, auditstore.TaskCollection{ExecutionItemID: auditdomain.DeterministicID("attempt", origin.ExecutionID, task.Document.TaskID), TaskID: task.Document.TaskID, GapCode: &gap})
	}
	if _, _, err := audits.Collect(ctx, collected); err != nil {
		t.Fatal(err)
	}
	current, err := audits.Get(ctx, owner, origin.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	deleting, _, err := audits.RequestDelete(ctx, auditstore.DeleteParams{OwnerID: owner, AuditID: origin.AuditID, ExpectedRevision: current.Revision, IdempotencyKey: "delete", RequestDigest: digest([]byte("delete"))})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := audits.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{Claim: claim, ExpectedRevision: deleting.Revision, ExpectedState: auditstore.AuditCancelling, TargetState: auditstore.AuditDeleting, Reason: deleting.StopReason}); err != nil {
		t.Fatal(err)
	}
	if err := deleteRun(ctx, pool, owner, run); err != nil {
		t.Fatal(err)
	}
	if _, _, err := audits.ReleaseDispatchHold(ctx, claim); err != nil {
		t.Fatal(err)
	}
	if err := audits.RenewClaim(ctx, claim, time.Minute); err != nil {
		t.Fatal(err)
	}
	for pass := 0; pass < 20; pass++ {
		if err := audits.PurgeClaimed(ctx, claim, auditdomain.ArtifactNamespace(origin.AuditID)); err != nil {
			t.Fatal(err)
		}
		if _, err := audits.Get(ctx, owner, origin.AuditID); errors.Is(err, auditstore.ErrNotFound) {
			break
		}
		if pass == 19 {
			t.Fatal("purge did not finish")
		}
	}
	source, err := service.GetProjectReceipt(ctx, owner, project, receipt.ReceiptID)
	if err != nil || source.Origin.Audit == nil || source.Origin.Audit.HypothesisSnapshot == nil || source.Origin.Audit.Hypothesis.Revision != 7 {
		t.Fatalf("retained source=%+v error=%v", source, err)
	}
	scope, _ := art.Project(project)
	read, err := scope.Read(ctx, source.Origin.Audit.HypothesisSnapshot.Ref)
	if err != nil {
		t.Fatal(err)
	}
	var h auditcontract.Hypothesis
	if json.Unmarshal(read.Payload.Data, &h) != nil || h.Revision != 7 || h.ID != "system-selected-hypothesis" || h.Statement != "A pinned assertion in this fixture" {
		t.Fatalf("lost exact assertion: %s", read.Payload.Data)
	}
	for _, exact := range source.ProjectEvidence {
		if _, err := scope.Read(ctx, exact.Ref); err != nil {
			t.Fatal(err)
		}
	}
}
