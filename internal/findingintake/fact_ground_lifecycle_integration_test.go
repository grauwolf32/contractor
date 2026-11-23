//go:build integration

package findingintake

import (
	"bytes"
	"context"
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
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresMemoryFactGroundsSurviveBothAuditLifecycles(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	storage := artifactpgtest.NewStorage()
	const owner, project = "fact-ground-owner", "fact-ground-project"
	digest := auditdomain.DigestBytes
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{OwnerID: owner, ProjectID: project, Kind: projectstore.KindProject, Name: "Exact fact grounds", IdempotencyKey: "project", RequestDigest: digest([]byte("project"))}); err != nil {
		t.Fatal(err)
	}
	// The empty callback selects the existing fixture's per-Run baseline identity.
	producer, originA := auditSubmissionGrantWithBaseline(t, ctx, pool, owner, project, "fact-ground-a", func(map[string]any) {})
	consumer, originB := auditSubmissionGrantWithBaseline(t, ctx, pool, owner, project, "fact-ground-b", func(map[string]any) {})
	art := artifacts.NewService(storage.Repository(pool))
	runA, _ := art.Run(producer.RunID)
	original, err := strictjson.Canonical(map[string]any{"schemaVersion": "contractor.memory-snapshot/v1", "namespace": "research", "notes": []any{map[string]any{"name": "service_banner", "content": "nginx reported; version and access boundary not independently checked", "description": "Reported technology", "tags": []string{"reported"}}}})
	if err != nil {
		t.Fatal(err)
	}
	proof, err := runA.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: "original-memory"}, artifacts.Payload{MediaType: "application/json", Data: original}, nil)
	if err != nil {
		t.Fatal(err)
	}
	intake, err := New(pool, storage)
	if err != nil {
		t.Fatal(err)
	}
	input := testSubmission("fact-ground-invocation", "reported-technology", []contracts.ArtifactRef{proof.Ref})
	input.Proposal.SeveritySuggestion = "informational"
	input.Proposal.Title = "Reported nginx"
	input.Proposal.Description = "A memory reports this technology; no independent probe is claimed."
	input.Incidental = true
	receipt, _, err := intake.Submit(ctx, producer, input)
	if err != nil || receipt.FindingID == nil || receipt.Origin.Audit == nil || receipt.Origin.Audit.AuditID != originA.AuditID || receipt.Origin.Audit.Hypothesis != nil {
		t.Fatal(receipt, err)
	}

	consumer.ReadPolicy = controlplane.ReadCurrentRun
	consumer.WritePolicy = controlplane.WriteInputsAndIntermediates
	consumer.FindingTools = []string{"select_project_findings"}
	selector, err := NewCollectionPublisher(pool, projectCollectionReads{}, storage)
	if err != nil {
		t.Fatal(err)
	}
	selected, err := selector.SelectAllocationFindings(ctx, consumer, FindingSelection{ClientKey: "fact-ground-selection", ReceiptIDs: []string{receipt.ReceiptID}, Findings: []CollectionFindingSelection{}})
	if err != nil || selected.EntryCount != 1 {
		t.Fatal(selected, err)
	}
	runB, _ := art.Run(consumer.RunID)
	selection, err := runB.Read(ctx, selected.Artifact.Ref)
	if err != nil {
		t.Fatal(err)
	}
	collection, pkg, err := auditdomain.DecodeFindingCollectionPackage(selection.Payload.Data)
	if err != nil || len(collection.Entries) != 1 {
		t.Fatal(collection, err)
	}
	entry := collection.Entries[0]
	if entry.Finding == nil || entry.Finding.FindingID != *receipt.FindingID || entry.ReceiptID != receipt.ReceiptID || len(entry.Evidence) != 1 {
		t.Fatal("selected ground does not identify exactly the chosen fact source")
	}
	evidence, ok := pkg.MemberByID(entry.Evidence[0].DocumentID)
	if !ok || !bytes.Equal(evidence.Data(), original) {
		t.Fatal("selection changed the original memory")
	}

	sourceScope, _ := artifacts.RunScope(consumer.RunID)
	targetScope, _ := artifacts.AuditScope(originB.AuditID)
	retained, err := art.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageAuditOutput, Source: sourceScope, SourceRef: selected.Artifact.Ref, Target: targetScope, TargetRef: contracts.ArtifactRef{Namespace: "grounds", Name: "selected-fact"}, Frozen: true})
	if err != nil {
		t.Fatal(err)
	}
	auditBArt, _ := art.Scoped(targetScope)
	metadata, err := auditBArt.Metadata(ctx, retained.TargetRef)
	if err != nil {
		t.Fatal(err)
	}
	material := auditcontract.Material{ScopeKind: "audit", ScopeID: originB.AuditID, Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size}
	audits := auditstoretest.NewStore(pool, storage).WithPurgeDependencies(DeleteAuditHoldsBatch)
	current, err := audits.Get(ctx, owner, originB.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	h := auditcontract.Hypothesis{Schema: auditcontract.HypothesisSchema, ID: "hypothesis-from-fact", Revision: 1, Statement: "The reported deployment exposes the selected access boundary", Scope: map[string]string{"conditions": "selected baseline only"}, Grounds: "The selected reported fact motivates a check; it is not a test result.", Sources: []auditcontract.Material{material}, Origin: auditcontract.HypothesisOrigin{Kind: "manual", AuditID: originB.AuditID}}
	if _, _, err := audits.WriteHypotheses(ctx, auditstore.WriteHypothesesParams{OwnerID: owner, AuditID: originB.AuditID, ExpectedRevision: current.Revision, IdempotencyKey: "grounded-hypothesis", RequestDigest: digest([]byte("grounded-hypothesis")), Writes: []auditstore.HypothesisWrite{{Document: h, SourceKey: "selected-fact"}}}); err != nil {
		t.Fatal(err)
	}
	// A later mutable source version cannot replace the selected memory or ZIP.
	if _, err := runA.Write(ctx, contracts.ArtifactRef{Namespace: proof.Ref.Namespace, Name: proof.Ref.Name}, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"later":"different bytes"}`)}, proof.Ref.Revision); err != nil {
		t.Fatal(err)
	}

	finishAndPurge := func(grant controlplane.AllocationGrant, origin *AuditOrigin) {
		t.Helper()
		claim := auditstore.ControllerClaim{AuditID: origin.AuditID, HolderID: grant.RunID + "-controller", Epoch: 1}
		runs := runstore.NewPostgresStore(pool, storage)
		if err := runs.MarkStageAllocationReleased(ctx, grant.AllocationID); err != nil {
			t.Fatal(err)
		}
		if _, err := runs.TransitionRun(ctx, grant.RunID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "fixture_finished"}); err != nil {
			t.Fatal(err)
		}
		cursor, err := runs.GetRunEventCursor(ctx, grant.RunID)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := audits.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: claim, ExecutionID: origin.ExecutionID, RunID: grant.RunID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)}); err != nil {
			t.Fatal(err)
		}
		tasks, err := audits.ListExecutionTasks(ctx, origin.AuditID, origin.ExecutionID)
		if err != nil {
			t.Fatal(err)
		}
		collected := auditstore.CollectParams{Claim: claim, ExecutionID: origin.ExecutionID, ReceiptID: grant.RunID + "-collection", RequestDigest: digest([]byte(grant.RunID + "-collection")), Disposition: auditstore.CollectionExecutionFailed}
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
		deleting, _, err := audits.RequestDelete(ctx, auditstore.DeleteParams{OwnerID: owner, AuditID: origin.AuditID, ExpectedRevision: current.Revision, IdempotencyKey: "delete-" + grant.RunID, RequestDigest: digest([]byte("delete-" + grant.RunID))})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := audits.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{Claim: claim, ExpectedRevision: deleting.Revision, ExpectedState: auditstore.AuditCancelling, TargetState: auditstore.AuditDeleting, Reason: deleting.StopReason}); err != nil {
			t.Fatal(err)
		}
		if err := deleteRun(ctx, pool, owner, grant.RunID); err != nil {
			t.Fatal(err)
		}
		if _, _, err := audits.ReleaseDispatchHold(ctx, claim); err != nil {
			t.Fatal(err)
		}
		if err := audits.RenewClaim(ctx, claim, time.Minute); err != nil {
			t.Fatal(err)
		}
		for range 20 {
			if err := audits.PurgeClaimed(ctx, claim, auditdomain.ArtifactNamespace(origin.AuditID)); err != nil {
				t.Fatal(err)
			}
			if _, err := audits.Get(ctx, owner, origin.AuditID); errors.Is(err, auditstore.ErrNotFound) {
				return
			}
		}
		t.Fatal("Audit purge did not settle")
	}
	finishAndPurge(producer, originA)
	saved, err := audits.GetHypothesis(ctx, owner, originB.AuditID, h.ID, h.Revision)
	if err != nil || len(saved.Document.Sources) != 1 || !saved.Document.Sources[0].Ref.SameExact(material.Ref) {
		t.Fatal("Audit B lost exact grounds after Audit A purge", saved, err)
	}
	read, err := auditBArt.Read(ctx, material.Ref)
	if err != nil || !bytes.Equal(read.Payload.Data, selection.Payload.Data) {
		t.Fatal("selected grounds changed", err)
	}
	finishAndPurge(consumer, originB)
	if _, err := audits.GetHypothesis(ctx, owner, originB.AuditID, h.ID, h.Revision); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatal("Audit B hypothesis registry survived purge", err)
	}
	if _, err := auditBArt.Read(ctx, material.Ref); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatal("Audit B grounds survived purge", err)
	}
	projectSource, err := intake.GetProjectReceipt(ctx, owner, project, receipt.ReceiptID)
	if err != nil || !projectSource.Origin.RunDeleted || projectSource.FindingID == nil || *projectSource.FindingID != *receipt.FindingID {
		t.Fatal(projectSource, err)
	}
	projectArt, _ := art.Project(project)
	read, err = projectArt.Read(ctx, projectSource.ProjectEvidence[0].Ref)
	if err != nil || !bytes.Equal(read.Payload.Data, original) {
		t.Fatal("Project lost the fact's original memory after both Audits were deleted", err)
	}
}
