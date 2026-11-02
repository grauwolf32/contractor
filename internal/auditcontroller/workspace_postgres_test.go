//go:build integration

package auditcontroller

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func workspaceFixture(t *testing.T) (context.Context, *postgresControllerHarness, *Controller, auditstore.AcceptedPreparationOutput, auditstore.Execution) {
	t.Helper()
	ctx, h, c := preparationHarness(t, true)
	preparationStep(t, ctx, c)
	seed := preparationExecutions(t, ctx, h, 1)[0]
	finishPreparationRun(t, ctx, h, seed, runstore.RunSucceeded, false)
	preparationStep(t, ctx, c)
	preparationStep(t, ctx, c)
	output, err := h.audits.GetPreparationOutput(ctx, h.started.Audit.AuditID, "z-seed", "list")
	if err != nil {
		t.Fatal(err)
	}
	preparationStep(t, ctx, c)
	consumer := preparationExecutions(t, ctx, h, 2)[0]
	return ctx, h, c, output, consumer
}

func TestPostgresAuditWorkspace(t *testing.T) {
	t.Run("handoff", func(t *testing.T) {
		ctx, h, _, output, consumer := workspaceFixture(t)
		scope, _ := artifacts.AuditScope(h.started.Audit.AuditID)
		workspace, _ := h.artifacts.Scoped(scope)
		retained, err := workspace.Metadata(ctx, output.Output.Retained.Ref)
		if err != nil {
			t.Fatal(err)
		}
		producer, _ := h.artifacts.Run(output.RunID)
		original, err := producer.Metadata(ctx, output.Output.Source.Ref)
		if err != nil {
			t.Fatal(err)
		}
		child, _ := h.artifacts.Run(*consumer.RunID)
		copied, err := child.Metadata(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "previous"})
		if err != nil || original.ContentCacheKey != retained.ContentCacheKey || retained.ContentCacheKey != copied.ContentCacheKey {
			t.Fatalf("fork duplicated version: %+v %+v %+v %v", original, retained, copied, err)
		}
		link, err := h.audits.GetArtifactLink(ctx, h.started.Audit.AuditID, "prepare:"+output.ExecutionID+":list")
		if err != nil || link.Scope != artifacts.ScopeAudit {
			t.Fatalf("accepted scope: %+v %v", link, err)
		}
		if err := rundeletion.New(h.pool).DeleteReleasedTerminalRun(ctx, h.started.Audit.OwnerID, output.RunID); err != nil {
			t.Fatal(err)
		}
		read, err := child.ReadExact(ctx, copied.Exact())
		if err != nil || auditdomain.DigestBytes(read.Payload.Data) != output.Output.Retained.Digest {
			t.Fatalf("lost accepted bytes: %v", err)
		}
		lineage, err := workspace.ListLineage(ctx, retained.Ref, artifacts.LineagePageQuery{Limit: 2})
		if err != nil || !slices.ContainsFunc(lineage, func(edge artifacts.LineageEdge) bool {
			return edge.Kind == artifacts.LineageAuditOutput && edge.SourceScopeID == output.RunID
		}) {
			t.Fatalf("lost source provenance: %+v %v", lineage, err)
		}
	})
	t.Run("publication", func(t *testing.T) {
		ctx, h, c, output, consumer := workspaceFixture(t)
		audit := h.started.Audit
		request := auditservice.MaterialPublicationRequest{AuditID: audit.AuditID, Source: output.Output.Retained.Ref, Target: contracts.ArtifactRef{Namespace: "materials", Name: "selected"}}
		result, err := h.service.PublishMaterial(ctx, audit.OwnerID, audit.ProjectID, "publication", request, nil)
		if err != nil {
			t.Fatal(err)
		}
		scope, _ := artifacts.AuditScope(audit.AuditID)
		workspace, _ := h.artifacts.Scoped(scope)
		sourceMetadata, err := workspace.Metadata(ctx, request.Source)
		if err != nil {
			t.Fatal(err)
		}
		destination, _ := h.artifacts.Project(audit.ProjectID)
		publishedMetadata, err := destination.Metadata(ctx, result.Artifact)
		if err != nil || sourceMetadata.ContentCacheKey != publishedMetadata.ContentCacheKey {
			t.Fatalf("Project publication duplicated immutable version: %v", err)
		}

		finishPreparationRun(t, ctx, h, consumer, runstore.RunSucceeded, false)
		preparationStep(t, ctx, c)
		preparationStep(t, ctx, c)
		for _, execution := range preparationExecutions(t, ctx, h, 2) {
			if err := rundeletion.New(h.pool).DeleteReleasedTerminalRun(ctx, audit.OwnerID, *execution.RunID); err != nil {
				t.Fatal(err)
			}
		}
		current, err := h.audits.Get(ctx, audit.OwnerID, audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		_, err = h.service.Delete(ctx, auditservice.MutationParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: current.Revision, IdempotencyKey: "delete", RequestDigest: auditdomain.DigestBytes([]byte("delete"))})
		if err != nil {
			t.Fatal(err)
		}
		for n := 0; n < 50; n++ {
			_, err := h.audits.Get(ctx, audit.OwnerID, audit.AuditID)
			if errors.Is(err, auditstore.ErrNotFound) {
				break
			}
			if _, err := c.RunOnce(ctx); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := h.audits.Get(ctx, audit.OwnerID, audit.AuditID); !errors.Is(err, auditstore.ErrNotFound) {
			t.Fatalf("Audit not purged: %v", err)
		}
		replay, err := h.service.PublishMaterial(ctx, audit.OwnerID, audit.ProjectID, "publication", request, nil)
		if err != nil || !replay.Replayed || !replay.Artifact.SameExact(result.Artifact) {
			t.Fatalf("post-purge replay: %+v %v", replay, err)
		}
		project, _ := h.artifacts.Project(audit.ProjectID)
		if _, err := project.Read(ctx, result.Artifact); err != nil {
			t.Fatal(err)
		}
		lineage, err := project.ListLineage(ctx, result.Artifact, artifacts.LineagePageQuery{Limit: 2})
		if err != nil || len(lineage) != 1 || lineage[0].SourceScope != artifacts.ScopeAudit || lineage[0].SourceScopeID != audit.AuditID {
			t.Fatalf("Project lineage: %+v %v", lineage, err)
		}
	})
	t.Run("replay", func(t *testing.T) {
		ctx, h, _, output, _ := workspaceFixture(t)
		a := h.started.Audit
		request := auditservice.MaterialPublicationRequest{AuditID: a.AuditID, Source: output.Output.Retained.Ref, Target: contracts.ArtifactRef{Namespace: "materials", Name: "one"}}
		first, err := h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, "same", request, nil)
		if err != nil {
			t.Fatal(err)
		}
		second, err := h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, "same", request, nil)
		if err != nil || !second.Replayed || !first.Artifact.SameExact(second.Artifact) {
			t.Fatalf("replay: %+v %v", second, err)
		}
		request.Target.Name = "other"
		if _, err := h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, "same", request, nil); !errors.Is(err, contracts.ErrIdempotencyConflict) {
			t.Fatalf("changed replay: %v", err)
		}
	})
	t.Run("cas", func(t *testing.T) {
		ctx, h, _, output, _ := workspaceFixture(t)
		a := h.started.Audit
		request := auditservice.MaterialPublicationRequest{AuditID: a.AuditID, Source: output.Output.Retained.Ref, Target: contracts.ArtifactRef{Namespace: "materials", Name: "one"}}
		first, err := h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, "first", request, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, "collision", request, nil); !errors.Is(err, artifacts.ErrArtifactConflict) {
			t.Fatalf("create collision: %v", err)
		}
		second, err := h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, "replace", request, first.Artifact.Revision)
		if err != nil {
			t.Fatal(err)
		}
		if second.Artifact.SameExact(first.Artifact) {
			t.Fatal("CAS did not advance binding")
		}
		if _, err := h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, "stale", request, first.Artifact.Revision); !errors.Is(err, artifacts.ErrArtifactConflict) {
			t.Fatalf("stale CAS: %v", err)
		}
		project, _ := h.artifacts.Project(a.ProjectID)
		one, _ := project.Metadata(ctx, first.Artifact)
		two, _ := project.Metadata(ctx, second.Artifact)
		if one.ContentCacheKey != two.ContentCacheKey {
			t.Fatal("CAS duplicated immutable version")
		}
	})
	t.Run("authority", func(t *testing.T) {
		ctx, h, _, output, _ := workspaceFixture(t)
		a := h.started.Audit
		request := auditservice.MaterialPublicationRequest{AuditID: a.AuditID, Source: output.Output.Retained.Ref, Target: contracts.ArtifactRef{Namespace: "materials", Name: "one"}}
		inputs := map[string]contracts.ArtifactRef{}
		baseline, err := auditservice.DecodeBaseline(a.BaselineSnapshot)
		if err != nil {
			t.Fatal(err)
		}
		for slot, exact := range baseline.Inputs {
			inputs[slot] = exact.Ref
		}
		peer, _, err := h.service.CreateDraft(ctx, auditservice.CreateDraftParams{OwnerID: a.OwnerID, ProjectID: a.ProjectID, AuditID: "peer-audit",
			Profile: auditservice.ProfileSelector{Name: a.Profile.Name, Version: a.Profile.Version}, Inputs: inputs,
			Scope: auditservice.Scope{Objective: "Independent material authority"}, IdempotencyKey: "peer-audit", RequestDigest: auditdomain.DigestBytes([]byte("peer-audit"))})
		if err != nil {
			t.Fatal(err)
		}
		foreignSource, _ := artifacts.RunScope(output.RunID)
		foreignTarget, _ := artifacts.AuditScope(peer.AuditID)
		if _, err := h.artifacts.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageAuditOutput, Source: foreignSource, SourceRef: output.Output.Source.Ref,
			Target: foreignTarget, TargetRef: request.Target, Frozen: true}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
			t.Fatalf("matching Project granted cross-Audit output: %v", err)
		}
		peerRequest := request
		peerRequest.AuditID = peer.AuditID
		if _, err := h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, "peer", peerRequest, nil); !errors.Is(err, auditstore.ErrNotFound) {
			t.Fatalf("matching Project granted cross-Audit publication: %v", err)
		}
		if _, err := h.service.PublishMaterial(ctx, "foreign", a.ProjectID, "foreign", request, nil); err == nil {
			t.Fatal("foreign owner published")
		}
		request.AuditID = "missing-audit"
		if _, err := h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, "foreign-audit", request, nil); err == nil {
			t.Fatal("foreign Audit published")
		}
		request.AuditID = a.AuditID
		missing := "missing"
		request.Source.Revision = &missing
		if _, err := h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, "missing", request, nil); err == nil {
			t.Fatal("missing exact source published")
		}
	})
}

func TestPostgresAuditMaterialConcurrentSources(t *testing.T) {
	ctx, h, _, output, _ := workspaceFixture(t)
	a := h.started.Audit
	other, err := h.audits.GetPreparationOutput(ctx, a.AuditID, "z-seed", "diagnostics")
	if err != nil {
		t.Fatal(err)
	}
	sources := []contracts.ArtifactRef{output.Output.Retained.Ref, other.Output.Retained.Ref}
	results := make([]auditservice.MaterialPublication, 2)
	failures := make([]error, 2)
	start := make(chan struct{})
	var workers sync.WaitGroup
	for i := range sources {
		workers.Add(1)
		go func() {
			defer workers.Done()
			<-start
			results[i], failures[i] = h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, fmt.Sprintf("source-%d", i), auditservice.MaterialPublicationRequest{
				AuditID: a.AuditID, Source: sources[i], Target: contracts.ArtifactRef{Namespace: "materials", Name: "contended"}}, nil)
		}()
	}
	close(start)
	workers.Wait()
	winners := 0
	for i, err := range failures {
		if err == nil {
			winners++
			if !results[i].Source.SameExact(sources[i]) {
				t.Fatal("publication selected another producer")
			}
		} else if !errors.Is(err, artifacts.ErrArtifactConflict) {
			t.Fatalf("concurrent publication: %v", err)
		}
	}
	if winners != 1 {
		t.Fatalf("published %d sources into one create-only destination", winners)
	}
}

func TestPostgresAuditMaterialPublicationMaintainsHistoryAfterRollback(t *testing.T) {
	ctx, h, _, output, _ := workspaceFixture(t)
	a := h.started.Audit
	project, _ := h.artifacts.Project(a.ProjectID)
	target := contracts.ArtifactRef{Namespace: "materials", Name: "history"}
	var current *string
	for i := range 200 {
		written, err := project.Write(ctx, target, artifacts.Payload{MediaType: "application/json", Data: []byte(fmt.Sprintf(`{"n":%d}`, i))}, current)
		if err != nil {
			t.Fatal(err)
		}
		current = written.Ref.Revision
	}
	policy, err := artifacts.NewHistoryPolicy(20, 1<<30)
	if err != nil {
		t.Fatal(err)
	}
	ctx = artifacts.WithHistoryPolicy(ctx, policy)
	request := auditservice.MaterialPublicationRequest{AuditID: a.AuditID, Source: output.Output.Retained.Ref, Target: target}
	for range 2 {
		_, err := h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, "maintenance", request, current)
		var pending *artifacts.HistoryMaintenanceError
		if !errors.As(err, &pending) || !pending.Maintained || pending.Removed != 64 {
			t.Fatalf("bounded history maintenance did not progress: %v", err)
		}
	}
	result, err := h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, "maintenance", request, current)
	if err != nil {
		t.Fatal(err)
	}
	replay, err := h.service.PublishMaterial(ctx, a.OwnerID, a.ProjectID, "maintenance", request, current)
	if err != nil || !replay.Replayed || !replay.Artifact.SameExact(result.Artifact) {
		t.Fatalf("maintained publication replay: %+v %v", replay, err)
	}
}
