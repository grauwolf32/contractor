package projectsitemap

import (
	"bytes"
	"encoding/json"
	"errors"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresSitemapSnapshotRebindsReportEvidenceToAuthorizedOriginalInput(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	service := NewService(pool, artifactpgtest.NewStorage())
	stores := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	payload := []byte("opaque observation evidence\x00\xff")
	var report contracts.ArtifactRef
	for _, runID := range []string{"producer", "repeat-producer"} {
		createSitemapRun(t, ctx, pool, runID)
		published := writeObservationEvidence(t, ctx, pool, runID, "report", payload)
		access := collectionAccess(t, ctx, service, runID)
		if _, err := service.CollectAllocation(ctx, access, runID, preparedEvidenceBatch(published, "/users")); err != nil {
			t.Fatal(err)
		}
		if runID == "producer" {
			report = published.Ref
		}
	}
	createSitemapRun(t, ctx, pool, "consumer")
	// Use the ordinary output publication and input fork, rather than inserting
	// fabricated lineage or granting the snapshot reader access to another Run.
	output, err := artifactstest.BindOutput(ctx, stores, "producer", "report", report)
	if err != nil {
		t.Fatal(err)
	}
	if err := artifactstest.FreezeOutputs(ctx, stores, "producer"); err != nil {
		t.Fatal(err)
	}
	producerScope, _ := artifacts.RunScope("producer")
	projectScope, _ := artifacts.ProjectScope("project")
	publication, err := stores.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageProjectOutputPublish, Source: producerScope, SourceRef: output.TargetRef, SourceFrozen: true, Target: projectScope, TargetRef: contracts.ArtifactRef{Namespace: "reports", Name: "evidence"}})
	if err != nil {
		t.Fatal(err)
	}
	input, err := artifactstest.ForkProjectInput(ctx, stores, "project", publication.TargetRef, "consumer", "selected-report")
	if err != nil {
		t.Fatal(err)
	}
	run, _ := stores.Run("consumer")
	// A working-copy replacement cannot authorize its different bytes. Selection
	// must resolve the immutable original input, as for declarative sources.
	working, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "selected-report"}, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"changed":true}`)}, input.TargetRef.Revision)
	if err != nil {
		t.Fatal(err)
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Document.Groups[0].Members[0].Evidence) != 2 {
		t.Fatal("test requires two producer locators for identical content")
	}
	selection := SnapshotSelection{Revision: view.Revision, Groups: []string{view.Document.Groups[0].ID}, RequireReferencedInputs: true}
	ungranted := collectionAccess(t, ctx, service, "consumer")
	if _, err := service.SnapshotAllocation(ctx, ungranted, "no-input", selection); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("unselected report became authorized: %v", err)
	}
	optional := selection
	optional.RequireReferencedInputs = false
	savedOptional, err := service.SnapshotAllocation(ctx, ungranted, "optional", optional)
	if err != nil {
		t.Fatal(err)
	}
	optionalRead, err := run.ReadExact(ctx, savedOptional.Artifact)
	if err != nil {
		t.Fatal(err)
	}
	var unavailable snapshotDocument
	if err := json.Unmarshal(optionalRead.Payload.Data, &unavailable); err != nil {
		t.Fatal(err)
	}
	if len(unavailable.Inputs) != 0 || len(unavailable.Document.Groups[0].Members[0].Evidence) != 0 || !slices.ContainsFunc(unavailable.Document.Gaps, func(g Gap) bool { return g.Reason == "source-unavailable" }) {
		t.Fatal("optional snapshot fabricated a retained report")
	}
	// Purge both original producers before materializing; snapshot only needs
	// its explicitly selected local original fork and must never reread them.
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	for _, runID := range []string{"producer", "repeat-producer"} {
		for _, transition := range [][2]runstore.WorkflowRunState{{runstore.RunInitializing, runstore.RunPending}, {runstore.RunPending, runstore.RunRunning}, {runstore.RunRunning, runstore.RunSucceeded}} {
			if _, err := runs.TransitionRun(ctx, runID, transition[0], transition[1], runstore.Reason{Code: "fixture-complete"}); err != nil {
				t.Fatal(err)
			}
		}
		if err := rundeletion.New(pool, artifactpgtest.NewStorage()).DeleteReleasedTerminalRun(ctx, "owner", runID); err != nil {
			t.Fatal(err)
		}
	}
	producer, _ := stores.Run("producer")
	if _, err := producer.Read(ctx, report); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("source report survived Run purge: %v", err)
	}
	policy := readerPolicy()
	policy.SnapshotInputs = []string{"selected-report"}
	access, err := service.AllocationAccess(ctx, "consumer", "research", policy)
	if err != nil {
		t.Fatal(err)
	}
	saved, err := service.SnapshotAllocation(ctx, access, "selected-report", selection)
	if err != nil {
		t.Fatal(err)
	}
	body, err := run.ReadExact(ctx, saved.Artifact)
	if err != nil {
		t.Fatal(err)
	}
	var snapshot snapshotDocument
	if err := json.Unmarshal(body.Payload.Data, &snapshot); err != nil {
		t.Fatal(err)
	}
	member := snapshot.Document.Groups[0].Members[0]
	if len(snapshot.Inputs) != 1 || len(member.Evidence) != 1 || member.Evidence[0].RunID != "consumer" || !member.Evidence[0].Artifact.Ref.SameExact(snapshot.Inputs[0].Ref) || snapshot.Inputs[0].Ref.SameExact(input.TargetRef) || snapshot.Inputs[0].Ref.SameExact(working.Ref) {
		t.Fatalf("evidence was not rebound/deduplicated/frozen: %+v", snapshot)
	}
	if slices.ContainsFunc(snapshot.Document.Gaps, func(g Gap) bool { return g.Reason == "source-unavailable" }) {
		t.Fatal("rebinding duplicates manufactured unavailable evidence")
	}
	metadata, err := run.Metadata(ctx, snapshot.Inputs[0].Ref)
	if err != nil || !metadata.Frozen {
		t.Fatalf("snapshot report not frozen: %v", err)
	}
	retained, err := run.ReadExact(ctx, snapshot.Inputs[0])
	if err != nil || !bytes.Equal(retained.Payload.Data, payload) {
		t.Fatalf("snapshot lost original report bytes: %v", err)
	}
	limited, _ := artifactpostgres.NewHistoryPolicy(2, artifactpostgres.DefaultHistoryMaxBytes)
	limitedRun, _ := artifacts.NewService(artifactpgtest.NewConfigured(artifactpgtest.NewRuntime(), limited).Repository(pool)).Run(access.RunID)
	if _, err := limitedRun.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "selected-report"}, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"changed":2}`)}, working.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	if _, err := run.Read(ctx, working.Ref); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("working input retained implicitly: %v", err)
	}
	if _, err := run.ReadExact(ctx, snapshot.Inputs[0]); err != nil {
		t.Fatalf("snapshot report lost after input history eviction: %v", err)
	}
}
