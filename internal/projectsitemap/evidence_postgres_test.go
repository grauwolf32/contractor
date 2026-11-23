package projectsitemap

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

const testObservationKind = "custom-scanner.v2"

func writeObservationEvidence(t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID, name string, data []byte) artifacts.WriteResult {
	t.Helper()
	run, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Run(runID)
	if err != nil {
		t.Fatal(err)
	}
	written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "research", Name: name}, artifacts.Payload{MediaType: "application/octet-stream", Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return written
}

func collectionAccess(t *testing.T, ctx context.Context, service *Service, runID string) AllocationAccess {
	t.Helper()
	access, err := service.AllocationAccess(ctx, runID, "research", readerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	return access
}

func preparedEvidenceBatch(report artifacts.WriteResult, path string) CollectionIngress {
	observed := observation(path)
	observed.At = report.RevisionCreatedAt.UTC().Format(time.RFC3339Nano)
	ref := report.Ref.Clone()
	return CollectionIngress{Kind: testObservationKind, Observations: []Observation{observed}, Artifact: &ref}
}

func TestPostgresSitemapPreparedBatchReplaysAfterEvidenceAndMapEviction(t *testing.T) {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	counter := &allocationReadCounter{}
	pool := sitemapPool(t, counter)
	history, _ := artifactpostgres.NewHistoryPolicy(2, artifactpostgres.DefaultHistoryMaxBytes)
	ctx := t.Context()
	artifactHistory = history
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	createSitemapRun(t, ctx, pool, "producer")
	service := NewService(pool, artifactStorage)
	access := collectionAccess(t, ctx, service, "producer") // Handler owns explicit-vs-automatic grants.
	// Unknown source format, non-JSON bytes, and an artifact larger than scanner
	// report bounds prove the Server does not own provider preprocessing.
	report := writeObservationEvidence(t, ctx, pool, "producer", "evidence", bytes.Repeat([]byte{0xff}, 1<<20))
	batch := preparedEvidenceBatch(report, "/users?expand=profile")
	counter.reset()
	first, err := service.CollectAllocation(ctx, access, "first", batch)
	if err != nil {
		t.Fatal(err)
	}
	var contentReads, metadataReads int
	for _, query := range counter.reset() {
		if strings.Contains(query, "blob.payload") && strings.Contains(query, "SELECT revision.revision") {
			contentReads++
		}
		if strings.Contains(query, "'sha256:' || encode(blob.sha256") {
			metadataReads++
		}
	}
	if contentReads != 1 || metadataReads != 1 {
		t.Fatalf("expected one map read and one evidence metadata read, got %d/%d", contentReads, metadataReads)
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Document.Groups) != 1 || len(view.Document.Sources) != 1 {
		t.Fatalf("bad publication: %+v", view.Document)
	}
	source := view.Document.Sources[0]
	member := view.Document.Groups[0].Members[0]
	if source.Artifact != nil || source.Provenance.SourceID != key(testObservationKind, "") || len(member.Summaries) != 1 || len(member.Evidence) != 1 {
		t.Fatalf("wrong source/evidence: %+v %+v", source, member)
	}
	summary := member.Summaries[0]
	if summary.First != batch.Observations[0].At || summary.First != summary.Last || summary.Status != 200 {
		t.Fatalf("prepared timestamp changed: %+v", summary)
	}
	if member.Evidence[0].RunID != "producer" || !member.Evidence[0].Artifact.Ref.SameExact(report.Ref) || member.Evidence[0].Artifact.MediaType != "application/octet-stream" {
		t.Fatal("missing exact evidence metadata")
	}
	run, _ := artifacts.NewService(artifactStorage.Repository(pool)).Run("producer")
	latest := report
	for index := range 4 {
		latest, err = run.Write(ctx, contracts.ArtifactRef{Namespace: "research", Name: "evidence"}, artifacts.Payload{MediaType: "application/octet-stream", Data: []byte(fmt.Sprintf("opaque %d", index))}, latest.Ref.Revision)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = service.CollectAllocation(ctx, access, fmt.Sprintf("advance-%d", index), CollectionIngress{Kind: "http", Observations: []Observation{observation(fmt.Sprintf("/advance/%d", index))}}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err = run.Read(ctx, report.Ref); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("evidence held implicitly: %v", err)
	}
	if _, err = readSitemap(service, ctx, "project", *first.Ref.Revision); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("map held implicitly: %v", err)
	}
	before, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	counter.reset()
	replay, err := service.CollectAllocation(ctx, access, "first", batch)
	if err != nil || !replay.Replayed || !replay.Ref.SameExact(first.Ref) {
		t.Fatalf("failed replay after eviction: %+v %v", replay, err)
	}
	for _, query := range counter.reset() {
		if strings.Contains(query, "artifact_") {
			t.Fatalf("replay read map/evidence or rewrote artifact: %s", query)
		}
	}
	after, err := readSitemap(service, ctx, "project", "")
	if err != nil || after.Revision != before.Revision {
		t.Fatal("replay republished map")
	}
	if _, err = service.CollectAllocation(ctx, access, "unavailable", batch); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("new batch followed latest evidence: %v", err)
	}
	changed := batch
	changed.Artifact = &latest.Ref
	if _, err = service.CollectAllocation(ctx, access, "first", changed); !errors.Is(err, contracts.ErrIdempotencyConflict) {
		t.Fatalf("changed exact evidence reused receipt: %v", err)
	}
	changed = batch
	changed.Gaps = []CollectionGap{{Reason: "projection", Count: 1}}
	if _, err = service.CollectAllocation(ctx, access, "first", changed); !errors.Is(err, contracts.ErrIdempotencyConflict) {
		t.Fatalf("changed gaps reused receipt: %v", err)
	}
}

func TestPostgresSitemapObservationsAggregateAcrossRunsAndBoundEvidence(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	service := NewService(pool, artifactpgtest.NewStorage())
	var firstTime, lastTime string
	var access AllocationAccess
	for index := range 32 {
		runID := fmt.Sprintf("producer-%d", index)
		createSitemapRun(t, ctx, pool, runID)
		access = collectionAccess(t, ctx, service, runID)
		report := writeObservationEvidence(t, ctx, pool, runID, "evidence", []byte("opaque evidence"))
		batch := preparedEvidenceBatch(report, "/users")
		at := batch.Observations[0].At
		if index == 0 {
			firstTime = at
		}
		lastTime = at
		if _, err := service.CollectAllocation(ctx, access, runID, batch); err != nil {
			t.Fatal(err)
		}
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	member := view.Document.Groups[0].Members[0]
	if len(view.Document.Sources) != 1 || len(view.Document.Groups) != 1 || len(member.Summaries) != 1 || len(member.Evidence) != MaxProvenance || member.Omitted != 0 {
		t.Fatalf("repeat Runs consumed behavior capacity: %+v", view.Document)
	}
	if member.Summaries[0].First != firstTime || member.Summaries[0].Last != lastTime {
		t.Fatal("prepared time range lost")
	}
	if !slices.ContainsFunc(view.Document.Gaps, func(g Gap) bool { return g.Reason == "evidence" && g.Count == 24 }) {
		t.Fatalf("evidence overflow not factual: %+v", view.Document.Gaps)
	}
	report := writeObservationEvidence(t, ctx, pool, access.RunID, "new-behavior", []byte("new evidence"))
	batch := preparedEvidenceBatch(report, "/users")
	batch.Observations[0].Status = 500
	if _, err := service.CollectAllocation(ctx, access, "new-behavior", batch); err != nil {
		t.Fatal(err)
	}
	other := batch
	other.Kind = "another-provider"
	if _, err := service.CollectAllocation(ctx, access, "other-source", other); err != nil {
		t.Fatal(err)
	}
	deployment := "staging"
	access.Policy.Deployment = &deployment
	if _, err := service.CollectAllocation(ctx, access, "staging", batch); err != nil {
		t.Fatal(err)
	}
	view, err = readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Document.Sources) != 3 || len(view.Document.Groups[0].Members[0].Summaries) != 4 {
		t.Fatalf("new behavior/source/deployment discarded: %+v", view.Document)
	}
}

func TestPostgresSitemapObservationBatchBoundaryAndRollback(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	createSitemapRun(t, ctx, pool, "producer")
	service := NewService(pool, artifactpgtest.NewStorage())
	access := collectionAccess(t, ctx, service, "producer")
	maximum := CollectionIngress{Kind: testObservationKind, Observations: []Observation{}}
	for index := range 100 {
		maximum.Observations = append(maximum.Observations, observation(fmt.Sprintf("/item/%03d", index)))
	}
	tooMany := maximum
	tooMany.Observations = append(slices.Clone(maximum.Observations), observation("/overflow"))
	oversized := CollectionIngress{Kind: testObservationKind, Observations: []Observation{}}
	for index := range 100 {
		oversized.Observations = append(oversized.Observations, observation("/"+strings.Repeat("x", 3000)+fmt.Sprint(index)))
	}
	invalid := CollectionIngress{Kind: testObservationKind, Observations: []Observation{observation("/valid"), observation("/invalid")}}
	invalid.Observations[1].Status = 0
	for _, batch := range []CollectionIngress{tooMany, oversized, invalid} {
		if _, err := service.CollectAllocation(ctx, access, "bounded", batch); !errors.Is(err, ErrInvalid) {
			t.Fatalf("invalid batch accepted: %v", err)
		}
		project, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Project("project")
		if _, err := project.Read(ctx, mapRef()); !errors.Is(err, artifacts.ErrArtifactNotFound) {
			t.Fatalf("invalid batch wrote map: %v", err)
		}
	}
	accepted, err := service.CollectAllocation(ctx, access, "bounded", maximum)
	if err != nil {
		t.Fatalf("failed batch consumed receipt: %v", err)
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Document.Groups) != 100 || !accepted.Changed {
		t.Fatalf("100-row boundary lost: %d groups", len(view.Document.Groups))
	}
	partial := CollectionIngress{Kind: testObservationKind, Observations: []Observation{}, Gaps: []CollectionGap{{Reason: "request-errors", Count: 2}, {Reason: "projection", Count: 3}}}
	if _, err := service.CollectAllocation(ctx, access, "partial", partial); err != nil {
		t.Fatal(err)
	}
	// Gap order is not semantic and must not create a new digest.
	slices.Reverse(partial.Gaps)
	replay, err := service.CollectAllocation(ctx, access, "partial", partial)
	if err != nil || !replay.Replayed {
		t.Fatalf("gap order changed receipt intent: %v", err)
	}
	view, err = readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.ContainsFunc(view.Document.Gaps, func(g Gap) bool { return g.Reason == "request-errors" && g.Count == 2 }) {
		t.Fatal("gap-only batch lost factual omission")
	}
	empty := CollectionIngress{Kind: "empty-source", Observations: []Observation{}}
	saved, err := service.CollectAllocation(ctx, access, "empty", empty)
	if err != nil {
		t.Fatal(err)
	}
	replay, err = service.CollectAllocation(ctx, access, "empty", empty)
	if err != nil || !replay.Replayed || !replay.Ref.SameExact(saved.Ref) {
		t.Fatal("empty batch did not retain its receipt")
	}
	view, err = readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Document.Groups) != 100 || !slices.ContainsFunc(view.Document.Sources, func(source Source) bool { return source.Provenance.Kind == "empty-source" }) {
		t.Fatal("empty batch invented observations or lost source")
	}
}

func TestPostgresSitemapObservationEvidenceVisibilityAndProjectLifecycle(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	service := NewService(pool, artifactpgtest.NewStorage())
	createSitemapRun(t, ctx, pool, "source")
	createSitemapRun(t, ctx, pool, "other")
	access := collectionAccess(t, ctx, service, "source")
	report := writeObservationEvidence(t, ctx, pool, "source", "evidence", []byte("opaque evidence"))
	batch := preparedEvidenceBatch(report, "/users")
	other := collectionAccess(t, ctx, service, "other")
	if _, err := service.CollectAllocation(ctx, other, "foreign-run", batch); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("cross-Run exact ref readable: %v", err)
	}
	foreign := access
	foreign.OwnerID = "other"
	if _, err := service.CollectAllocation(ctx, foreign, "foreign-owner", batch); !errors.Is(err, projectstore.ErrNotFound) {
		t.Fatalf("foreign owner admitted: %v", err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := service.CollectAllocation(cancelled, access, "cancelled", batch); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled batch admitted: %v", err)
	}
	if _, err := service.CollectAllocation(ctx, access, "first", batch); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE projects SET lifecycle_state='deleting',deletion_phase='cancelling',deletion_requested_at=clock_timestamp(),revision=revision+1,updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond') WHERE project_id='project'`); err != nil {
		t.Fatal(err)
	}
	if _, err := service.CollectAllocation(ctx, access, "first", batch); !errors.Is(err, projectstore.ErrDeleting) {
		t.Fatalf("receipt bypassed Project deletion fence: %v", err)
	}
}

func TestPostgresSitemapNormalizedObservationBatchRejectsInvalidRows(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	service := NewService(pool, artifactpgtest.NewStorage())
	createSitemapRun(t, ctx, pool, "producer")
	access := collectionAccess(t, ctx, service, "producer")
	valid := observation("/valid")
	for index, change := range []func(*Observation){
		func(o *Observation) { o.At = "2026-10-29T15:00:00+03:00" },
		func(o *Observation) { o.Parameters = []string{"invalid\nname"} },
		func(o *Observation) { o.Method = "get" },
		func(o *Observation) { o.Outcome = "response"; o.Status = 0 },
		func(o *Observation) { o.URL = "file:///tmp/not-http" },
	} {
		bad := observation("/bad")
		change(&bad)
		for _, kind := range []string{"http", "caido", testObservationKind} {
			if _, err := service.CollectAllocation(ctx, access, fmt.Sprintf("invalid-%d-%s", index, kind), CollectionIngress{Kind: kind, Observations: []Observation{valid, bad}}); !errors.Is(err, ErrInvalid) {
				t.Fatalf("source %s repaired malformed normalized row: %v", kind, err)
			}
		}
	}
	project, _ := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Project("project")
	if _, err := project.Read(ctx, mapRef()); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatal("invalid normalized batch partially wrote map")
	}
	// The Runtime may prepare a valid remaining row and aggregate its omissions.
	batch := CollectionIngress{Kind: "caido", Observations: []Observation{valid}, Gaps: []CollectionGap{{Reason: "projection", Count: 5}}}
	if _, err := service.CollectAllocation(ctx, access, "prepared", batch); err != nil {
		t.Fatal(err)
	}
	changed := batch
	changed.Gaps = []CollectionGap{{Reason: "projection", Count: 6}}
	if _, err := service.CollectAllocation(ctx, access, "prepared", changed); !errors.Is(err, contracts.ErrIdempotencyConflict) {
		t.Fatalf("changed omissions reused receipt: %v", err)
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Document.Groups) != 1 || len(view.Document.Groups[0].Members[0].Evidence) != 0 || len(view.Document.Gaps) != 1 || view.Document.Gaps[0].Count != 5 {
		t.Fatalf("prepared metadata changed: %+v", view.Document)
	}
}
