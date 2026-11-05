//go:build !race

package projectsitemap

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/helpers/disktest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
)

// This serial volume proof keeps the production operation deadline. The complete
// normal release unit pass runs it only with the explicit heavy-I/O opt-in;
// concurrency and lifecycle tests stay raced.
func TestPostgresSitemapSingleRunExceedsRemovedPublicationCaps(t *testing.T) {
	disktest.Require(t)
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	pool := sitemapPool(t)
	ctx, cancel := context.WithTimeout(t.Context(), 8*time.Minute)
	defer cancel()
	createSitemapRun(t, ctx, pool, "producer")
	policy, err := artifactpostgres.NewHistoryPolicy(2, artifactpostgres.DefaultHistoryMaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	artifactHistory = policy
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	// A realistic map of 300 annotated operations ensures the assertion measures
	// actual encoded publication bytes, rather than a synthetic byte counter.
	d := Empty()
	owner := Authority{SourceID: "owner", Kind: "manual"}
	d.Sources = append(d.Sources, Source{Provenance: owner.provenance(), Generation: 1, PolicyDigest: key(Projection)})
	note := strings.Repeat("A public route description. ", 140)
	for i := range 300 {
		path := fmt.Sprintf("/seed/%d", i)
		group, err := ensureGroup(&d, "https://example.test", "GET", path)
		if err != nil {
			t.Fatal(err)
		}
		group.Members = append(group.Members, Member{ID: memberID(group.Origin, group.Method, path), Path: path, Summaries: []Summary{}, Examples: []string{}, Evidence: []Evidence{}, Annotations: []Annotation{{Provenance: owner.provenance(), Note: note, Tags: []string{}}}})
	}
	data, err := encode(d)
	if err != nil {
		t.Fatal(err)
	}
	artifactsService := artifacts.NewService(artifactStorage.Repository(pool))
	scope, _ := artifacts.ProjectScope("project")
	if _, err := artifactsService.Create(ctx, scope, mapRef(), artifacts.Payload{MediaType: MediaType, Data: data}); err != nil {
		t.Fatal(err)
	}
	store, _ := artifactsService.Project("project")
	service := NewService(pool, artifactStorage)
	a := httpAuthority("producer")
	var bytesPublished int64
	var first Result
	for i := range 257 {
		result, err := service.Mutate(ctx, a, fmt.Sprintf("publish-%d", i), Mutation{Operation: "observe", Observations: []Observation{observation(fmt.Sprintf("/observed/%d", i))}})
		if err != nil {
			t.Fatalf("publication %d: %v", i, err)
		}
		if !result.Changed || result.Replayed {
			t.Fatalf("publication %d was not new", i)
		}
		if i == 0 {
			first = result
		}
		metadata, err := store.Metadata(ctx, result.Ref)
		if err != nil {
			t.Fatal(err)
		}
		bytesPublished += metadata.Size
	}
	if bytesPublished <= 256<<20 {
		t.Fatalf("only %d encoded bytes published", bytesPublished)
	}
	var retained int
	var retainedBytes int64
	if err := pool.QueryRow(ctx, `SELECT retained_revisions,retained_bytes FROM artifact_bindings WHERE scope_kind='project' AND scope_id='project' AND namespace='sitemap' AND name='map'`).Scan(&retained, &retainedBytes); err != nil {
		t.Fatal(err)
	}
	if retained != 2 || retainedBytes > policy.MaxBytes() {
		t.Fatalf("common history exceeded: %d / %d", retained, retainedBytes)
	}
	if _, err := readSitemap(service, ctx, "project", *first.Ref.Revision); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("old unpinned source: %v", err)
	}
	t.Logf("single producer Run: accepted=257, encoded_publication_bytes=%d, retained_revisions=%d, retained_bytes=%d", bytesPublished, retained, retainedBytes)
}
