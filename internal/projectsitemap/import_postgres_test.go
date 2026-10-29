package projectsitemap

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

func writeScannerReport(t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID, name string, data []byte) artifacts.WriteResult {
	t.Helper()
	run, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Run(runID)
	if err != nil {
		t.Fatal(err)
	}
	report, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "research", Name: name}, artifacts.Payload{MediaType: "application/json", Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return report
}

func importAccess(t *testing.T, ctx context.Context, service *Service, runID string) AllocationAccess {
	t.Helper()
	access, err := service.AllocationAccess(ctx, runID, "research", readerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	return access
}

func TestPostgresSitemapScannerImportReplaysAfterReportAndMapEviction(t *testing.T) {
	pool := sitemapPool(t)
	history, _ := artifacts.NewHistoryPolicy(2, artifacts.DefaultHistoryMaxBytes)
	ctx := artifacts.WithHistoryPolicy(t.Context(), history)
	createSitemapRun(t, ctx, pool, "scanner")
	service := NewService(pool)
	access := importAccess(t, ctx, service, "scanner") // Explicit import does not need automatic collection.
	report := writeScannerReport(t, ctx, pool, "scanner", "report", scannerFixture(t, "ffuf"))
	selection := ImportSelection{Format: "ffuf-report@1", Artifact: report.Ref}
	first, err := service.ImportAllocation(ctx, access, "first", selection)
	if err != nil {
		t.Fatal(err)
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Document.Groups) != 1 || len(view.Document.Sources) != 1 {
		t.Fatalf("bad import: %+v", view.Document)
	}
	source := view.Document.Sources[0]
	member := view.Document.Groups[0].Members[0]
	if source.Artifact != nil || source.Provenance.SourceID != key("ffuf", "") || len(member.Summaries) != 1 || len(member.Evidence) != 1 {
		t.Fatalf("wrong source/evidence: %+v %+v", source, member)
	}
	summary := member.Summaries[0]
	if summary.First != report.RevisionCreatedAt.UTC().Format(time.RFC3339Nano) || summary.First != summary.Last || summary.Status != 200 {
		t.Fatalf("wrong immutable report timestamp: %+v", summary)
	}
	if member.Evidence[0].RunID != "scanner" || !member.Evidence[0].Artifact.Ref.SameExact(report.Ref) {
		t.Fatal("missing exact report locator")
	}
	run, _ := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Run("scanner")
	latest := report
	for i := range 4 {
		data := editScannerFixture(t, "ffuf", func(e, o map[string]any) { o["durationMs"] = i + 1 })
		latest, err = run.Write(ctx, contracts.ArtifactRef{Namespace: "research", Name: "report"}, artifacts.Payload{MediaType: "application/json", Data: data}, latest.Ref.Revision)
		if err != nil {
			t.Fatal(err)
		}
		access.Policy.CollectObservations = true
		if _, err = service.CollectAllocation(ctx, access, fmt.Sprintf("advance-%d", i), CollectionIngress{Kind: "http", Observations: []Observation{observation(fmt.Sprintf("/advance/%d", i))}}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err = run.Read(ctx, report.Ref); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("report held implicitly: %v", err)
	}
	if _, err = readSitemap(service, ctx, "project", *first.Ref.Revision); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("map held implicitly: %v", err)
	}
	before, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	replay, err := service.ImportAllocation(ctx, access, "first", selection)
	if err != nil || !replay.Replayed || !replay.Ref.SameExact(first.Ref) {
		t.Fatalf("failed replay after eviction: %+v %v", replay, err)
	}
	after, err := readSitemap(service, ctx, "project", "")
	if err != nil || after.Revision != before.Revision {
		t.Fatal("replay republished map")
	}
	if _, err = service.ImportAllocation(ctx, access, "unavailable", selection); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("new import followed latest: %v", err)
	}
	selection.Artifact = latest.Ref
	if _, err = service.ImportAllocation(ctx, access, "first", selection); !errors.Is(err, contracts.ErrIdempotencyConflict) {
		t.Fatalf("changed source reused receipt: %v", err)
	}
}

func TestPostgresSitemapScannerImportsAggregateAcrossRunsAndBoundEvidence(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	service := NewService(pool)
	var firstTime, lastTime string
	var access AllocationAccess
	for i := range 32 {
		runID := fmt.Sprintf("scanner-%d", i)
		createSitemapRun(t, ctx, pool, runID)
		access = importAccess(t, ctx, service, runID)
		report := writeScannerReport(t, ctx, pool, runID, "report", scannerFixture(t, "ffuf"))
		at := report.RevisionCreatedAt.UTC().Format(time.RFC3339Nano)
		if i == 0 {
			firstTime = at
		}
		lastTime = at
		if _, err := service.ImportAllocation(ctx, access, runID, ImportSelection{Format: "ffuf-report@1", Artifact: report.Ref}); err != nil {
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
		t.Fatal("report time range lost")
	}
	if !slices.ContainsFunc(view.Document.Gaps, func(g Gap) bool { return g.Reason == "evidence" && g.Count == 24 }) {
		t.Fatalf("evidence overflow not factual: %+v", view.Document.Gaps)
	}
	different := editScannerFixture(t, "ffuf", func(e, o map[string]any) { o["results"].([]any)[0].(map[string]any)["status"] = 500 })
	report := writeScannerReport(t, ctx, pool, access.RunID, "new-behavior", different)
	if _, err := service.ImportAllocation(ctx, access, "new-behavior", ImportSelection{Format: "ffuf-report@1", Artifact: report.Ref}); err != nil {
		t.Fatal(err)
	}
	katana := writeScannerReport(t, ctx, pool, access.RunID, "katana", scannerFixture(t, "katana"))
	if _, err := service.ImportAllocation(ctx, access, "katana", ImportSelection{Format: "katana-report@1", Artifact: katana.Ref}); err != nil {
		t.Fatal(err)
	}
	deployment := "staging"
	access.Policy.Deployment = &deployment
	if _, err := service.ImportAllocation(ctx, access, "staging", ImportSelection{Format: "ffuf-report@1", Artifact: report.Ref}); err != nil {
		t.Fatal(err)
	}
	view, err = readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Document.Sources) != 3 || len(view.Document.Groups[0].Members[0].Summaries) != 4 {
		t.Fatalf("new behavior/kind/deployment discarded: %+v", view.Document)
	}
}

func TestPostgresSitemapScannerImportBoundaryAndRejectedReportRollback(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	createSitemapRun(t, ctx, pool, "scanner")
	service := NewService(pool)
	access := importAccess(t, ctx, service, "scanner")
	for index, data := range [][]byte{[]byte(`{"schemaVersion":1}`), ffufBoundaryReport(t, 101), []byte(strings.Repeat(" ", maxScannerReportBytes+1))} {
		report := writeScannerReport(t, ctx, pool, "scanner", fmt.Sprintf("invalid-%d", index), data)
		if _, err := service.ImportAllocation(ctx, access, "bounded", ImportSelection{Format: "ffuf-report@1", Artifact: report.Ref}); !errors.Is(err, ErrInvalid) {
			t.Fatalf("invalid report accepted: %v", err)
		}
		project, _ := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project("project")
		if _, err := project.Read(ctx, mapRef()); !errors.Is(err, artifacts.ErrArtifactNotFound) {
			t.Fatalf("invalid import wrote map: %v", err)
		}
	}
	report := writeScannerReport(t, ctx, pool, "scanner", "valid", ffufBoundaryReport(t, 100))
	accepted, err := service.ImportAllocation(ctx, access, "bounded", ImportSelection{Format: "ffuf-report@1", Artifact: report.Ref})
	if err != nil {
		t.Fatalf("failed import consumed receipt: %v", err)
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Document.Groups) != 100 || !accepted.Changed {
		t.Fatalf("100-row boundary lost: %d groups", len(view.Document.Groups))
	}
	partial := editScannerFixture(t, "ffuf", func(e, o map[string]any) {
		o["results"] = []any{}
		o["scanComplete"] = false
		o["status"] = "failed"
		o["errorCode"] = "scan_request_failed"
		o["requestErrors"] = 1
	})
	report = writeScannerReport(t, ctx, pool, "scanner", "partial", partial)
	if _, err = service.ImportAllocation(ctx, access, "partial", ImportSelection{Format: "ffuf-report@1", Artifact: report.Ref}); err != nil {
		t.Fatal(err)
	}
	view, err = readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.ContainsFunc(view.Document.Gaps, func(g Gap) bool { return g.Reason == "request-errors" && g.Count == 1 }) {
		t.Fatal("empty partial report lost factual gap")
	}
}

func TestPostgresSitemapScannerImportVisibilityAndProjectLifecycle(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	service := NewService(pool)
	createSitemapRun(t, ctx, pool, "source")
	createSitemapRun(t, ctx, pool, "other")
	access := importAccess(t, ctx, service, "source")
	report := writeScannerReport(t, ctx, pool, "source", "report", scannerFixture(t, "ffuf"))
	selection := ImportSelection{Format: "ffuf-report@1", Artifact: report.Ref}
	other := importAccess(t, ctx, service, "other")
	if _, err := service.ImportAllocation(ctx, other, "foreign-run", selection); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("cross-Run exact ref readable: %v", err)
	}
	reserved := selection
	reserved.Artifact.Namespace = artifactpolicy.RunSystemNamespace
	if _, err := service.ImportAllocation(ctx, access, "system", reserved); !errors.Is(err, artifacts.ErrReservedNamespace) {
		t.Fatalf("hidden Run namespace readable: %v", err)
	}
	foreign := access
	foreign.OwnerID = "other"
	if _, err := service.ImportAllocation(ctx, foreign, "foreign-owner", selection); !errors.Is(err, projectstore.ErrNotFound) {
		t.Fatalf("foreign owner admitted: %v", err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := service.ImportAllocation(cancelled, access, "cancelled", selection); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled import admitted: %v", err)
	}
	if _, err := service.ImportAllocation(ctx, access, "first", selection); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE projects SET lifecycle_state='deleting',deletion_phase='cancelling',deletion_requested_at=clock_timestamp(),revision=revision+1,updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond') WHERE project_id='project'`); err != nil {
		t.Fatal(err)
	}
	if _, err := service.ImportAllocation(ctx, access, "first", selection); !errors.Is(err, projectstore.ErrDeleting) {
		t.Fatalf("receipt bypassed Project deletion fence: %v", err)
	}
}

func TestPostgresSitemapCaidoCollectionSkipsOnlyInvalidRows(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	service := NewService(pool)
	createSitemapRun(t, ctx, pool, "caido")
	access := importAccess(t, ctx, service, "caido")
	access.Policy.CollectObservations = true
	valid := observation("/users?password=excluded")
	valid.At = "2026-10-29T15:00:00+03:00"
	invalid := observation("/invalid")
	invalid.At = "unknown"
	input := CollectionIngress{Kind: "caido", Observations: []Observation{valid, invalid}}
	committed, err := service.CollectAllocation(ctx, access, "mixed", input)
	if err != nil {
		t.Fatal(err)
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Document.Groups) != 1 || len(view.Document.Sources) != 1 || view.Document.Sources[0].Provenance.Kind != "caido" {
		t.Fatalf("invalid rows poisoned valid rows: %+v", view.Document)
	}
	member := view.Document.Groups[0].Members[0]
	// One omitted query-value projection plus one rejected provider row.
	if len(member.Evidence) != 0 || member.Summaries[0].First != "2026-10-29T12:00:00Z" || len(view.Document.Gaps) != 1 || view.Document.Gaps[0].Reason != "projection" || view.Document.Gaps[0].Count != 2 {
		t.Fatalf("false evidence/timestamp/gap: %+v", view.Document)
	}
	replay, err := service.CollectAllocation(ctx, access, "mixed", input)
	if err != nil || !replay.Replayed || !replay.Ref.SameExact(committed.Ref) {
		t.Fatal("Caido receipt failed")
	}
	input.Observations = append(input.Observations, invalid)
	if _, err = service.CollectAllocation(ctx, access, "mixed", input); !errors.Is(err, contracts.ErrIdempotencyConflict) {
		t.Fatalf("changed invalid count reused receipt: %v", err)
	}
	input.Kind = "http"
	if _, err = service.CollectAllocation(ctx, access, "strict-http", input); !errors.Is(err, ErrInvalid) {
		t.Fatalf("HTTP became lenient: %v", err)
	}
	deployment := "all-invalid"
	access.Policy.Deployment = &deployment
	if _, err = service.CollectAllocation(ctx, access, "all-invalid", CollectionIngress{Kind: "caido", Observations: []Observation{invalid}}); err != nil {
		t.Fatal(err)
	}
	view, err = readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Document.Sources) != 2 || !slices.ContainsFunc(view.Document.Gaps, func(g Gap) bool {
		return g.SourceID == key("caido", deployment) && g.Reason == "projection" && g.Count == 1
	}) {
		t.Fatal("all-invalid batch lost source/gap")
	}
	if _, err = service.CollectAllocation(ctx, access, "later-valid", CollectionIngress{Kind: "caido", Observations: []Observation{valid}}); err != nil {
		t.Fatal(err)
	}
}
