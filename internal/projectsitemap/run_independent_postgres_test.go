package projectsitemap

import (
	"bytes"
	"errors"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPostgresSitemapRepeatedRunsAggregateBehavior(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	service := NewService(pool)
	policy := readerPolicy()
	policy.CollectObservations = true
	deployment := "production"
	policy.Deployment = &deployment
	start := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	var access AllocationAccess
	for index := range 32 {
		runID := fmt.Sprintf("repeat-%d", index)
		createSitemapRun(t, ctx, pool, runID)
		var err error
		access, err = service.AllocationAccess(ctx, runID, "research", policy)
		if err != nil {
			t.Fatal(err)
		}
		o := observation("/shared")
		o.Parameters = []string{"expand"}
		// Deliver newest first so aggregation must preserve both ends of time.
		o.At = start.Add(time.Duration(31-index) * time.Minute).Format(time.RFC3339Nano)
		if _, err := service.CollectAllocation(ctx, access, runID, CollectionIngress{Kind: "http", Observations: []Observation{o}}); err != nil {
			t.Fatal(err)
		}
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Document.Groups) != 1 || len(view.Document.Sources) != 1 || len(view.Document.Groups[0].Members) != 1 {
		t.Fatalf("repeated Runs multiplied inventory: %+v", view.Document)
	}
	member := view.Document.Groups[0].Members[0]
	if len(member.Summaries) != 1 || member.Omitted != 0 || len(view.Document.Gaps) != 0 {
		t.Fatalf("repeated Runs consumed summary capacity: %+v", view.Document)
	}
	summary := member.Summaries[0]
	if summary.First != start.Format(time.RFC3339Nano) || summary.Last != start.Add(31*time.Minute).Format(time.RFC3339Nano) || summary.Provenance.Deployment != deployment {
		t.Fatalf("source or time range lost: %+v", summary)
	}
	encoded, err := encode(view.Document)
	if err != nil {
		t.Fatal(err)
	}
	for _, removed := range []string{`"runId"`, `"context"`, `"sessionGeneration"`, `"credentialRevision"`} {
		if bytes.Contains(encoded, []byte(removed)) {
			t.Fatalf("retained execution attribution: %s", removed)
		}
	}
	// A new status after more than sixteen producer Runs must still fit.
	failure := observation("/shared")
	failure.At = start.Add(32 * time.Minute).Format(time.RFC3339Nano)
	failure.Parameters, failure.Status = []string{"expand"}, 500
	timeout := failure
	timeout.Outcome, timeout.Status = "timeout", 0
	shape := observation("/shared")
	shape.Parameters = []string{"page"}
	if _, err := service.CollectAllocation(ctx, access, "new-behavior", CollectionIngress{Kind: "http", Observations: []Observation{failure, timeout, shape}}); err != nil {
		t.Fatal(err)
	}
	staging := "staging"
	access.Policy.Deployment = &staging
	if _, err := service.CollectAllocation(ctx, access, "other-deployment", CollectionIngress{Kind: "http", Observations: []Observation{failure}}); err != nil {
		t.Fatal(err)
	}
	view, err = readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	member = view.Document.Groups[0].Members[0]
	if len(member.Summaries) != 5 || len(view.Document.Sources) != 2 || member.Omitted != 0 || len(view.Document.Gaps) != 0 {
		t.Fatalf("behavior or deployment diversity lost: %+v", view.Document)
	}
	for _, want := range []struct {
		deployment, outcome, parameter string
		status                         int
	}{{deployment, "response", "expand", 200}, {deployment, "response", "expand", 500}, {deployment, "timeout", "expand", 0}, {deployment, "response", "page", 200}, {staging, "response", "expand", 500}} {
		if !slices.ContainsFunc(member.Summaries, func(s Summary) bool {
			return s.Provenance.Deployment == want.deployment && s.Outcome == want.outcome && s.Status == want.status && slices.Equal(s.Parameters, []string{want.parameter})
		}) {
			t.Fatalf("missing behavior: %+v", want)
		}
	}
}

func TestPostgresSitemapManualRunsShareMetadataAndIsolateReceipts(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	service := NewService(pool)
	createSitemapRun(t, ctx, pool, "editor-one")
	createSitemapRun(t, ctx, pool, "editor-two")
	first, err := service.AllocationAccess(ctx, "editor-one", "research", readerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	second, err := service.AllocationAccess(ctx, "editor-two", "research", readerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	declare := EditorMutation{Operation: "declare", ExpectedRevision: "absent", URL: "https://example.test/shared", Method: "GET"}
	created, err := service.MutateAllocation(ctx, first, "declare-one", declare)
	if err != nil {
		t.Fatal(err)
	}
	// The same map semantics do not make different producers the same request.
	if _, err := service.MutateAllocation(ctx, second, "declare-one", declare); !errors.Is(err, contracts.ErrIdempotencyConflict) {
		t.Fatalf("another Run replayed a producer receipt: %v", err)
	}
	if replay, err := service.MutateAllocation(ctx, first, "declare-one", declare); err != nil || !replay.Replayed || !replay.Ref.SameExact(created.Ref) {
		t.Fatalf("original receipt lost: %+v %v", replay, err)
	}
	declare.ExpectedRevision = *created.Ref.Revision
	repeated, err := service.MutateAllocation(ctx, second, "declare-two", declare)
	if err != nil || repeated.Changed || !repeated.Ref.SameExact(created.Ref) {
		t.Fatalf("another Run duplicated declaration: %+v %v", repeated, err)
	}
	member := memberID("https://example.test", "GET", "/shared")
	annotation := EditorMutation{Operation: "annotate", ExpectedRevision: *created.Ref.Revision, MemberIDs: []string{member}, Note: "first note", Tags: []string{"old"}}
	annotated, err := service.MutateAllocation(ctx, first, "note-one", annotation)
	if err != nil {
		t.Fatal(err)
	}
	annotation.ExpectedRevision, annotation.Note, annotation.Tags = *annotated.Ref.Revision, "updated note", []string{"current"}
	if _, err := service.MutateAllocation(ctx, second, "note-two", annotation); err != nil {
		t.Fatal(err)
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	group := view.Document.Groups[0]
	if len(view.Document.Sources) != 1 || len(group.Declarations) != 1 || len(group.Members[0].Annotations) != 1 {
		t.Fatalf("manual Runs multiplied metadata: %+v", view.Document)
	}
	note := group.Members[0].Annotations[0]
	if note.Note != "updated note" || !slices.Equal(note.Tags, []string{"current"}) {
		t.Fatalf("annotation was not updated: %+v", note)
	}
}
