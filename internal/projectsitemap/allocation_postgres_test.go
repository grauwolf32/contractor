package projectsitemap

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func readerPolicy() contracts.ProjectSitemapPolicy {
	return contracts.ProjectSitemapPolicy{Version: contracts.SitemapAccessVersion}
}

func TestPostgresSitemapSharedMapSnapshotAndReplay(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	createSitemapRun(t, ctx, pool, "consumer")
	service := NewService(pool)
	traffic := httpAuthority("traffic")
	first, err := service.Mutate(ctx, traffic, "visible", Mutation{Operation: "observe", Observations: []Observation{observation("/visible")}})
	if err != nil {
		t.Fatal(err)
	}
	manual := Authority{OwnerID: "owner", ProjectID: "project", SourceID: "owner", Kind: "manual"}
	mixed, err := service.Mutate(ctx, manual, "declared", Mutation{Operation: "declare", ExpectedRevision: *first.Ref.Revision, URL: "https://example.test/declared", Method: "GET"})
	if err != nil {
		t.Fatal(err)
	}
	access, err := service.AllocationAccess(ctx, "consumer", "research", readerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if access.OwnerID != "owner" || access.ProjectID != "project" || access.ManualSourceID == "owner" {
		t.Fatal("allocation authority was not derived")
	}
	wire, _ := json.Marshal(access)
	if bytes.Contains(wire, []byte(`"owner"`)) || bytes.Contains(wire, []byte(`"project"`)) {
		t.Fatal("access exposed scope selectors")
	}
	view, err := service.ReadAllocation(ctx, access, *mixed.Ref.Revision)
	if err != nil || len(view.Document.Groups) != 2 {
		t.Fatalf("shared view: %+v %v", view, err)
	}
	ownerView, err := service.Read(ctx, "owner", "project", *mixed.Ref.Revision)
	if err != nil || !reflect.DeepEqual(ownerView, view) {
		t.Fatalf("allocation map differs from owner map: %v", err)
	}
	selection := SnapshotSelection{Revision: *mixed.Ref.Revision}
	for _, group := range view.Document.Groups {
		selection.Groups = append(selection.Groups, group.ID)
		if group.Route == "/declared" && (len(group.Declarations) != 1 || len(group.Members[0].Summaries) != 0) {
			t.Fatal("a declaration fabricated an observation")
		}
	}
	saved, err := service.SnapshotAllocation(ctx, access, "save", selection)
	if err != nil {
		t.Fatal(err)
	}
	if saved.Replayed || saved.Artifact.Ref.Namespace != "research" || saved.Artifact.Ref.Revision == nil {
		t.Fatal("invalid materialized context")
	}
	store, _ := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Run("consumer")
	read, err := store.Read(ctx, saved.Artifact.Ref)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(read.Payload.Data, []byte("/declared")) || !bytes.Contains(read.Payload.Data, []byte("/visible")) {
		t.Fatal("snapshot lost knowledge from another source")
	}
	var snapshot snapshotDocument
	if err := json.Unmarshal(read.Payload.Data, &snapshot); err != nil {
		t.Fatal(err)
	}
	if snapshot.Source.Revision == nil || *snapshot.Source.Revision != *mixed.Ref.Revision || len(snapshot.Document.Groups) != 2 {
		t.Fatal("context lost original exact source")
	}
	metadata, err := store.Metadata(ctx, saved.Artifact.Ref)
	if err != nil || !metadata.Frozen {
		t.Fatalf("context is not frozen: %v", err)
	}
	policy, _ := artifacts.NewHistoryPolicy(2, artifacts.DefaultHistoryMaxBytes)
	ctx = artifacts.WithHistoryPolicy(ctx, policy)
	for i := range 4 {
		if _, err := service.Mutate(ctx, traffic, fmt.Sprintf("advance-%d", i), Mutation{Operation: "observe", Observations: []Observation{observation(fmt.Sprintf("/advance/%d", i))}}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := service.Read(ctx, "owner", "project", *first.Ref.Revision); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("unheld revision retained: %v", err)
	}
	if _, err := service.Read(ctx, "owner", "project", *mixed.Ref.Revision); err != nil {
		t.Fatalf("projection lineage did not protect exact source: %v", err)
	}
	replay, err := service.SnapshotAllocation(ctx, access, "save", selection)
	if err != nil || !replay.Replayed || !replay.Artifact.Ref.SameExact(saved.Artifact.Ref) {
		t.Fatalf("snapshot replay: %+v %v", replay, err)
	}
	selection.Groups = nil
	selection.Members = []string{view.Document.Groups[0].Members[0].ID}
	if _, err := service.SnapshotAllocation(ctx, access, "save", selection); !errors.Is(err, contracts.ErrIdempotencyConflict) {
		t.Fatalf("changed intent reused context key: %v", err)
	}
}

func TestPostgresSitemapEditorsShareMembersAndDeriveProjectAuthority(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	createSitemapRun(t, ctx, pool, "consumer")
	service := NewService(pool)
	access, err := service.AllocationAccess(ctx, "consumer", "research", readerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	producer := httpAuthority("producer")
	producer.SourceID = "earlier-research"
	seeded, err := service.Mutate(ctx, producer, "seed", Mutation{Operation: "observe", Observations: []Observation{observation("/shared")}})
	if err != nil {
		t.Fatal(err)
	}
	full, err := service.Read(ctx, "owner", "project", "")
	if err != nil {
		t.Fatal(err)
	}
	command := Mutation{Operation: "remove", MarkerID: "removed-by-consumer", ExpectedRevision: *seeded.Ref.Revision, MemberIDs: []string{full.Document.Groups[0].Members[0].ID}}
	removedEarlier, err := service.MutateAllocation(ctx, access, "hide-earlier-research", command)
	if err != nil {
		t.Fatalf("editor cannot select another producer's member: %v", err)
	}
	restored, err := service.MutateAllocation(ctx, access, "restore-earlier-research", Mutation{Operation: "restore", MarkerID: command.MarkerID, ExpectedRevision: *removedEarlier.Ref.Revision})
	if err != nil {
		t.Fatal(err)
	}
	command = Mutation{Operation: "declare", ExpectedRevision: *restored.Ref.Revision, URL: "https://example.test/manual", Method: "GET"}
	added, err := service.MutateAllocation(ctx, access, "declare", command)
	if err != nil {
		t.Fatal(err)
	}
	view, err := service.ReadAllocation(ctx, access, *added.Ref.Revision)
	if err != nil || len(view.Document.Groups) != 2 || len(view.Document.Markers) != 0 {
		t.Fatalf("declared shared view: %+v %v", view, err)
	}
	command = Mutation{Operation: "remove", MarkerID: "removed-by-consumer", ExpectedRevision: *added.Ref.Revision, MemberIDs: []string{view.Document.Groups[0].Members[0].ID}}
	removed, err := service.MutateAllocation(ctx, access, "remove", command)
	if err != nil {
		t.Fatal(err)
	}
	if replay, err := service.MutateAllocation(ctx, access, "remove", command); err != nil || !replay.Replayed || !replay.Ref.SameExact(removed.Ref) {
		t.Fatalf("accepted selection replay: %+v %v", replay, err)
	}
	stores := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	scope, _ := artifacts.ProjectScope("project")
	schema, err := stores.Create(ctx, scope, contracts.ArtifactRef{Namespace: "schema", Name: "selected"}, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"openapi":"3.1.0"}`)})
	if err != nil {
		t.Fatal(err)
	}
	_, err = artifactstest.ForkProjectInput(ctx, stores, "project", schema.Ref, "consumer", "schema")
	if err != nil {
		t.Fatal(err)
	}
	forged := contracts.ExactContent(schema.Ref, "application/json", []byte(`{"openapi":"3.1.0"}`))
	if _, err := service.MutateAllocation(ctx, access, "forged-project-source", Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: access.ManualSourceID, Artifact: &forged, Rules: []Rule{}}}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("manual editor gained a Project artifact selector: %v", err)
	}
	selected := readerPolicy()
	selected.SnapshotInputs = []string{"schema"}
	granted, err := service.AllocationAccess(ctx, "consumer", "research", selected)
	if err != nil || granted.ManualSourceID != access.ManualSourceID {
		t.Fatalf("snapshot input configuration changed the map source: %+v %v", granted, err)
	}
	selected.SnapshotInputs = []string{"unselected"}
	if _, err := service.AllocationAccess(ctx, "consumer", "research", selected); err != nil {
		t.Fatalf("missing optional snapshot input blocked map access: %v", err)
	}
	if _, err := service.AllocationAccess(ctx, "unknown", "research", readerPolicy()); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("foreign Run: %v", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE projects SET lifecycle_state='deleting',deletion_phase='cancelling',deletion_requested_at=clock_timestamp(),revision=revision+1,updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond') WHERE project_id='project'`); err != nil {
		t.Fatal(err)
	}
	if _, err := service.AllocationAccess(ctx, "consumer", "research", readerPolicy()); !errors.Is(err, projectstore.ErrDeleting) {
		t.Fatalf("deleting Project authority: %v", err)
	}
}

func TestPostgresSitemapContextProjectionSerializesWithHistoryEviction(t *testing.T) {
	for iteration := range 6 {
		t.Run(fmt.Sprint(iteration), func(t *testing.T) {
			pool := sitemapPool(t)
			ctx := t.Context()
			createSitemapRun(t, ctx, pool, "consumer")
			service := NewService(pool)
			policy, _ := artifacts.NewHistoryPolicy(2, artifacts.DefaultHistoryMaxBytes)
			ctx = artifacts.WithHistoryPolicy(ctx, policy)
			authority := httpAuthority("producer")
			original, err := service.Mutate(ctx, authority, "original", Mutation{Operation: "observe", Observations: []Observation{observation("/selected")}})
			if err != nil {
				t.Fatal(err)
			}
			access, err := service.AllocationAccess(ctx, "consumer", "research", readerPolicy())
			if err != nil {
				t.Fatal(err)
			}
			view, err := service.ReadAllocation(ctx, access, *original.Ref.Revision)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := service.Mutate(ctx, authority, "advance", Mutation{Operation: "observe", Observations: []Observation{observation("/advance")}}); err != nil {
				t.Fatal(err)
			}
			start := make(chan struct{})
			var wg sync.WaitGroup
			var saved SnapshotResult
			var snapshotErr, writeErr error
			wg.Go(func() {
				<-start
				saved, snapshotErr = service.SnapshotAllocation(ctx, access, "snapshot", SnapshotSelection{Revision: *original.Ref.Revision, Groups: []string{view.Document.Groups[0].ID}})
			})
			wg.Go(func() {
				<-start
				_, writeErr = service.Mutate(ctx, authority, "evict", Mutation{Operation: "observe", Observations: []Observation{observation("/evict")}})
			})
			close(start)
			wg.Wait()
			// A snapshot may protect the planned victim before eviction locks it.
			// The store then commits maintenance and asks the caller to retry.
			if errors.Is(writeErr, artifacts.ErrHistoryMaintenancePending) {
				var pending *artifacts.HistoryMaintenanceError
				if snapshotErr != nil || !errors.As(writeErr, &pending) || !pending.Maintained || pending.Cause != nil {
					t.Fatalf("unexpected maintenance outcome: snapshot=%v write=%v", snapshotErr, writeErr)
				}
				var retried Result
				retried, writeErr = service.Mutate(ctx, authority, "evict", Mutation{Operation: "observe", Observations: []Observation{observation("/evict")}})
				if writeErr == nil && (!retried.Changed || retried.Replayed) {
					t.Fatalf("maintenance committed the mutation before retry: %+v", retried)
				}
			}
			if writeErr != nil {
				t.Fatal(writeErr)
			}
			if snapshotErr != nil && !errors.Is(snapshotErr, artifacts.ErrArtifactNotFound) {
				t.Fatal(snapshotErr)
			}
			if snapshotErr == nil {
				if _, err := service.Read(ctx, "owner", "project", *original.Ref.Revision); err != nil {
					t.Fatalf("committed snapshot lost exact source: %v", err)
				}
				run, _ := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Run("consumer")
				if _, err := run.Read(ctx, saved.Artifact.Ref); err != nil {
					t.Fatal(err)
				}
			} else {
				var count int
				err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind='run' AND scope_id='consumer'`).Scan(&count)
				if err != nil || count != 0 {
					t.Fatalf("failed exact selection left artifacts: %d %v", count, err)
				}
			}
			// A caller cannot turn the unavailable exact ref into latest or cache bytes.
			missing := strings.Repeat("x", 32)
			if _, err := service.SnapshotAllocation(ctx, access, "missing", SnapshotSelection{Revision: missing, Groups: []string{view.Document.Groups[0].ID}}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
				t.Fatalf("missing exact source: %v", err)
			}
		})
	}
}

func TestPostgresSitemapSnapshotMaterializesOnlyNeededAuthorizedExactInputs(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	createSitemapRun(t, ctx, pool, "consumer")
	service := NewService(pool)
	stores := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	scope, _ := artifacts.ProjectScope("project")
	originalBytes := []byte(`{"openapi":"3.1.0","info":{"version":"1"}}`)
	schema, err := stores.Create(ctx, scope, contracts.ArtifactRef{Namespace: "schemas", Name: "api"}, artifacts.Payload{MediaType: "application/json", Data: originalBytes})
	if err != nil {
		t.Fatal(err)
	}
	fork, err := artifactstest.ForkProjectInput(ctx, stores, "project", schema.Ref, "consumer", "api")
	if err != nil {
		t.Fatal(err)
	}
	policy := readerPolicy()
	policy.SnapshotInputs = []string{"api"}
	run, _ := stores.Run("consumer")
	workingBytes := []byte(`{"openapi":"3.1.0","info":{"version":"worker-canary"}}`)
	nextSchema, err := stores.Replace(ctx, scope, contracts.ArtifactRef{Namespace: "schemas", Name: "api"}, artifacts.Payload{MediaType: "application/json", Data: workingBytes}, *schema.Ref.Revision)
	if err != nil {
		t.Fatal(err)
	}
	nextInput, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "api"}, artifacts.Payload{MediaType: "application/json", Data: workingBytes}, fork.TargetRef.Revision)
	if err != nil {
		t.Fatal(err)
	}
	authority := Authority{OwnerID: "owner", ProjectID: "project", SourceID: "schema", Kind: "openapi", RunID: "producer"}
	exact := contracts.ExactContent(schema.Ref, "application/json", originalBytes)
	_, err = service.Mutate(ctx, authority, "schema", Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: "schema", Artifact: &exact, Rules: []Rule{{SourceID: "schema", Origin: "https://example.test", Method: "GET", Path: "/declared"}}}})
	if err != nil {
		t.Fatal(err)
	}
	authority.SourceID = "working-copy-only"
	workingExact := contracts.ExactContent(nextSchema.Ref, "application/json", workingBytes)
	published, err := service.Mutate(ctx, authority, "ungranted-schema", Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: authority.SourceID, Artifact: &workingExact, Rules: []Rule{{SourceID: authority.SourceID, Origin: "https://example.test", Method: "GET", Path: "/working-copy-canary"}}}})
	if err != nil {
		t.Fatal(err)
	}
	access, err := service.AllocationAccess(ctx, "consumer", "research", policy)
	if err != nil {
		t.Fatal(err)
	}
	view, err := service.ReadAllocation(ctx, access, *published.Ref.Revision)
	if err != nil || len(view.Document.Groups) != 2 {
		t.Fatalf("shared view hid known schema routes: %+v %v", view, err)
	}
	declared := slices.IndexFunc(view.Document.Groups, func(group Group) bool { return group.Route == "/declared" })
	working := slices.IndexFunc(view.Document.Groups, func(group Group) bool { return group.Route == "/working-copy-canary" })
	if declared < 0 || working < 0 {
		t.Fatal("shared view lost declaration metadata")
	}
	saved, err := service.SnapshotAllocation(ctx, access, "schema-context", SnapshotSelection{Revision: *published.Ref.Revision, Groups: []string{view.Document.Groups[declared].ID}, RequireReferencedInputs: true})
	if err != nil {
		t.Fatal(err)
	}
	body, err := run.Read(ctx, saved.Artifact.Ref)
	if err != nil {
		t.Fatal(err)
	}
	var snapshot snapshotDocument
	if err := json.Unmarshal(body.Payload.Data, &snapshot); err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Inputs) != 1 || snapshot.Inputs[0].Ref.Namespace != "research" || snapshot.Inputs[0].Ref.SameExact(nextInput.Ref) || snapshot.Inputs[0].Ref.SameExact(fork.TargetRef) {
		t.Fatal("context retained only a mutable input mention")
	}
	metadata, err := run.Metadata(ctx, snapshot.Inputs[0].Ref)
	if err != nil || !metadata.Frozen {
		t.Fatalf("needed input clone: %+v %v", metadata, err)
	}
	input, err := run.ReadExact(ctx, snapshot.Inputs[0])
	if err != nil || !bytes.Equal(input.Payload.Data, originalBytes) {
		t.Fatalf("input clone content: %v", err)
	}
	limited, _ := artifacts.NewHistoryPolicy(2, artifacts.DefaultHistoryMaxBytes)
	if _, err := run.Write(artifacts.WithHistoryPolicy(ctx, limited), contracts.ArtifactRef{Namespace: "inputs", Name: "api"}, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"later":true}`)}, nextInput.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	if _, err := run.Read(ctx, nextInput.Ref); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("a working-copy mention created an unauthorized hold: %v", err)
	}
	if _, err := run.ReadExact(ctx, snapshot.Inputs[0]); err != nil {
		t.Fatalf("frozen context input lost its selected bytes: %v", err)
	}
	full, err := service.AllocationAccess(ctx, "consumer", "ungranted", readerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.SnapshotAllocation(ctx, full, "missing-required", SnapshotSelection{Revision: *published.Ref.Revision, Groups: []string{view.Document.Groups[declared].ID}, RequireReferencedInputs: true}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("ungranted schema materialized: %v", err)
	}
	selection := SnapshotSelection{Revision: *published.Ref.Revision, Groups: []string{view.Document.Groups[working].ID}, RequireReferencedInputs: true}
	if _, err := service.SnapshotAllocation(ctx, access, "working-required", selection); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("an advanced working input authorized different original bytes: %v", err)
	}
	selection.RequireReferencedInputs = false
	optional, err := service.SnapshotAllocation(ctx, access, "working-optional", selection)
	if err != nil {
		t.Fatal(err)
	}
	optionalRead, err := run.Read(ctx, optional.Artifact.Ref)
	if err != nil {
		t.Fatal(err)
	}
	var optionalSnapshot snapshotDocument
	if err := json.Unmarshal(optionalRead.Payload.Data, &optionalSnapshot); err != nil {
		t.Fatal(err)
	}
	if len(optionalSnapshot.Inputs) != 0 || len(optionalSnapshot.Document.Groups) != 1 || optionalSnapshot.Document.Groups[0].Route != "/working-copy-canary" || !slices.ContainsFunc(optionalSnapshot.Document.Gaps, func(gap Gap) bool { return gap.Reason == "source-unavailable" }) {
		t.Fatal("optional input handling discarded known routes or invented retained bytes")
	}
}
