package projectsitemap

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestPostgresSitemapAllocationProjectionSnapshotAndReplay(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	createSitemapRun(t, ctx, pool, "consumer")
	service := NewService(pool)
	bb := httpAuthority("traffic")
	first, err := service.Mutate(ctx, bb, "visible", Mutation{Operation: "observe", Observations: []Observation{observation("/visible")}})
	if err != nil {
		t.Fatal(err)
	}
	sa := httpAuthority("source-run")
	sa.SourceID, sa.Mode = "private-source", "source-assisted"
	mixed, err := service.Mutate(ctx, sa, "restricted", Mutation{Operation: "observe", Observations: []Observation{observation("/restricted-canary")}})
	if err != nil {
		t.Fatal(err)
	}
	access, err := service.AllocationAccess(ctx, "consumer", "research", readerPolicy("blackbox"))
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
	if err != nil || len(view.Document.Groups) != 1 || view.Document.Groups[0].Route != "/visible" {
		t.Fatalf("safe view: %+v %v", view, err)
	}
	selection := SnapshotSelection{Revision: *mixed.Ref.Revision, ExpectedPolicyDigest: access.PolicyDigest, Groups: []string{view.Document.Groups[0].ID}}
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
	if bytes.Contains(read.Payload.Data, []byte("restricted-canary")) || bytes.Contains(read.Payload.Data, []byte("private-source")) {
		t.Fatal("snapshot materialized restricted bytes")
	}
	var snapshot snapshotDocument
	if err := json.Unmarshal(read.Payload.Data, &snapshot); err != nil {
		t.Fatal(err)
	}
	if snapshot.Source.Revision == nil || *snapshot.Source.Revision != *mixed.Ref.Revision || len(snapshot.Document.Groups) != 1 {
		t.Fatal("context lost original exact source")
	}
	metadata, err := store.Metadata(ctx, saved.Artifact.Ref)
	if err != nil || !metadata.Frozen {
		t.Fatalf("context is not frozen: %v", err)
	}
	policy, _ := artifacts.NewHistoryPolicy(2, artifacts.DefaultHistoryMaxBytes)
	ctx = artifacts.WithHistoryPolicy(ctx, policy)
	for i := range 4 {
		if _, err := service.Mutate(ctx, bb, fmt.Sprintf("advance-%d", i), Mutation{Operation: "observe", Observations: []Observation{observation(fmt.Sprintf("/advance/%d", i))}}); err != nil {
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

func TestPostgresSitemapAllocationUsesOnlySelectedExactInputsAndVisibleSelections(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	createSitemapRun(t, ctx, pool, "consumer")
	service := NewService(pool)
	access, err := service.AllocationAccess(ctx, "consumer", "research", readerPolicy("blackbox"))
	if err != nil {
		t.Fatal(err)
	}
	sa := httpAuthority("private")
	sa.Mode, sa.SourceID = "source-assisted", "restricted"
	seeded, err := service.Mutate(ctx, sa, "seed", Mutation{Operation: "observe", Observations: []Observation{observation("/private")}})
	if err != nil {
		t.Fatal(err)
	}
	full, err := service.Read(ctx, "owner", "project", "")
	if err != nil {
		t.Fatal(err)
	}
	command := Mutation{Operation: "remove", MarkerID: "removed-by-consumer", ExpectedRevision: *seeded.Ref.Revision, MemberIDs: []string{full.Document.Groups[0].Members[0].ID}}
	if _, err := service.MutateAllocation(ctx, access, "hide-foreign", command); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("restricted member edit: %v", err)
	}
	command = Mutation{Operation: "declare", ExpectedRevision: *seeded.Ref.Revision, URL: "https://example.test/manual", Method: "GET"}
	added, err := service.MutateAllocation(ctx, access, "declare", command)
	if err != nil {
		t.Fatal(err)
	}
	view, err := service.ReadAllocation(ctx, access, *added.Ref.Revision)
	if err != nil || len(view.Document.Groups) != 1 || view.Document.Groups[0].Route != "/manual" {
		t.Fatalf("declared safe view: %+v %v", view, err)
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
	input, err := artifactstest.ForkProjectInput(ctx, stores, "project", schema.Ref, "consumer", "schema")
	if err != nil {
		t.Fatal(err)
	}
	forged := contracts.ExactContent(schema.Ref, "application/json", []byte(`{"openapi":"3.1.0"}`))
	if _, err := service.MutateAllocation(ctx, access, "forged-project-source", Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: access.ManualSourceID, Artifact: &forged, Rules: []Rule{}}}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("manual editor gained a Project artifact selector: %v", err)
	}
	selected := readerPolicy("blackbox")
	selected.OpenAPIInputs = []string{"schema"}
	granted, err := service.AllocationAccess(ctx, "consumer", "research", selected)
	if err != nil || len(granted.Inputs) != 1 || !granted.Inputs[0].Ref.SameExact(input.TargetRef) || granted.PolicyDigest == access.PolicyDigest {
		t.Fatalf("selected input authority: %+v %v", granted, err)
	}
	selected.OpenAPIInputs = []string{"unselected"}
	if _, err := service.AllocationAccess(ctx, "consumer", "research", selected); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("missing input: %v", err)
	}
	if _, err := service.AllocationAccess(ctx, "unknown", "research", readerPolicy("blackbox")); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("foreign Run: %v", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE projects SET lifecycle_state='deleting',deletion_phase='cancelling',deletion_requested_at=clock_timestamp(),revision=revision+1,updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond') WHERE project_id='project'`); err != nil {
		t.Fatal(err)
	}
	if _, err := service.AllocationAccess(ctx, "consumer", "research", readerPolicy("blackbox")); !errors.Is(err, projectstore.ErrDeleting) {
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
			access, err := service.AllocationAccess(ctx, "consumer", "research", readerPolicy("blackbox"))
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
				saved, snapshotErr = service.SnapshotAllocation(ctx, access, "snapshot", SnapshotSelection{Revision: *original.Ref.Revision, ExpectedPolicyDigest: access.PolicyDigest, Groups: []string{view.Document.Groups[0].ID}})
			})
			wg.Go(func() {
				<-start
				_, writeErr = service.Mutate(ctx, authority, "evict", Mutation{Operation: "observe", Observations: []Observation{observation("/evict")}})
			})
			close(start)
			wg.Wait()
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
			if _, err := service.SnapshotAllocation(ctx, access, "missing", SnapshotSelection{Revision: missing, ExpectedPolicyDigest: access.PolicyDigest, Groups: []string{view.Document.Groups[0].ID}}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
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
	policy := readerPolicy("blackbox")
	policy.OpenAPIInputs = []string{"api"}
	initialAccess, err := service.AllocationAccess(ctx, "consumer", "research", policy)
	if err != nil {
		t.Fatal(err)
	}
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
	authority := Authority{OwnerID: "owner", ProjectID: "project", SourceID: "schema", Kind: "openapi", Mode: "source-assisted", RunID: "producer"}
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
	if len(access.Inputs) != 1 || !access.Inputs[0].Ref.SameExact(fork.TargetRef) || access.PolicyDigest != initialAccess.PolicyDigest {
		t.Fatal("working input advanced immutable source authority")
	}
	view, err := service.ReadAllocation(ctx, access, *published.Ref.Revision)
	if err != nil || len(view.Document.Groups) != 1 || view.Document.Groups[0].Route != "/declared" {
		t.Fatalf("working copy declassified another schema: %+v %v", view, err)
	}
	saved, err := service.SnapshotAllocation(ctx, access, "schema-context", SnapshotSelection{Revision: *published.Ref.Revision, ExpectedPolicyDigest: access.PolicyDigest, Groups: []string{view.Document.Groups[0].ID}, RequireReferencedInputs: true})
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
	full, err := service.AllocationAccess(ctx, "consumer", "ungranted", readerPolicy("source-assisted"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.SnapshotAllocation(ctx, full, "missing-required", SnapshotSelection{Revision: *published.Ref.Revision, ExpectedPolicyDigest: full.PolicyDigest, Groups: []string{view.Document.Groups[0].ID}, RequireReferencedInputs: true}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("ungranted schema materialized: %v", err)
	}
}
