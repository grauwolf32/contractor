//go:build integration

package findingintake

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

type projectCollectionReads struct{ emptyCollectionReviews }

func (projectCollectionReads) ReadCollectionFindings(ctx context.Context, tx pgx.Tx, owner string, ids []string) (map[string]*auditdomain.FindingCollectionFinding, error) {
	return querystore.New(tx, artifactpgtest.NewStorage()).CollectionFindings(ctx, owner, ids)
}

func TestProjectFindingAllocationReaderSelectsExactInputsAndReplaysAfterDeletion(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	storage := artifactpgtest.NewStorage()
	const owner, project = "reader-owner", "reader-project"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Reader", IdempotencyKey: "reader", RequestDigest: auditdomain.DigestBytes([]byte("reader"))}); err != nil {
		t.Fatal(err)
	}
	producer := quotaSubmissionGrant(t, ctx, pool, owner, project, "reader-producer")
	consumer := quotaSubmissionGrant(t, ctx, pool, owner, project, "reader-consumer")
	consumer.ReadPolicy = controlplane.ReadCurrentRun
	consumer.WritePolicy = controlplane.WriteInputsAndIntermediates
	consumer.FindingTools = []string{"list_project_findings", "read_project_finding", "select_project_findings"}
	store := artifacts.NewService(storage.Repository(pool))
	run, _ := store.Run(producer.RunID)
	written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "evidence", Name: "proof"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("exact original proof")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	intake, err := New(pool, storage)
	if err != nil {
		t.Fatal(err)
	}
	var receipts []Receipt
	for i := range 101 {
		input := testSubmission("reader-invocation", fmt.Sprintf("problem-%03d", i), []contracts.ArtifactRef{written.Ref})
		input.PublishProblem = true
		receipt, _, err := intake.Submit(ctx, producer, input)
		if err != nil || receipt.FindingID == nil {
			t.Fatalf("publication %d: %+v %v", i, receipt, err)
		}
		receipts = append(receipts, receipt)
	}
	publisher, err := NewCollectionPublisher(pool, projectCollectionReads{}, storage)
	if err != nil {
		t.Fatal(err)
	}
	countBindings := func() int {
		t.Helper()
		var count int
		if err := pool.QueryRow(ctx, "SELECT count(*) FROM artifact_bindings").Scan(&count); err != nil {
			t.Fatal(err)
		}
		return count
	}
	before := countBindings()
	first, err := publisher.ListAllocationFindings(ctx, consumer, 100, "")
	if err != nil || len(first.Items) != 100 || first.NextCursor == nil {
		t.Fatalf("first page: %+v %v", first, err)
	}
	second, err := publisher.ListAllocationFindings(ctx, consumer, 100, *first.NextCursor)
	if err != nil || len(second.Items) != 1 || second.NextCursor != nil {
		t.Fatalf("second page: %+v %v", second, err)
	}
	seen := map[string]bool{}
	for _, row := range append(first.Items, second.Items...) {
		if seen[row.FindingID] {
			t.Fatal("duplicate page row")
		}
		seen[row.FindingID] = true
	}
	selected := first.Items[0]
	detail, err := publisher.ReadAllocationFinding(ctx, consumer, selected.FindingID, selected.Revision)
	if err != nil || detail.FindingID != selected.FindingID {
		t.Fatal(detail, err)
	}
	if countBindings() != before {
		t.Fatal("live reads retained artifacts")
	}
	if _, err := publisher.ReadAllocationFinding(ctx, consumer, selected.FindingID, selected.Revision+1); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale read: %v", err)
	}
	missing := consumer
	missing.FindingTools = nil
	if _, err := publisher.ListAllocationFindings(ctx, missing, 100, ""); !errors.Is(err, ErrAccessDenied) {
		t.Fatalf("unselected reader: %v", err)
	}
	fenced := consumer
	fenced.WriteFenced = true
	if _, err := publisher.ListAllocationFindings(ctx, fenced, 100, ""); !errors.Is(err, ErrAccessDenied) {
		t.Fatalf("fenced reader: %v", err)
	}
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "foreign-reader-project", OwnerID: owner, Kind: projectstore.KindProject, Name: "Foreign", IdempotencyKey: "foreign-reader", RequestDigest: auditdomain.DigestBytes([]byte("foreign-reader"))}); err != nil {
		t.Fatal(err)
	}
	foreign := quotaSubmissionGrant(t, ctx, pool, owner, "foreign-reader-project", "foreign-reader")
	foreign.ReadPolicy = consumer.ReadPolicy
	foreign.WritePolicy = consumer.WritePolicy
	foreign.FindingTools = consumer.FindingTools
	if _, err := publisher.ReadAllocationFinding(ctx, foreign, selected.FindingID, selected.Revision); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-Project detail: %v", err)
	}
	if _, err := publisher.ListAllocationFindings(ctx, foreign, 100, *first.NextCursor); !errors.Is(err, ErrInvalid) {
		t.Fatalf("cross-Project cursor: %v", err)
	}
	if _, err := publisher.SelectAllocationFindings(ctx, foreign, FindingSelection{ClientKey: "foreign", Findings: []CollectionFindingSelection{{FindingID: selected.FindingID, Revision: selected.Revision}}, ReceiptIDs: []string{}}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-Project selection: %v", err)
	}
	before = countBindings()

	input := FindingSelection{ClientKey: "chosen", Findings: []CollectionFindingSelection{{FindingID: selected.FindingID, Revision: selected.Revision}}, ReceiptIDs: []string{}}
	invalid := input
	invalid.ClientKey = "stale"
	invalid.Findings = []CollectionFindingSelection{{FindingID: selected.FindingID, Revision: selected.Revision + 1}}
	if _, err := publisher.SelectAllocationFindings(ctx, consumer, invalid); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale selection: %v", err)
	}
	if countBindings() != before {
		t.Fatal("failed selection published partial data")
	}
	snapshot, err := publisher.SelectAllocationFindings(ctx, consumer, input)
	if err != nil || snapshot.Replayed || snapshot.EntryCount != 1 {
		t.Fatal(snapshot, err)
	}
	var versions int
	if err := pool.QueryRow(ctx, `SELECT count(DISTINCT version_id) FROM artifact_binding_revisions WHERE scope_kind='run' AND scope_id=$1 AND name='chosen' AND namespace=ANY($2::text[])`, consumer.RunID, []string{CollectionNamespace, consumer.Namespace}).Scan(&versions); err != nil || versions != 1 {
		t.Fatalf("Stage output fork copied package version: %d %v", versions, err)
	}
	current, _ := store.Run(consumer.RunID)
	bytes, err := current.Read(ctx, snapshot.Artifact.Ref)
	if err != nil {
		t.Fatal(err)
	}
	collection, pkg, err := auditdomain.DecodeFindingCollectionPackage(bytes.Payload.Data)
	if err != nil || len(collection.Entries) != 1 || collection.Entries[0].Finding == nil || collection.Entries[0].Finding.FindingID != selected.FindingID {
		t.Fatalf("retained selection: %+v %v", collection, err)
	}
	evidence := collection.Entries[0].Evidence[0]
	member, ok := pkg.MemberByID(evidence.DocumentID)
	if !ok || string(member.Data()) != "exact original proof" {
		t.Fatal("snapshot changed evidence bytes")
	}
	if _, err := current.Write(ctx, contracts.ArtifactRef{Namespace: consumer.Namespace, Name: "occupied"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("existing output")}, nil); err != nil {
		t.Fatal(err)
	}
	occupied := input
	occupied.ClientKey = "occupied"
	if _, err := publisher.SelectAllocationFindings(ctx, consumer, occupied); !errors.Is(err, ErrConflict) {
		t.Fatalf("Stage output collision: %v", err)
	}
	if _, err := current.Metadata(ctx, contracts.ArtifactRef{Namespace: CollectionNamespace, Name: "occupied"}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("collision retained partial snapshot: %v", err)
	}
	hidden := input
	hidden.ClientKey = "memory.reserved"
	if _, err := publisher.SelectAllocationFindings(ctx, consumer, hidden); !errors.Is(err, ErrAccessDenied) {
		t.Fatalf("selection claimed a Memory binding: %v", err)
	}

	// A source deletion is real; replay must use the published exact snapshot.
	runs := runstore.NewPostgresStore(pool, storage)
	if err := runs.MarkStageAllocationReleased(ctx, producer.AllocationID); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, producer.RunID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "fixture_finished"}); err != nil {
		t.Fatal(err)
	}
	if err := deleteRun(ctx, pool, owner, producer.RunID); err != nil {
		t.Fatal(err)
	}
	publisher, err = NewCollectionPublisher(pool, projectCollectionReads{}, storage)
	if err != nil {
		t.Fatal(err)
	}
	replay, err := publisher.SelectAllocationFindings(ctx, consumer, input)
	if err != nil || !replay.Replayed || !reflect.DeepEqual(replay.Artifact, snapshot.Artifact) || replay.SnapshotAt != snapshot.SnapshotAt {
		t.Fatalf("replay after response loss/source deletion: %+v %v", replay, err)
	}
	input.Findings = []CollectionFindingSelection{}
	if _, err := publisher.SelectAllocationFindings(ctx, consumer, input); !errors.Is(err, ErrCollectionKeyConflict) {
		t.Fatalf("changed selection: %v", err)
	}
	if len(receipts) != 101 {
		t.Fatal("fixture did not publish every problem")
	}
}

func TestProjectFindingAllocationReaderSharesAuditAndRunSources(t *testing.T) {
	ctx := t.Context()
	pool := isolatedFindingPool(t, ctx)
	storage := artifactpgtest.NewStorage()
	const owner, project = "shared-reader-owner", "shared-reader-project"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Shared sources", IdempotencyKey: "shared-reader", RequestDigest: auditdomain.DigestBytes([]byte("shared-reader"))}); err != nil {
		t.Fatal(err)
	}
	// A reduced engagement has no accessMode. The former SQL mode lookup
	// classified its presence as invalid and denied the authorized reader.
	baseline := func(document map[string]any) {
		origin := "https://target.example"
		document["engagement"] = map[string]any{"document": engagementcontract.Document{
			Schema:     engagementcontract.Schema,
			Egress:     engagementcontract.Egress{Routes: []engagementcontract.Route{{Origin: origin, PathPrefixes: []string{"/"}, Methods: []string{"GET"}, AddressRanges: []string{"192.0.2.0/24"}}}, Exclusions: []engagementcontract.Route{}},
			Identities: []engagementcontract.Identity{{ID: "anonymous", Kind: "anonymous", Origins: []string{origin}}},
			Limits:     &engagementcontract.HTTPLimits{},
		}}
	}
	consumer, _ := auditSubmissionGrantWithBaseline(t, ctx, pool, owner, project, "shared-consumer", baseline)
	consumer.ReadPolicy = controlplane.ReadCurrentRun
	consumer.WritePolicy = controlplane.WriteInputsAndIntermediates
	consumer.FindingTools = []string{"list_project_findings", "read_project_finding", "select_project_findings"}
	known, _ := auditSubmissionGrantWithBaseline(t, ctx, pool, owner, project, "first-audit-source", baseline)
	assisted, _ := auditSubmissionGrantWithBaseline(t, ctx, pool, owner, project, "second-audit-source", baseline)
	unknown := quotaSubmissionGrant(t, ctx, pool, owner, project, "unknown-source")
	intake, err := New(pool, storage)
	if err != nil {
		t.Fatal(err)
	}
	publisher, err := NewCollectionPublisher(pool, projectCollectionReads{}, storage)
	if err != nil {
		t.Fatal(err)
	}
	var receipts []Receipt
	for index, grant := range []controlplane.AllocationGrant{known, assisted, unknown} {
		run, _ := artifacts.NewService(storage.Repository(pool)).Run(grant.RunID)
		proof, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: "proof"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("exact proof")}, nil)
		if err != nil {
			t.Fatal(err)
		}
		input := testSubmission("source-invocation", fmt.Sprintf("problem-%d", index), []contracts.ArtifactRef{proof.Ref})
		input.PublishProblem = true
		receipt, _, err := intake.Submit(ctx, grant, input)
		if err != nil || receipt.FindingID == nil {
			t.Fatal(receipt, err)
		}
		receipts = append(receipts, receipt)
	}
	if value, err := publisher.ListAllocationFindings(ctx, consumer, 100, ""); err != nil || len(value.Items) != 3 {
		t.Fatalf("shared Project sources: %+v %v", value, err)
	}
	for index, receipt := range receipts {
		_, err := publisher.ReadAllocationFinding(ctx, consumer, *receipt.FindingID, 1)
		if err != nil {
			t.Fatalf("source %d read: %v", index, err)
		}
	}
	selection := FindingSelection{ClientKey: "known", Findings: []CollectionFindingSelection{{FindingID: *receipts[0].FindingID, Revision: 1}}, ReceiptIDs: []string{}}
	if snapshot, err := publisher.SelectAllocationFindings(ctx, consumer, selection); err != nil || snapshot.EntryCount != 1 {
		t.Fatalf("first source selection: %+v %v", snapshot, err)
	}
	selection.ClientKey = "mixed-sources"
	selection.Findings = append(selection.Findings, CollectionFindingSelection{FindingID: *receipts[1].FindingID, Revision: 1})
	if snapshot, err := publisher.SelectAllocationFindings(ctx, consumer, selection); err != nil || snapshot.EntryCount != 2 {
		t.Fatalf("mixed source selection: %+v %v", snapshot, err)
	}
	// Explicit owner selection uses the same ordinary Project source authority.
	snapshot, err := publisher.PublishCollection(ctx, PublishCollectionParams{OwnerID: owner, Request: PublishCollectionRequest{ClientKey: "owner-selection", Sources: []CollectionSelection{{Kind: "project", ID: project, Findings: selection.Findings, ReceiptIDs: []string{}}}}})
	if err != nil || snapshot.EntryCount != 2 {
		t.Fatalf("direct owner selection: %+v %v", snapshot, err)
	}
}
