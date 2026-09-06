//go:build integration

package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgxpool"
)

func collectionTestPool(t *testing.T) (context.Context, *pgxpool.Pool) {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required for the selected findings collection integration test")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	return ctx, isolatedAuditServicePool(t, ctx, url)
}

func newCollectionTestPublisher(t *testing.T, pool *pgxpool.Pool) *findingintake.CollectionPublisher {
	t.Helper()
	publisher, err := findingintake.NewCollectionPublisher(pool, &Service{pool: pool})
	if err != nil {
		t.Fatal(err)
	}
	return publisher
}

func seedCollectionRun(t *testing.T, ctx context.Context, pool *pgxpool.Pool, owner, runID string) {
	t.Helper()
	snapshot, err := config.Load("../config/testdata/valid", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	_, err = runstore.NewPostgresStore(pool).CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: owner, WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: encoded, Parameters: map[string]string{}, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	})
	if err != nil {
		t.Fatal(err)
	}
}

func seedCollectionRunReceipt(t *testing.T, ctx context.Context, pool *pgxpool.Pool, owner, runID, suffix string, missingEvidence bool) {
	t.Helper()
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	run, err := service.Run(runID)
	if err != nil {
		t.Fatal(err)
	}
	write, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "evidence", Name: "shared"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("same evidence")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	evidence := findingintake.ExactArtifact{Ref: write.Ref, Digest: digestBytes([]byte("same evidence")), MediaType: write.MediaType, SizeBytes: write.Size}
	if missingEvidence {
		evidence.Ref.Name = "missing"
	}
	proposal := auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: suffix, Title: suffix, Description: "A generic observation.",
		Subject: auditdomain.FindingSubject{Kind: "function", Key: "shared"}, Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
		EvidenceIDs: []string{"evidence-1"}, ProposedChecks: []auditdomain.ProposedCheck{}, Limitations: []string{},
	}
	body, err := auditdomain.EncodeFindingProposal(proposal)
	if err != nil {
		t.Fatal(err)
	}
	written, err := service.WriteFindingProposal(ctx, runID, suffix, artifacts.Payload{MediaType: "application/json", Data: body})
	if err != nil {
		t.Fatal(err)
	}
	ref, _ := json.Marshal(written.Ref)
	evidenceJSON, _ := json.Marshal([]findingintake.ExactArtifact{evidence})
	_, err = pool.Exec(ctx, `INSERT INTO finding_proposal_receipts (
receipt_id, proposal_id, allocation_id, runtime_agent_id, runtime_instance_id, stage_execution_id, logical_agent_name,
invocation_id, submission_id, client_key, request_digest, run_id, owner_id, workflow_name, workflow_version,
workflow_schema_version, workflow_configuration_ref, workflow_closure_digest, proposal_ref, proposal_digest, proposal_media_type, proposal_size_bytes, evidence
) VALUES ($1,$2,$3,'runtime','instance','stage','worker',$4,$5,$6,$7,$8,$9,'source','1','contractor/v1alpha1',
'{"name":"source","version":"1"}',$7,$10,$11,'application/json',$12,$13)`,
		"receipt-"+suffix, "proposal-"+suffix, "allocation-"+suffix, "invocation-"+suffix, "submission-"+suffix, suffix,
		serviceTestDigest(suffix), runID, owner, ref, digestBytes(body), written.Size, evidenceJSON)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_retention(receipt_id,state) VALUES($1,'source-held')`, "receipt-"+suffix); err != nil {
		t.Fatal(err)
	}
}

func TestFindingCollectionPublicationRunRetentionAndReplay(t *testing.T) {
	ctx, pool := collectionTestPool(t)
	const owner = "collection-owner"
	for _, suffix := range []string{"a", "b"} {
		seedCollectionRun(t, ctx, pool, owner, "run-"+suffix)
		seedCollectionRunReceipt(t, ctx, pool, owner, "run-"+suffix, suffix, false)
	}
	publisher := newCollectionTestPublisher(t, pool)
	params := findingintake.PublishCollectionParams{OwnerID: owner, Request: findingintake.PublishCollectionRequest{ClientKey: "snapshot", Sources: []findingintake.CollectionSelection{
		{Kind: "run", ID: "run-b", ReceiptIDs: []string{"receipt-b"}, Findings: []findingintake.CollectionFindingSelection{}},
		{Kind: "run", ID: "run-a", ReceiptIDs: []string{"receipt-a"}, Findings: []findingintake.CollectionFindingSelection{}},
	}}}
	var results [2]findingintake.PublishedCollection
	var errs [2]error
	var wg sync.WaitGroup
	for i := range 2 {
		wg.Add(1)
		go func() { defer wg.Done(); results[i], errs[i] = publisher.PublishCollection(ctx, params) }()
	}
	wg.Wait()
	if errs[0] != nil || errs[1] != nil || results[0].Replayed == results[1].Replayed || results[0].Artifact.Digest != results[1].Artifact.Digest {
		t.Fatalf("concurrent publication: results=%+v errors=%v", results, errs)
	}
	result := results[0]
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	user, _ := service.User(owner)
	read, err := user.Read(ctx, result.Artifact.Ref)
	if err != nil {
		t.Fatal(err)
	}
	collection, _, err := auditdomain.DecodeFindingCollectionPackage(read.Payload.Data)
	if err != nil || len(collection.Entries) != 2 || len(collection.Documents) != 4 {
		t.Fatalf("collection = %+v %v", collection, err)
	}
	for _, entry := range collection.Entries {
		if len(entry.Reviews) != 0 || entry.AuditOrigin != nil {
			t.Fatal("ordinary Run proposal acquired a review")
		}
	}
	if collection.Entries[0].Evidence[0].DocumentID == collection.Entries[1].Evidence[0].DocumentID {
		t.Fatal("source bindings collided")
	}
	seedCollectionRun(t, ctx, pool, owner, "reader")
	fork, err := service.ForkInput(ctx, owner, result.Artifact.Ref, "reader", "findings")
	if err != nil {
		t.Fatal(err)
	}
	// Advance the source binding and add a proposal after publication. Neither
	// changes the selection or the retained exact document bytes.
	sourceRun, _ := service.Run("run-a")
	previous, err := sourceRun.Read(ctx, contracts.ArtifactRef{Namespace: "evidence", Name: "shared"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sourceRun.Write(ctx, contracts.ArtifactRef{Namespace: "evidence", Name: "shared"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("later evidence")}, previous.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	seedCollectionRun(t, ctx, pool, owner, "run-later")
	seedCollectionRunReceipt(t, ctx, pool, owner, "run-later", "later", false)
	runs := runstore.NewPostgresStore(pool)
	for _, id := range []string{"run-a", "run-b"} {
		if _, err := runs.TransitionRun(ctx, id, runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "test_done"}); err != nil {
			t.Fatal(err)
		}
		if err := runs.DeleteReleasedTerminalRun(ctx, owner, id); err != nil {
			t.Fatal(err)
		}
	}
	reader, _ := service.Run("reader")
	retained, err := reader.Read(ctx, fork.TargetRef)
	if err != nil || digestBytes(retained.Payload.Data) != result.Artifact.Digest {
		t.Fatalf("consumer ZIP after deletion: %v", err)
	}
	replay, err := publisher.PublishCollection(ctx, params)
	if err != nil || !replay.Replayed || replay.Artifact.Digest != result.Artifact.Digest || replay.EntryCount != 2 {
		t.Fatalf("replay after deletion: %+v %v", replay, err)
	}
	params.Request.Sources = params.Request.Sources[:1]
	if _, err := publisher.PublishCollection(ctx, params); !errors.Is(err, findingintake.ErrConflict) {
		t.Fatalf("changed selection = %v", err)
	}
}

func TestFindingCollectionPublicationRejectsPartialForeignAndInterruptedWrites(t *testing.T) {
	ctx, pool := collectionTestPool(t)
	seedCollectionRun(t, ctx, pool, "owner", "run-good")
	seedCollectionRunReceipt(t, ctx, pool, "owner", "run-good", "good", false)
	seedCollectionRun(t, ctx, pool, "foreign", "run-foreign")
	seedCollectionRunReceipt(t, ctx, pool, "foreign", "run-foreign", "foreign", false)
	seedCollectionRun(t, ctx, pool, "owner", "run-missing")
	seedCollectionRunReceipt(t, ctx, pool, "owner", "run-missing", "missing", true)
	publisher := newCollectionTestPublisher(t, pool)
	user, _ := artifacts.NewService(artifacts.NewPostgresRepository(pool)).User("owner")
	for _, tc := range []struct{ key, run, receipt string }{
		{"foreign-source", "run-foreign", "receipt-foreign"},
		{"foreign-receipt", "run-good", "receipt-foreign"},
		{"missing-bytes", "run-missing", "receipt-missing"},
		{"unknown-receipt", "run-good", "receipt-unknown"},
	} {
		t.Run(tc.key, func(t *testing.T) {
			sources := []findingintake.CollectionSelection{
				{Kind: "run", ID: "run-good", ReceiptIDs: []string{"receipt-good"}, Findings: []findingintake.CollectionFindingSelection{}},
			}
			if tc.run == "run-good" {
				sources[0].ReceiptIDs = append(sources[0].ReceiptIDs, tc.receipt)
			} else {
				sources = append(sources, findingintake.CollectionSelection{Kind: "run", ID: tc.run, ReceiptIDs: []string{tc.receipt}, Findings: []findingintake.CollectionFindingSelection{}})
			}
			_, err := publisher.PublishCollection(ctx, findingintake.PublishCollectionParams{OwnerID: "owner", Request: findingintake.PublishCollectionRequest{ClientKey: tc.key, Sources: sources}})
			if !errors.Is(err, findingintake.ErrNotFound) && !errors.Is(err, artifacts.ErrArtifactNotFound) {
				t.Fatalf("inaccessible selection error = %v", err)
			}
			if _, err := user.Read(ctx, contracts.ArtifactRef{Namespace: findingintake.CollectionNamespace, Name: tc.key}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
				t.Fatalf("partial ZIP is visible: %v", err)
			}
		})
	}
	if _, err := pool.Exec(ctx, `CREATE FUNCTION fail_collection_receipt() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
IF NEW.namespace='finding-collection-receipts' THEN RAISE EXCEPTION 'injected receipt write failure'; END IF; RETURN NEW; END $$;
CREATE TRIGGER fail_collection_receipt BEFORE INSERT ON artifact_bindings FOR EACH ROW EXECUTE FUNCTION fail_collection_receipt()`); err != nil {
		t.Fatal(err)
	}
	params := findingintake.PublishCollectionParams{OwnerID: "owner", Request: findingintake.PublishCollectionRequest{ClientKey: "interrupted", Sources: []findingintake.CollectionSelection{
		{Kind: "run", ID: "run-good", ReceiptIDs: []string{"receipt-good"}, Findings: []findingintake.CollectionFindingSelection{}},
	}}}
	if _, err := publisher.PublishCollection(ctx, params); err == nil {
		t.Fatal("injected publication failure succeeded")
	}
	if _, err := user.Read(ctx, contracts.ArtifactRef{Namespace: findingintake.CollectionNamespace, Name: "interrupted"}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("rolled-back ZIP visible: %v", err)
	}
	if _, err := pool.Exec(ctx, `DROP TRIGGER fail_collection_receipt ON artifact_bindings`); err != nil {
		t.Fatal(err)
	}
	if result, err := publisher.PublishCollection(ctx, params); err != nil || result.Replayed || result.EntryCount != 1 {
		t.Fatalf("retry after rollback: %+v %v", result, err)
	}
	params.Request.ClientKey = "empty"
	params.Request.Sources[0].ReceiptIDs = []string{}
	if result, err := publisher.PublishCollection(ctx, params); err != nil || result.EntryCount != 0 {
		t.Fatalf("explicit empty selection: %+v %v", result, err)
	}
}

func TestFindingCollectionAuditContributionsAndPinnedReview(t *testing.T) {
	ctx, pool := collectionTestPool(t)
	const owner, projectID, auditID = "owner", "project", "audit"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: projectID, OwnerID: owner, Kind: projectstore.KindProject, Name: "Collection", IdempotencyKey: "project", RequestDigest: serviceTestDigest("project")}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := auditstore.NewPostgresStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: owner, ProjectID: projectID, Profile: auditstore.ProfileIdentity{Name: "review", Version: "1", Digest: serviceTestDigest("profile")},
		ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`), InputSelection: json.RawMessage(`{}`),
		Limits:         auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 2, MaxItemsTotal: 2, MaxSubmittedRuns: 2, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20},
		IdempotencyKey: "audit", RequestDigest: serviceTestDigest("audit"),
	}); err != nil {
		t.Fatal(err)
	}
	first := seedAuditFinding(t, ctx, pool, projectID, owner, auditID, "first")
	second := seedAuditFinding(t, ctx, pool, projectID, owner, auditID, "second")
	// Model the existing contribution relation: one finding has two source
	// receipts, including one that is not its FirstProposal.
	if _, err := pool.Exec(ctx, `DELETE FROM audit_findings WHERE finding_id=$1`, second); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO audit_finding_contributions(finding_id,audit_id,receipt_id,relation,proposal_ref)
SELECT $1,$2,receipt_id,'contributing',proposal_ref FROM finding_proposal_audit_holds WHERE receipt_id='receipt-second' AND audit_id=$2`, first, auditID); err != nil {
		t.Fatal(err)
	}
	publisher := newCollectionTestPublisher(t, pool)
	params := findingintake.PublishCollectionParams{OwnerID: owner, Request: findingintake.PublishCollectionRequest{ClientKey: "audit-snapshot", Sources: []findingintake.CollectionSelection{
		{Kind: "audit", ID: auditID, ReceiptIDs: []string{"receipt-second"}, Findings: []findingintake.CollectionFindingSelection{{FindingID: first, Revision: 1}}},
	}}}
	result, err := publisher.PublishCollection(ctx, params)
	if err != nil || result.EntryCount != 2 {
		t.Fatalf("all contributions: %+v %v", result, err)
	}
	user, _ := artifacts.NewService(artifacts.NewPostgresRepository(pool)).User(owner)
	read, err := user.Read(ctx, result.Artifact.Ref)
	if err != nil {
		t.Fatal(err)
	}
	value, _, err := auditdomain.DecodeFindingCollectionPackage(read.Payload.Data)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range value.Entries {
		if entry.Retention != "audit-held" || len(entry.Reviews) != 1 || entry.Reviews[0].FindingID != first || entry.Reviews[0].Revision != 1 || entry.Reviews[0].DecisionID != "" {
			t.Fatalf("captured provenance: %+v", entry)
		}
	}
	for _, document := range value.Documents {
		if document.Scope.Kind != "project" {
			t.Fatalf("deleted Run retained source: %+v", document)
		}
	}
	if _, err := pool.Exec(ctx, `UPDATE audit_findings SET state='needs-evidence',revision=revision+1 WHERE finding_id=$1`, first); err != nil {
		t.Fatal(err)
	}
	replay, err := publisher.PublishCollection(ctx, params)
	if err != nil || !replay.Replayed || replay.Artifact.Digest != result.Artifact.Digest {
		t.Fatalf("review change refreshed snapshot: %+v %v", replay, err)
	}
	params.Request.ClientKey = "stale-review"
	if _, err := publisher.PublishCollection(ctx, params); !errors.Is(err, findingintake.ErrConflict) {
		t.Fatalf("stale finding revision = %v", err)
	}
	params.Request.Sources[0].Findings[0].Revision = 2
	if _, err := publisher.PublishCollection(ctx, params); err != nil {
		t.Fatalf("explicit new review revision: %v", err)
	}
}
