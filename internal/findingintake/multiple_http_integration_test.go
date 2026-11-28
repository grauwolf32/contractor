//go:build integration

package findingintake

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestPostgresMultipleHTTPFindingEvidenceLimitsReplayAndRollback(t *testing.T) {
	ctx := t.Context()
	pool := isolatedFindingPool(t, ctx)
	storage := artifactpgtest.NewStorage()
	const owner, project, run = "multi-http-owner", "multi-http-project", "multi-http-run"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Multiple HTTP", IdempotencyKey: project, RequestDigest: auditdomain.DigestBytes([]byte(project))}); err != nil {
		t.Fatal(err)
	}
	grant := quotaSubmissionGrant(t, ctx, pool, owner, project, run)
	service, err := New(pool, storage)
	if err != nil {
		t.Fatal(err)
	}
	store := artifacts.NewService(storage.Repository(pool))
	runStore, err := store.Run(run)
	if err != nil {
		t.Fatal(err)
	}
	note, err := runStore.Write(ctx, contracts.ArtifactRef{Namespace: grant.Namespace, Name: "note"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("observed context")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var capture auditdomain.CapturedHTTPExchange
	if err := strictjson.Decode([]byte(`{"source":"http","request_id":"1","request_tag":"probe","observed_at":"2026-11-06T18:20:00Z","attempts":[{"request":{"method":"GET","url":"https://target.test/","message":{"kind":"structured","headers":[],"body_base64":""}},"response":{"status":200,"message":{"kind":"structured","headers":[],"body_base64":"AAE="},"complete":true}}]}`), &capture); err != nil {
		t.Fatal(err)
	}
	selections := func(n int) []SelectedHTTPExchange {
		values := make([]SelectedHTTPExchange, n)
		for i := range values {
			c := capture
			c.RequestID = fmt.Sprint(i + 1)
			values[i] = SelectedHTTPExchange{InvocationID: fmt.Sprintf("original-%d", i), Capture: c}
		}
		return values
	}
	input := testSubmission("consumer", "mixed", []contracts.ArtifactRef{note.Ref})
	input.HTTPEvidence = selections(2)
	receipt, replayed, err := service.Submit(ctx, grant, input)
	if err != nil || replayed || len(receipt.Evidence) != 3 || len(receipt.ProjectEvidence) != 3 {
		t.Fatalf("mixed evidence: %+v %v", receipt, err)
	}
	read, err := service.GetProjectReceipt(ctx, owner, project, receipt.ReceiptID)
	if err != nil || len(read.Document.HTTPExchanges) != 2 {
		t.Fatalf("Project read: %+v %v", read.Document, err)
	}
	for index, selection := range input.HTTPEvidence {
		artifact, err := runStore.Read(ctx, receipt.Evidence[index+1].Ref)
		if err != nil {
			t.Fatal(err)
		}
		envelope, err := auditdomain.DecodeHTTPExchangeArtifact(artifact.Payload.Data)
		if err != nil || envelope.Origin.InvocationID != selection.InvocationID || envelope.Capture.RequestID != selection.Capture.RequestID {
			t.Fatalf("origin: %+v %v", envelope, err)
		}
		if read.Document.HTTPExchanges[index].ResponseBodyEvidenceID != fmt.Sprintf("evidence-%d", index+2) {
			t.Fatal("wrong response evidence link")
		}
	}
	replay, ok, err := service.Submit(ctx, grant, input)
	if err != nil || !ok || !reflect.DeepEqual(replay.Evidence, receipt.Evidence) {
		t.Fatal("replay changed evidence", err)
	}
	changed := input
	changed.HTTPEvidence = selections(2)
	changed.HTTPEvidence[1].Capture.RequestTag = "changed"
	if _, _, err := service.Submit(ctx, grant, changed); !errors.Is(err, ErrConflict) {
		t.Fatal("changed second capture replay", err)
	}
	// Reject duplicate identities before publication, including different content.
	duplicate := testSubmission("consumer", "duplicates", nil)
	duplicate.HTTPEvidence = selections(2)
	duplicate.HTTPEvidence[1] = duplicate.HTTPEvidence[0]
	duplicate.HTTPEvidence[1].Capture.RequestTag = "different"
	if _, _, err := service.Submit(ctx, grant, duplicate); !errors.Is(err, ErrInvalid) {
		t.Fatal("duplicate capture accepted", err)
	}
	// The first new exchange must roll back if the second collides with a frozen binding.
	rollback := testSubmission("consumer", "rollback", nil)
	rollback.HTTPEvidence = selections(2)
	rollback.HTTPEvidence[0].InvocationID = "new-first"
	rollback.HTTPEvidence[1].Capture.RequestTag = "collision"
	if _, _, err := service.Submit(ctx, grant, rollback); !errors.Is(err, ErrConflict) {
		t.Fatal("second collision", err)
	}
	name := auditdomain.HTTPExchangeArtifactName("new-first", "http", "1")
	if _, err := runStore.Metadata(ctx, contracts.ArtifactRef{Namespace: "http-exchanges", Name: name}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatal("partial exchange survived", err)
	}
	var count int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM finding_proposal_receipts WHERE run_id=$1", run).Scan(&count); err != nil || count != 1 {
		t.Fatal("partial receipt", count, err)
	}
	// Exercise actual storage at the total source ceiling, not only a validator.
	maximum := testSubmission("consumer", "maximum", []contracts.ArtifactRef{note.Ref})
	maximum.HTTPEvidence = selections(MaxEvidenceRefs - 1)
	if saved, _, err := service.Submit(ctx, grant, maximum); err != nil || len(saved.Evidence) != MaxEvidenceRefs {
		t.Fatalf("256 mixed evidence: %d %v", len(saved.Evidence), err)
	}
	maximum.HTTPEvidence = selections(MaxEvidenceRefs)
	if _, _, err := service.Submit(ctx, grant, maximum); !errors.Is(err, ErrInvalid) {
		t.Fatal("257 total evidence accepted", err)
	}
	fact := testSubmission("consumer", "fact-maximum", []contracts.ArtifactRef{note.Ref})
	fact.Proposal.SeveritySuggestion = "informational"
	fact.HTTPEvidence = selections(3)
	if saved, _, err := service.Submit(ctx, grant, fact); err != nil || len(saved.Evidence) != 4 {
		t.Fatal("four fact evidence", err)
	}
	fact.HTTPEvidence = selections(4)
	if _, _, err := service.Submit(ctx, grant, fact); !errors.Is(err, ErrInvalid) {
		t.Fatal("five fact evidence accepted", err)
	}
}
