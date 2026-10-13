//go:build integration

package findingintake

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func TestPostgresFindingHTTPExchangePublishesAtomically(t *testing.T) {
	for _, backend := range []string{"postgresql", "filesystem"} {
		t.Run(backend, func(t *testing.T) {
			ctx := t.Context()
			pool := isolatedFindingPool(t, ctx)
			if backend == "filesystem" {
				if err := artifacts.ClaimBlobBackend(ctx, pool, artifacts.BlobFilesystem); err != nil {
					t.Fatal(err)
				}
				files, err := artifacts.OpenFilesystemBlobStore(ctx, t.TempDir())
				if err != nil {
					t.Fatal(err)
				}
				defer files.Close()
				ctx = artifacts.WithBlobRuntime(ctx, artifacts.NewBlobRuntime(files, nil))
			}
			const owner, project, run = "http-owner", "http-project", "http-run"
			if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
				ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "HTTP evidence",
				IdempotencyKey: "http-project", RequestDigest: auditdomain.DigestBytes([]byte("http-project")),
			}); err != nil {
				t.Fatal(err)
			}
			grant := quotaSubmissionGrant(t, ctx, pool, owner, project, run)
			service, err := New(pool)
			if err != nil {
				t.Fatal(err)
			}
			var capture auditdomain.CapturedHTTPExchange
			if err := strictjson.Decode([]byte(`{"source":"http","request_id":"1","request_tag":"r-test-h000001","attempts":[{"request":{"method":"POST","url":"https://target.example/","message":{"kind":"structured","headers":[],"body_base64":"AAE="}},"response":{"status":422,"message":{"kind":"structured","headers":[],"body_base64":"/wA="},"complete":true}}]}`), &capture); err != nil {
				t.Fatal(err)
			}
			missingRevision := "missing"
			rejected := testSubmission("http-invocation", "rejected", []contracts.ArtifactRef{{Namespace: grant.Namespace, Name: "absent", Revision: &missingRevision}})
			rejected.HTTPEvidence = &capture
			if _, _, err := service.Submit(ctx, grant, rejected); !errors.Is(err, ErrAccessDenied) {
				t.Fatalf("invalid evidence: %v", err)
			}
			var count int
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind = 'run' AND scope_id = $1 AND name LIKE 'http.exchange.%'`, run).Scan(&count); err != nil || count != 0 {
				t.Fatalf("HTTP artifacts after rejection: %d, %v", count, err)
			}
			// Force a failure after the HTTP binding was created inside the
			// transaction, proving its rollback together with the proposal.
			lateRejected := testSubmission("http-invocation", "late-rejected", []contracts.ArtifactRef{})
			lateRejected.HTTPEvidence = &capture
			occupied := deterministicID("proposal", grant.AllocationID, lateRejected.InvocationID, lateRejected.SubmissionID)
			collisionPayload, err := artifacts.PreparePayload(ctx, artifacts.Payload{MediaType: "application/json", Data: []byte(`{}`)})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Create(ctx, mustRunScope(run),
				contracts.ArtifactRef{Namespace: "finding-proposals", Name: occupied}, collisionPayload); err != nil {
				t.Fatal(err)
			}
			if _, _, err := service.Submit(ctx, grant, lateRejected); !errors.Is(err, ErrConflict) {
				t.Fatalf("late proposal collision: %v", err)
			}
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind = 'run' AND scope_id = $1 AND name LIKE 'http.exchange.%'`, run).Scan(&count); err != nil || count != 0 {
				t.Fatalf("HTTP artifacts after rollback: %d, %v", count, err)
			}
			input := testSubmission("http-invocation", "selected", []contracts.ArtifactRef{})
			input.HTTPEvidence = &capture
			type result struct {
				receipt  Receipt
				replayed bool
				err      error
			}
			results := make(chan result, 2)
			var workers sync.WaitGroup
			for range 2 {
				workers.Go(func() {
					receipt, replayed, err := service.Submit(ctx, grant, input)
					results <- result{receipt, replayed, err}
				})
			}
			workers.Wait()
			close(results)
			var saved Receipt
			replays := 0
			for result := range results {
				if result.err != nil {
					t.Fatal(result.err)
				}
				if result.replayed {
					replays++
				}
				if len(result.receipt.Evidence) != 1 {
					t.Fatal("selected evidence was not pinned")
				}
				if saved.ReceiptID != "" && !saved.Evidence[0].Ref.SameExact(result.receipt.Evidence[0].Ref) {
					t.Fatal("replay published another evidence revision")
				}
				saved = result.receipt
			}
			if replays != 1 {
				t.Fatalf("replays = %d", replays)
			}
			store := artifacts.NewService(artifacts.NewPostgresRepository(pool))
			runStore, err := store.Run(run)
			if err != nil {
				t.Fatal(err)
			}
			proposalRead, err := runStore.Read(ctx, saved.Proposal.Ref)
			if err != nil {
				t.Fatal(err)
			}
			proposal, err := auditdomain.DecodeFindingProposal(proposalRead.Payload.Data)
			if err != nil || len(proposal.EvidenceIDs) != 1 || proposal.HTTPExchange == nil {
				t.Fatalf("stored finding did not reference captured evidence: %v", err)
			}
			read, err := runStore.Read(ctx, saved.Evidence[0].Ref)
			if err != nil {
				t.Fatal(err)
			}
			if read.Payload.MediaType != auditdomain.HTTPExchangeMediaType {
				t.Fatal("incorrect HTTP evidence media type")
			}
			var document map[string]any
			if err := json.Unmarshal(read.Payload.Data, &document); err != nil {
				t.Fatal(err)
			}
			association := document["finding"].(map[string]any)
			if association["proposal_id"] != saved.ProposalID || association["receipt_id"] != saved.ReceiptID {
				t.Fatal("HTTP artifact not bound to its finding")
			}
			encodedCapture, err := json.Marshal(document["capture"])
			if err != nil {
				t.Fatal(err)
			}
			var retained auditdomain.CapturedHTTPExchange
			if err := strictjson.Decode(encodedCapture, &retained); err != nil ||
				*retained.Attempts[0].Request.Message.BodyBase64 != "AAE=" ||
				*retained.Attempts[0].Response.Message.BodyBase64 != "/wA=" {
				t.Fatalf("binary capture was changed: %v", err)
			}
			metadata, err := runStore.Metadata(ctx, saved.Evidence[0].Ref)
			if err != nil || !metadata.Frozen {
				t.Fatalf("evidence not frozen: %+v, %v", metadata, err)
			}
			changed := capture
			changed.RequestTag = "r-changed"
			input.HTTPEvidence = &changed
			if _, _, err := service.Submit(ctx, grant, input); !errors.Is(err, ErrConflict) {
				t.Fatalf("changed replay: %v", err)
			}
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind = 'run' AND scope_id = $1 AND name LIKE 'http.exchange.%'`, run).Scan(&count); err != nil || count != 1 {
				t.Fatalf("HTTP artifacts after replay: %d, %v", count, err)
			}
			raw := base64.StdEncoding.EncodeToString([]byte("GET /no-response HTTP/1.1\r\nHost: target.example\r\n\r\n"))
			requestOnly := auditdomain.CapturedHTTPExchange{
				Source: "caido", RequestID: "caido-request-only",
				Attempts: []auditdomain.CapturedHTTPAttempt{{
					Request: &auditdomain.CapturedHTTPRequest{Method: "GET", URL: "https://target.example/no-response",
						Message: auditdomain.HTTPMessage{Kind: "raw", DataBase64: &raw}},
					Error: "response_unavailable",
				}},
			}
			missingResponse := testSubmission("http-invocation", "request-only", []contracts.ArtifactRef{})
			missingResponse.HTTPEvidence = &requestOnly
			selected, _, err := service.Submit(ctx, grant, missingResponse)
			if err != nil || len(selected.Evidence) != 1 {
				t.Fatalf("select request-only Caido evidence: %v", err)
			}
			requestRead, err := runStore.Read(ctx, selected.Evidence[0].Ref)
			if err != nil {
				t.Fatal(err)
			}
			var requestDocument map[string]json.RawMessage
			if err := json.Unmarshal(requestRead.Payload.Data, &requestDocument); err != nil {
				t.Fatal(err)
			}
			var requestCapture auditdomain.CapturedHTTPExchange
			if err := strictjson.Decode(requestDocument["capture"], &requestCapture); err != nil {
				t.Fatal(err)
			}
			requestAttempt := requestCapture.Attempts[0]
			if requestAttempt.Response != nil || requestAttempt.Error != "response_unavailable" ||
				*requestAttempt.Request.Message.DataBase64 != raw {
				t.Fatal("request-only evidence changed bytes or fabricated an outcome")
			}
		})
	}
}
