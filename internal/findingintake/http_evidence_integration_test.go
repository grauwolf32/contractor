//go:build integration

package findingintake

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends/filesystem"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestPostgresFindingHTTPExchangePublishesAtomically(t *testing.T) {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	for _, backend := range []string{"postgresql", "filesystem"} {
		t.Run(backend, func(t *testing.T) {
			ctx := t.Context()
			pool := isolatedFindingPool(t, ctx)
			if backend == "filesystem" {
				files, err := filesystem.Open(ctx, t.TempDir())
				if err != nil {
					t.Fatal(err)
				}
				defer files.Close()
				artifactRuntime = blob.NewRuntime(files, nil)
				artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
			}
			const owner, project, run = "http-owner", "http-project", "http-run"
			if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
				ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "HTTP evidence",
				IdempotencyKey: "http-project", RequestDigest: auditdomain.DigestBytes([]byte("http-project")),
			}); err != nil {
				t.Fatal(err)
			}
			grant := quotaSubmissionGrant(t, ctx, pool, owner, project, run)
			service, err := New(pool, artifactStorage)
			if err != nil {
				t.Fatal(err)
			}
			var capture auditdomain.CapturedHTTPExchange
			if err := strictjson.Decode([]byte(`{"source":"http","request_id":"1","request_tag":"r-test-h000001","observed_at":"2026-11-06T18:20:00Z","attempts":[{"request":{"method":"POST","url":"https://target.example/","message":{"kind":"structured","headers":[],"body_base64":"AAE="}},"response":{"status":422,"message":{"kind":"structured","headers":[],"body_base64":"/wA="},"complete":true}}]}`), &capture); err != nil {
				t.Fatal(err)
			}
			missingRevision := "missing"
			rejected := testSubmission("http-invocation", "rejected", []contracts.ArtifactRef{{Namespace: grant.Namespace, Name: "absent", Revision: &missingRevision}})
			rejected.HTTPEvidence = &capture
			rejected.HTTPEvidenceInvocationID = rejected.InvocationID
			if _, _, err := service.Submit(ctx, grant, rejected); !errors.Is(err, ErrAccessDenied) {
				t.Fatalf("invalid evidence: %v", err)
			}
			var count int
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = 'http-exchanges'`, run).Scan(&count); err != nil || count != 0 {
				t.Fatalf("HTTP artifacts after rejection: %d, %v", count, err)
			}
			// Force a failure after the HTTP binding was created inside the
			// transaction, proving its rollback together with the proposal.
			lateRejected := testSubmission("http-invocation", "late-rejected", []contracts.ArtifactRef{})
			lateRejected.HTTPEvidence = &capture
			lateRejected.HTTPEvidenceInvocationID = lateRejected.InvocationID
			occupied := deterministicID("proposal", grant.AllocationID, lateRejected.InvocationID, lateRejected.SubmissionID)
			collisionPayload, err := artifactStorage.PreparePayload(ctx, pool, artifacts.Payload{MediaType: "application/json", Data: []byte(`{}`)})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := artifacts.NewService(artifactStorage.Repository(pool)).Create(ctx, mustRunScope(run),
				contracts.ArtifactRef{Namespace: "finding-proposals", Name: occupied}, collisionPayload); err != nil {
				t.Fatal(err)
			}
			if _, _, err := service.Submit(ctx, grant, lateRejected); !errors.Is(err, ErrConflict) {
				t.Fatalf("late proposal collision: %v", err)
			}
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = 'http-exchanges'`, run).Scan(&count); err != nil || count != 0 {
				t.Fatalf("HTTP artifacts after rollback: %d, %v", count, err)
			}
			input := testSubmission("http-invocation", "selected", []contracts.ArtifactRef{})
			input.HTTPEvidence = &capture
			input.HTTPEvidenceInvocationID = input.InvocationID
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
			store := artifacts.NewService(artifactStorage.Repository(pool))
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
			association := document["origin"].(map[string]any)
			if association["run_id"] != run || association["invocation_id"] != input.InvocationID || document["finding"] != nil {
				t.Fatal("HTTP artifact has incorrect independent origin")
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
			duplicate := testSubmission("http-invocation", "duplicate-selection", []contracts.ArtifactRef{saved.Evidence[0].Ref})
			duplicate.HTTPEvidence = &capture
			duplicate.HTTPEvidenceInvocationID = duplicate.InvocationID
			if _, _, err := service.Submit(ctx, grant, duplicate); !errors.Is(err, ErrInvalid) {
				t.Fatalf("duplicate raw/exact selection: %v", err)
			}
			reused := testSubmission("later-invocation", "reuse-selected", []contracts.ArtifactRef{saved.Evidence[0].Ref})
			reusedReceipt, _, err := service.Submit(ctx, grant, reused)
			if err != nil || len(reusedReceipt.Evidence) != 1 || !reusedReceipt.Evidence[0].Ref.SameExact(saved.Evidence[0].Ref) || reusedReceipt.Evidence[0].Digest != saved.Evidence[0].Digest {
				t.Fatalf("exact exchange reuse: %+v %v", reusedReceipt, err)
			}
			// Selection in B preserves A's origin, whether A already published
			// the exchange or B is its first publisher.
			for _, requestID := range []string{"1", "2"} {
				selectedCapture := capture
				selectedCapture.RequestID = requestID
				later := testSubmission("later-invocation", "reuse-local-"+requestID, []contracts.ArtifactRef{})
				later.HTTPEvidence = &selectedCapture
				later.HTTPEvidenceInvocationID = input.InvocationID
				receipt, replayed, err := service.Submit(ctx, grant, later)
				if err != nil || replayed || len(receipt.Evidence) != 1 || receipt.Origin.InvocationID != later.InvocationID {
					t.Fatalf("cross-invocation selection: %+v %v", receipt, err)
				}
				if requestID == "1" && !receipt.Evidence[0].Ref.SameExact(saved.Evidence[0].Ref) {
					t.Fatal("later selection duplicated the original exchange")
				}
				payload, err := runStore.Read(ctx, receipt.Evidence[0].Ref)
				if err != nil {
					t.Fatal(err)
				}
				envelope, err := auditdomain.DecodeHTTPExchangeArtifact(payload.Payload.Data)
				if err != nil || envelope.Origin.InvocationID != input.InvocationID {
					t.Fatalf("later selection rewrote capture provenance: %+v %v", envelope.Origin, err)
				}
				if replay, replayed, err := service.Submit(ctx, grant, later); err != nil || !replayed || !replay.Evidence[0].Ref.SameExact(receipt.Evidence[0].Ref) {
					t.Fatalf("later selection replay: %+v %v", replay, err)
				}
				changedOrigin := later
				changedOrigin.HTTPEvidenceInvocationID = later.InvocationID
				if _, _, err := service.Submit(ctx, grant, changedOrigin); !errors.Is(err, ErrConflict) {
					t.Fatalf("changed capture origin replay: %v", err)
				}
				later.EvidenceRefs = []contracts.ArtifactRef{receipt.Evidence[0].Ref}
				later.Proposal.EvidenceIDs = []string{"evidence-1"}
				if _, _, err := service.Submit(ctx, grant, later); !errors.Is(err, ErrInvalid) {
					t.Fatalf("duplicate raw/exact selection in later invocation: %v", err)
				}
			}
			unchanged, err := runStore.Read(ctx, saved.Evidence[0].Ref)
			if err != nil || string(unchanged.Payload.Data) != string(read.Payload.Data) {
				t.Fatalf("later consumer rewrote exchange: %v", err)
			}
			changed := capture
			changed.RequestTag = "r-changed"
			input.HTTPEvidence = &changed
			input.HTTPEvidenceInvocationID = input.InvocationID
			if _, _, err := service.Submit(ctx, grant, input); !errors.Is(err, ErrConflict) {
				t.Fatalf("changed replay: %v", err)
			}
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = 'http-exchanges'`, run).Scan(&count); err != nil || count != 2 {
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
			missingResponse.HTTPEvidenceInvocationID = missingResponse.InvocationID
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
