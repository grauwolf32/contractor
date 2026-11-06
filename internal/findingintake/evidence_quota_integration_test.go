//go:build integration

package findingintake

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// A proposal's evidence, the stored sizes of its exact evidence refs plus the
// selected HTTP exchange Artifact, may total exactly LIM-FND-50 bytes. One
// byte more is the permanent invalid_finding_request answer of ErrInvalid and
// stores no receipt, proposal or exchange Artifact; a replay of an admitted
// proposal returns its receipt before the quota is checked (FND-R55).
func TestPostgresProposalEvidenceQuotaAdmitsExactlyItsLimit(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	const owner, project, run = "evidence-owner", "evidence-project", "evidence-run"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Evidence quota",
		IdempotencyKey: "evidence-project", RequestDigest: auditdomain.DigestBytes([]byte("evidence-project")),
	}); err != nil {
		t.Fatal(err)
	}
	grant := quotaSubmissionGrant(t, ctx, pool, owner, project, run)
	service, err := New(pool, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	var capture auditdomain.CapturedHTTPExchange
	if err := strictjson.Decode([]byte(`{"source":"http","request_id":"1","request_tag":"r-test-q000001","observed_at":"2026-11-06T18:20:00Z","attempts":[{"request":{"method":"GET","url":"https://target.example/","message":{"kind":"structured","headers":[],"body_base64":""}},"response":{"status":200,"message":{"kind":"structured","headers":[],"body_base64":"AAE="},"complete":true}}]}`), &capture); err != nil {
		t.Fatal(err)
	}
	// Exchange content records the original Run/invocation independently of its consumer.
	exchangeBytes := func(input Submission) int64 {
		data, err := auditdomain.EncodeHTTPExchangeArtifact(capture,
			grant.RunID, input.InvocationID)
		if err != nil {
			t.Fatal(err)
		}
		return int64(len(data))
	}
	exchangeAt := testSubmission("exchange-invocation", "exchange-at-limit", nil)
	exchangeOver := testSubmission("exchange-invocation", "exchange-over-limit", nil)
	exchange := exchangeBytes(exchangeAt)
	if exchangeBytes(exchangeOver) != exchange {
		t.Fatal("exchange Artifacts of the two submissions differ in size")
	}

	runArtifacts, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Run(run)
	if err != nil {
		t.Fatal(err)
	}
	write := func(name string, size int64) contracts.ArtifactRef {
		written, err := runArtifacts.Write(ctx, artifacts.ArtifactRef{Namespace: grant.Namespace, Name: name},
			artifacts.Payload{MediaType: "application/octet-stream", Data: bytes.Repeat([]byte{'e'}, int(size))}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return written.Ref
	}
	// One large Artifact serves every case; small ones complete each total.
	large := write("large", maxProposalEvidenceBytes-exchange-3)
	atLimit := []contracts.ArtifactRef{large, write("refs-at-limit", exchange+3)}
	overLimit := []contracts.ArtifactRef{large, write("refs-over-limit", exchange+4)}
	exchangeAt.EvidenceRefs, exchangeAt.Proposal.EvidenceIDs = []contracts.ArtifactRef{large, write("exchange-at-limit", 3)}, []string{"evidence-1", "evidence-2"}
	exchangeAt.HTTPEvidence = &capture
	exchangeOver.EvidenceRefs, exchangeOver.Proposal.EvidenceIDs = []contracts.ArtifactRef{large, write("exchange-over-limit", 4)}, []string{"evidence-1", "evidence-2"}
	exchangeOver.HTTPEvidence = &capture

	for _, over := range []Submission{testSubmission("refs-invocation", "refs-over-limit", overLimit), exchangeOver} {
		_, _, err := service.Submit(ctx, grant, over)
		if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "proposal evidence quota exceeded") {
			t.Fatalf("%s: submission one byte over LIM-FND-50 = %v, want the evidence quota rejection", over.Proposal.ClientKey, err)
		}
	}
	if receipts, proposals := evidenceQuotaStored(t, ctx, pool, run); receipts != 0 || proposals != 0 {
		t.Fatalf("rejected submissions stored %d receipts and %d finding-proposals bindings", receipts, proposals)
	}

	for _, admitted := range []Submission{testSubmission("refs-invocation", "refs-at-limit", atLimit), exchangeAt} {
		receipt, replayed, err := service.Submit(ctx, grant, admitted)
		if err != nil || replayed {
			t.Fatalf("%s: submission at LIM-FND-50 = (replayed=%t, %v), want admitted", admitted.Proposal.ClientKey, replayed, err)
		}
		var total int64
		for _, evidence := range receipt.Evidence {
			total += evidence.SizeBytes
		}
		if total != maxProposalEvidenceBytes {
			t.Fatalf("%s: admitted evidence totals %d bytes, want %d", admitted.Proposal.ClientKey, total, maxProposalEvidenceBytes)
		}
		replay, replayed, err := service.Submit(ctx, grant, admitted)
		if err != nil || !replayed || replay.ReceiptID != receipt.ReceiptID {
			t.Fatalf("%s: replay at the limit = (%s, replayed=%t, %v)", admitted.Proposal.ClientKey, replay.ReceiptID, replayed, err)
		}
	}
	// Exchanges have their own frozen namespace; only proposals are counted here.
	if receipts, proposals := evidenceQuotaStored(t, ctx, pool, run); receipts != 2 || proposals != 2 {
		t.Fatalf("admitted submissions stored %d receipts and %d finding-proposals bindings, want 2 and 2", receipts, proposals)
	}
}

// evidenceQuotaStored counts the Run's receipts and finding-proposals
// bindings, which hold proposals independently of selected HTTP exchanges.
func evidenceQuotaStored(t *testing.T, ctx context.Context, pool *pgxpool.Pool, run string) (int, int) {
	t.Helper()
	var receipts, proposals int
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM finding_proposal_receipts WHERE run_id = $1),
       (SELECT count(*) FROM artifact_bindings
         WHERE scope_kind = 'run' AND scope_id = $1 AND namespace = 'finding-proposals')`, run).Scan(&receipts, &proposals); err != nil {
		t.Fatal(err)
	}
	return receipts, proposals
}
