//go:build integration

package findingintake

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// stageReadCounter counts full StageExecution reads (runstore
// GetStageExecution), not the narrower lock or existence probes.
type stageReadCounter struct{ reads atomic.Int64 }

func (c *stageReadCounter) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	if strings.Contains(data.SQL, "resume_source_execution_id") &&
		strings.HasSuffix(data.SQL, "FROM stage_executions WHERE stage_execution_id = $1") {
		c.reads.Add(1)
	}
	return ctx
}
func (*stageReadCounter) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

// Anchor and fingerprint derivation share one lazily read pre-transaction
// Stage snapshot. The intake transaction still reads the Stage itself for
// provenance, and a submission whose derivations need no Stage reads it only
// there.
func TestPostgresIntakePreparationSharesOneStageRead(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	base := isolatedFindingPool(t, ctx)
	counter := &stageReadCounter{}
	cfg := base.Config()
	cfg.MaxConns = 1
	cfg.ConnConfig.Tracer = counter
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	var fixture struct {
		File      string
		Line      int64
		ZipBase64 string
		Derived   auditdomain.FindingAnchorDerivation
	}
	data, err := os.ReadFile("testdata/anchor-runtime.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	zipData, err := base64.StdEncoding.DecodeString(fixture.ZipBase64)
	if err != nil {
		t.Fatal(err)
	}
	const owner, project, run = "stage-share-owner", "stage-share-project", "stage-share-run"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Stage share", IdempotencyKey: "stage-share", RequestDigest: auditdomain.DigestBytes([]byte("stage-share"))}); err != nil {
		t.Fatal(err)
	}
	var refs []contracts.ArtifactRef
	grant := quotaSubmissionGrantWithInputs(t, ctx, pool, owner, project, run, func() map[string]runstore.PinnedContextArtifact {
		refs = writeFingerprintReportInputs(t, ctx, pool, run, []byte(`{"version":"2.1.0","runs":[]}`), zipData)
		return map[string]runstore.PinnedContextArtifact{"source": {Artifact: &refs[0]}}
	})
	fixture.Derived.SourceRef = &refs[0]
	service, err := New(pool)
	if err != nil {
		t.Fatal(err)
	}
	var prepared int64 = -1
	service.afterPrepare = func(context.Context) { prepared = counter.reads.Load() }
	submit := func(grant controlplane.AllocationGrant, input Submission) (Receipt, int64, int64, error) {
		counter.reads.Store(0)
		prepared = -1
		receipt, _, err := service.Submit(ctx, grant, input)
		return receipt, prepared, counter.reads.Load(), err
	}
	located := func(invocation string, evidence []contracts.ArtifactRef) Submission {
		input := testSubmission(invocation, "key-"+invocation, evidence)
		input.Proposal.Subject = nil
		input.Proposal.Locations = []auditdomain.FindingLocation{{File: fixture.File, Line: &fixture.Line}}
		input.Proposal.StandardRefs = []auditdomain.StandardReference{{Scheme: "CWE", Version: "4.20", RequirementID: "CWE-321"}}
		return input
	}

	both := located("both", refs[:1])
	both.DerivedAnchors, err = strictjson.Canonical(fixture.Derived)
	if err != nil {
		t.Fatal(err)
	}
	receipt, before, total, err := submit(grant, both)
	if err != nil || receipt.Anchors.Availability != "available" || receipt.Fingerprint.Availability != "unavailable" {
		t.Fatalf("anchor and fingerprint derivation=%+v %+v %v", receipt.Anchors, receipt.Fingerprint, err)
	}
	if before != 1 || total != 2 {
		t.Fatalf("both derivations read the Stage %d times before the transaction and %d in total, want 1 and 2", before, total)
	}

	fingerprintOnly := located("fingerprint-only", refs[1:])
	receipt, before, total, err = submit(grant, fingerprintOnly)
	if err != nil || receipt.Anchors.Reason != "not_provided" {
		t.Fatalf("fingerprint-only derivation=%+v %v", receipt.Anchors, err)
	}
	if before != 1 || total != 2 {
		t.Fatalf("one derivation read the Stage %d times before the transaction and %d in total, want 1 and 2", before, total)
	}

	neither := testSubmission("neither", "key-neither", refs[:1])
	receipt, before, total, err = submit(grant, neither)
	if err != nil || receipt.Anchors.Reason != "not_provided" || receipt.Fingerprint.Reason != "not_provided" {
		t.Fatalf("ordinary submission=%+v %+v %v", receipt.Anchors, receipt.Fingerprint, err)
	}
	if before != 0 || total != 1 {
		t.Fatalf("ordinary submission read the Stage %d times before the transaction and %d in total, want 0 and 1", before, total)
	}

	// A Stage of another Run denies the shared snapshot before the
	// transaction; the second derivation does not read it again.
	foreign := grant
	foreign.RunID = "stage-share-other-run"
	foreignInput := located("foreign", refs[:1])
	foreignInput.DerivedAnchors = both.DerivedAnchors
	if _, before, total, err = submit(foreign, foreignInput); !errors.Is(err, ErrAccessDenied) || before != -1 || total != 1 {
		t.Fatalf("foreign Stage=%v, prepared after %d reads, %d reads in total", err, before, total)
	}
	var count int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM finding_proposal_receipts").Scan(&count); err != nil || count != 3 {
		t.Fatalf("receipts=%d %v", count, err)
	}
}
