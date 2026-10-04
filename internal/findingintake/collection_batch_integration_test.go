//go:build integration

package findingintake

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

type collectionQueryCounter struct {
	count       atomic.Int64
	runReads    atomic.Int64
	directReads atomic.Int64
}

func (c *collectionQueryCounter) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	c.count.Add(1)
	if strings.Contains(data.SQL, "FROM workflow_runs WHERE run_id = $1") {
		c.runReads.Add(1)
	}
	if strings.Contains(data.SQL, "blob.payload") && len(data.Args) >= 4 &&
		data.Args[2] == "outputs" && data.Args[3] == "result" {
		c.directReads.Add(1)
	}
	return ctx
}
func (*collectionQueryCounter) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

func TestPostgresRunProposalPagesUseBatchedReadsAndSkipRetainedDocuments(t *testing.T) {
	f := newDeletionImportFixture(t)
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	f.ctx = ctx
	for i := 1; i < 200; i++ {
		insertAuditChildReceipt(t, f, fmt.Sprintf("candidate-%03d", i))
	}
	trace := &collectionQueryCounter{}
	actor, _ := deletionActorPool(t, f, trace)
	reader, err := New(actor)
	if err != nil {
		t.Fatal(err)
	}
	trace.count.Store(0)
	page, err := reader.ListRun(ctx, f.request.OwnerID, f.request.RunID, ListQuery{Limit: 200})
	if err != nil || len(page) != 200 {
		t.Fatalf("Run page: %d receipts, %v", len(page), err)
	}
	if got := trace.count.Load(); got > 9 { // page + holds + seven artifact batches
		t.Fatalf("200-receipt Run page used %d queries, want at most 9", got)
	}
	if page[0].Document.ClientKey == "" || page[199].Document.ClientKey == "" {
		t.Fatal("batched Run page omitted proposal documents")
	}
	start := time.Now()
	for _, receipt := range page {
		request := f.request
		request.Proposal = receipt.Proposal.Ref
		if _, _, err := f.intake.RetainAuditCollection(ctx, request); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("retained 200 proposals in %s", time.Since(start))
	trace.count.Store(0)
	collected, err := reader.ListAuditCollection(ctx, f.request.OwnerID, f.request.AuditID, f.request.RunID, ListQuery{Limit: 200})
	if err != nil || len(collected) != 200 {
		t.Fatalf("collection page: %d receipts, %v", len(collected), err)
	}
	if got := trace.count.Load(); got != 2 {
		t.Fatalf("retained collection page used %d queries, want 2", got)
	}
	for _, receipt := range collected {
		if !receipt.Retained || receipt.Receipt.Document.ClientKey != "" {
			t.Fatal("retained collection page rehydrated a proposal")
		}
	}
}

const largeDirectProposals = 2000

func newLargeDirectCollectionFixture(t *testing.T) deletionImportFixture {
	t.Helper()
	f := newDirectCollectionFixture(t)
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	t.Cleanup(cancel)
	f.ctx = ctx
	verifications := make([]auditdomain.DirectVerificationResult, largeDirectProposals)
	for i := range verifications {
		key := fmt.Sprintf("candidate-%04d", i)
		insertAuditChildReceipt(t, f, key)
		verifications[i] = auditdomain.DirectVerificationResult{
			InvocationID: key + "-invocation", ClientKey: key,
			Assessment: "supported", Summary: "Verified exact candidate.", EvidenceIDs: []string{},
		}
	}
	encoded, err := auditdomain.EncodeDirectVerificationSet(auditdomain.DirectVerificationSet{
		Schema: auditdomain.DirectVerificationsSchema, Verifications: verifications,
	})
	if err != nil {
		t.Fatal(err)
	}
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(f.pool))
	runArtifacts, err := artifactService.Run(f.request.RunID)
	if err != nil {
		t.Fatal(err)
	}
	output, err := runArtifacts.Write(ctx,
		artifacts.ArtifactRef{Namespace: "builder", Name: "direct-result"},
		artifacts.Payload{MediaType: auditdomain.DirectVerificationsMediaType, Data: encoded}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifactService.BindOutputExact(ctx, f.request.RunID, "result", output.Ref, nil); err != nil {
		t.Fatal(err)
	}
	if err := artifactService.FreezeRunOutputs(ctx, f.request.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := runstore.NewPostgresStore(f.pool).TransitionRun(ctx, f.request.RunID,
		runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "collection_fixture_succeeded"}); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestPostgresLargeDirectCollectionFinishesWithinDefaultTimeout(t *testing.T) {
	f := newLargeDirectCollectionFixture(t)
	ctx := f.ctx
	trace := &collectionQueryCounter{}
	actorPool, _ := deletionActorPool(t, f, trace)
	actor, err := New(actorPool)
	if err != nil {
		t.Fatal(err)
	}
	f.intake = actor
	start := time.Now()
	collectionCtx, cancel := context.WithTimeout(ctx, largeDirectCollectionTestTimeout)
	defer cancel()
	if err := collectFixtureProposals(collectionCtx, f); err != nil {
		t.Fatalf("fresh 2,001-proposal collection: %v", err)
	}
	t.Logf("fresh 2,001-proposal collection completed in %s", time.Since(start))
	if trace.runReads.Load() != 1 || trace.directReads.Load() != 1 {
		t.Fatalf("fresh collection resolved Run/output %d/%d times, want once each", trace.runReads.Load(), trace.directReads.Load())
	}
}

func TestPostgresLargeDirectCollectionFinishesAfterTimedOutAttempt(t *testing.T) {
	f := newLargeDirectCollectionFixture(t)
	ctx := f.ctx
	const proposals = largeDirectProposals
	firstCtx, firstCancel := context.WithTimeout(ctx, 3*time.Second)
	defer firstCancel()
	if err := collectFixtureProposals(firstCtx, f); err == nil {
		t.Fatal("short first collection unexpectedly finished")
	} else {
		t.Logf("first collection stopped: %v", err)
	}
	var firstCount int
	if err := f.pool.QueryRow(ctx, `SELECT count(*) FROM finding_proposal_audit_holds WHERE audit_id = $1`, f.request.AuditID).Scan(&firstCount); err != nil {
		t.Fatal(err)
	}
	if firstCount == 0 || firstCount >= proposals+1 {
		t.Fatalf("timed-out collection retained %d proposals", firstCount)
	}
	start := time.Now()
	retryCtx, retryCancel := context.WithTimeout(ctx, largeDirectCollectionTestTimeout)
	defer retryCancel()
	if err := collectFixtureProposals(retryCtx, f); err != nil {
		t.Fatalf("collection retry after %d holds: %v", firstCount, err)
	}
	t.Logf("collected %d remaining proposals in %s", proposals+1-firstCount, time.Since(start))
	var finalCount int
	if err := f.pool.QueryRow(ctx, `SELECT count(*) FROM finding_proposal_audit_holds WHERE audit_id = $1`, f.request.AuditID).Scan(&finalCount); err != nil || finalCount != proposals+1 {
		t.Fatalf("retained %d/%d proposals: %v", finalCount, proposals+1, err)
	}
	trace := &collectionQueryCounter{}
	actorPool, _ := deletionActorPool(t, f, trace)
	actor, err := New(actorPool)
	if err != nil {
		t.Fatal(err)
	}
	f.intake = actor
	trace.count.Store(0)
	if err := collectFixtureProposals(ctx, f); err != nil {
		t.Fatal(err)
	}
	if got := trace.count.Load(); got > 2*((proposals+1)/200+1) || trace.directReads.Load() != 0 {
		t.Fatalf("completed collection replay used %d queries and %d output reads", got, trace.directReads.Load())
	}
}

func collectFixtureProposals(ctx context.Context, f deletionImportFixture) error {
	query := ListQuery{Limit: 200}
	ctx = WithCollectionDirectVerificationCache(ctx)
	for {
		page, err := f.intake.ListAuditCollection(ctx, f.request.OwnerID, f.request.AuditID, f.request.RunID, query)
		if err != nil {
			return err
		}
		pending := make([]ImportRequest, 0, len(page))
		for _, candidate := range page {
			if candidate.Retained && (candidate.PostTerminalRetained || candidate.DirectAssessed) {
				continue
			}
			request := f.request
			request.Proposal = candidate.Receipt.Proposal.Ref
			pending = append(pending, request)
		}
		if len(pending) != 0 {
			if err := f.intake.RetainAuditCollectionBatch(ctx, pending); err != nil {
				return err
			}
		}
		if len(page) < query.Limit {
			return nil
		}
		last := page[len(page)-1].Receipt
		query.AfterCreatedAt, query.AfterReceiptID = &last.CreatedAt, last.ReceiptID
	}
}
