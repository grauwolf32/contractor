//go:build integration

package findingintake

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
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
	requests := make([]ImportRequest, len(page))
	for index, receipt := range page {
		requests[index] = f.request
		requests[index].Proposal = receipt.Proposal.Ref
	}
	if err := f.intake.RetainAuditCollectionBatch(ctx, requests); err != nil {
		t.Fatal(err)
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

// Every attempt runs under the production collection timeout and must keep
// what it committed, so the collection finishes across attempts however long
// the whole Run takes.
func TestPostgresLargeDirectCollectionFinishesWithinDefaultTimeout(t *testing.T) {
	f := newLargeDirectCollectionFixture(t)
	trace := &collectionQueryCounter{}
	transactions := &collectionTransactionTracer{}
	actorPool, _ := deletionActorPool(t, f, multiQueryTracer{trace, transactions})
	actor, err := New(actorPool)
	if err != nil {
		t.Fatal(err)
	}
	f.intake = actor
	start := time.Now()
	attempts := collectAcrossAttempts(t, f, true)
	t.Logf("fresh 2,001-proposal collection completed in %d attempt(s), %s", attempts, time.Since(start))
	assertFixtureHolds(t, f, largeDirectProposals+1)
	// Each attempt resolves the terminal Run and its direct output once.
	if trace.runReads.Load() != int64(attempts) || trace.directReads.Load() != int64(attempts) {
		t.Fatalf("collection resolved Run/output %d/%d times over %d attempt(s)",
			trace.runReads.Load(), trace.directReads.Load(), attempts)
	}
	transactions.assertBounded(t)
}

// An attempt that stops after committing part of the Run keeps that part,
// and the next attempt retains only the rest. The first attempt is ended
// right after its first retention transaction commits, as an expiring
// deadline would, independent of collection speed.
func TestPostgresLargeDirectCollectionFinishesAfterTimedOutAttempt(t *testing.T) {
	f := newLargeDirectCollectionFixture(t)
	ctx := f.ctx
	const proposals = largeDirectProposals
	firstCtx, firstCancel := context.WithCancel(ctx)
	defer firstCancel()
	interrupted := &collectionTransactionTracer{interrupt: firstCancel}
	actorPool, _ := deletionActorPool(t, f, interrupted)
	actor, err := New(actorPool)
	if err != nil {
		t.Fatal(err)
	}
	first := f
	first.intake = actor
	if err := collectFixtureProposals(firstCtx, first, true); err == nil {
		t.Fatal("interrupted first collection unexpectedly finished")
	} else {
		t.Logf("first collection stopped: %v", err)
	}
	firstCount := countFixtureHolds(t, f)
	if interrupted.committed != 1 || firstCount == 0 || firstCount >= proposals+1 {
		t.Fatalf("interrupted collection committed %d transaction(s) retaining %d/%d proposals",
			interrupted.committed, firstCount, proposals+1)
	}
	retried := &collectionTransactionTracer{}
	retryPool, _ := deletionActorPool(t, f, retried)
	retryActor, err := New(retryPool)
	if err != nil {
		t.Fatal(err)
	}
	f.intake = retryActor
	attempts := collectAcrossAttempts(t, f, true)
	t.Logf("collected %d remaining proposals in %d attempt(s)", proposals+1-firstCount, attempts)
	assertFixtureHolds(t, f, proposals+1)
	retried.assertBounded(t)
	trace := &collectionQueryCounter{}
	actorPool, _ = deletionActorPool(t, f, trace)
	actor, err = New(actorPool)
	if err != nil {
		t.Fatal(err)
	}
	f.intake = actor
	trace.count.Store(0)
	if err := collectFixtureProposals(ctx, f, true); err != nil {
		t.Fatal(err)
	}
	if got := trace.count.Load(); got > 2*((proposals+1)/200+1) || trace.directReads.Load() != 0 {
		t.Fatalf("completed collection replay used %d queries and %d output reads", got, trace.directReads.Load())
	}
}

// A page of proposals that each cite many evidence revisions may not commit
// in one transaction within the production timeout. Bounded transactions
// keep each attempt's progress, so the page completes across attempts. The
// first attempt is ended after its first retention transaction commits.
func TestPostgresEvidenceHeavyCollectionPageCommitsBoundedTransactions(t *testing.T) {
	const proposals, evidencePerProposal = 200, 48
	f := newDeletionImportFixture(t)
	ctx, cancel := context.WithTimeout(t.Context(), 4*time.Minute)
	defer cancel()
	f.ctx = ctx
	evidence := writeFixtureEvidence(t, f, evidencePerProposal)
	for i := range proposals {
		insertAuditChildReceiptWithEvidence(t, f, fmt.Sprintf("evidence-candidate-%03d", i), evidence)
	}
	firstCtx, firstCancel := context.WithCancel(ctx)
	defer firstCancel()
	transactions := &collectionTransactionTracer{interrupt: firstCancel}
	actorPool, _ := deletionActorPool(t, f, transactions)
	actor, err := New(actorPool)
	if err != nil {
		t.Fatal(err)
	}
	f.intake = actor
	if err := collectFixtureProposals(firstCtx, f, false); err == nil {
		t.Fatal("interrupted first collection unexpectedly finished")
	}
	if firstCount := countFixtureHolds(t, f); firstCount == 0 || firstCount >= proposals {
		t.Fatalf("interrupted attempt retained %d/%d proposals", firstCount, proposals+1)
	}
	start := time.Now()
	attempts := collectAcrossAttempts(t, f, false)
	t.Logf("retained %d proposals with %d evidence revisions each in %d more attempt(s), %d transactions, %s",
		proposals, evidencePerProposal, attempts, transactions.committed, time.Since(start))
	assertFixtureHolds(t, f, proposals+1)
	var retainedEvidence int
	if err := f.pool.QueryRow(ctx, `
SELECT COALESCE(sum(jsonb_array_length(evidence)), 0)::integer
  FROM finding_proposal_audit_holds WHERE audit_id = $1`, f.request.AuditID).Scan(&retainedEvidence); err != nil ||
		retainedEvidence != proposals*evidencePerProposal {
		t.Fatalf("retained evidence = %d, %v; want %d", retainedEvidence, err, proposals*evidencePerProposal)
	}
	if transactions.committed < proposals*(1+evidencePerProposal)/maxCollectionTransactionWork {
		t.Fatalf("page committed in %d transaction(s)", transactions.committed)
	}
	transactions.assertBounded(t)
}

// collectionAttemptBudget limits how many production-timeout attempts a
// fixture collection may take. Each attempt commits at least one bounded
// transaction, so the limit only stops a collection that makes no progress.
const collectionAttemptBudget = 20

// collectAcrossAttempts repeats collection as the Controller does, each
// attempt under the production timeout, until one finishes. It returns the
// number of attempts; an attempt that stops without new holds fails the test.
func collectAcrossAttempts(t *testing.T, f deletionImportFixture, sourceSucceeded bool) int {
	t.Helper()
	for attempt := 1; attempt <= collectionAttemptBudget; attempt++ {
		before := countFixtureHolds(t, f)
		attemptCtx, cancel := context.WithTimeout(f.ctx, collectionAttemptTimeout)
		err := collectFixtureProposals(attemptCtx, f, sourceSucceeded)
		expired := attemptCtx.Err() != nil
		cancel()
		if err == nil {
			return attempt
		}
		if !expired {
			t.Fatalf("collection attempt %d: %v", attempt, err)
		}
		if after := countFixtureHolds(t, f); after <= before {
			t.Fatalf("collection attempt %d committed no progress (%d holds): %v", attempt, after, err)
		}
	}
	t.Fatalf("collection did not finish in %d attempts", collectionAttemptBudget)
	return 0
}

func countFixtureHolds(t *testing.T, f deletionImportFixture) int {
	t.Helper()
	var count int
	if err := f.pool.QueryRow(f.ctx, `
SELECT count(*) FROM finding_proposal_audit_holds WHERE audit_id = $1`, f.request.AuditID).Scan(&count); err != nil {
		t.Fatal(err)
	}
	return count
}

func assertFixtureHolds(t *testing.T, f deletionImportFixture, want int) {
	t.Helper()
	if got := countFixtureHolds(t, f); got != want {
		t.Fatalf("retained %d/%d proposals", got, want)
	}
}

// writeFixtureEvidence writes count exact evidence revisions into the
// fixture's source Run, in canonical reference order.
func writeFixtureEvidence(t *testing.T, f deletionImportFixture, count int) []ExactArtifact {
	t.Helper()
	runArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(f.pool)).Run(f.request.RunID)
	if err != nil {
		t.Fatal(err)
	}
	evidence := make([]ExactArtifact, count)
	for i := range evidence {
		data := []byte(fmt.Sprintf("evidence body %03d", i))
		written, err := runArtifacts.Write(f.ctx,
			artifacts.ArtifactRef{Namespace: "evidence", Name: fmt.Sprintf("item-%03d", i)},
			artifacts.Payload{MediaType: "text/plain", Data: data}, nil)
		if err != nil {
			t.Fatal(err)
		}
		evidence[i] = ExactArtifact{
			Ref: written.Ref, Digest: auditdomain.DigestBytes(data),
			MediaType: written.MediaType, SizeBytes: written.Size,
		}
	}
	return evidence
}

// collectionTransactionTracer observes the retention transactions of a
// single-connection actor pool: how many committed and the most artifact
// imports one of them performed. With interrupt set, it cancels the attempt
// right after the first retention transaction commits.
type collectionTransactionTracer struct {
	mu         sync.Mutex
	imports    int
	maxImports int
	committed  int
	interrupt  context.CancelFunc
}

type collectionCommitKey struct{}

func (c *collectionTransactionTracer) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	statement := strings.ToLower(strings.TrimSpace(data.SQL))
	c.mu.Lock()
	defer c.mu.Unlock()
	switch {
	case strings.HasPrefix(statement, "begin"), statement == "rollback":
		c.imports = 0
	case strings.Contains(statement, "'audit_import'"):
		c.imports++
	case statement == "commit":
		return context.WithValue(ctx, collectionCommitKey{}, c.imports)
	}
	return ctx
}

func (c *collectionTransactionTracer) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryEndData) {
	imports, commit := ctx.Value(collectionCommitKey{}).(int)
	if !commit || imports == 0 || data.Err != nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.committed++
	c.maxImports = max(c.maxImports, imports)
	if c.interrupt != nil {
		c.interrupt()
		c.interrupt = nil
	}
}

// assertBounded checks that no retention transaction exceeded the work bound
// by more than the one proposal that may cross it.
func (c *collectionTransactionTracer) assertBounded(t *testing.T) {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.committed == 0 || c.maxImports > maxCollectionTransactionWork+1+MaxEvidenceRefs {
		t.Fatalf("%d retention transaction(s), largest imported %d artifacts", c.committed, c.maxImports)
	}
}

type multiQueryTracer []pgx.QueryTracer

func (m multiQueryTracer) TraceQueryStart(ctx context.Context, conn *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	for _, tracer := range m {
		ctx = tracer.TraceQueryStart(ctx, conn, data)
	}
	return ctx
}

func (m multiQueryTracer) TraceQueryEnd(ctx context.Context, conn *pgx.Conn, data pgx.TraceQueryEndData) {
	for _, tracer := range m {
		tracer.TraceQueryEnd(ctx, conn, data)
	}
}

func collectFixtureProposals(ctx context.Context, f deletionImportFixture, sourceSucceeded bool) error {
	query := ListQuery{Limit: 200}
	ctx = WithCollectionDirectVerificationCache(ctx)
	for {
		page, err := f.intake.ListAuditCollection(ctx, f.request.OwnerID, f.request.AuditID, f.request.RunID, query)
		if err != nil {
			return err
		}
		pending := make([]ImportRequest, 0, len(page))
		for _, candidate := range page {
			if !candidate.NeedsRetention(sourceSucceeded) {
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

// A proposal whose own evidence cannot be retained is rejected alone: one
// with a missing evidence revision and one whose evidence no longer matches
// its receipt leave the other proposals of the page collectable.
func TestPostgresCollectionRejectsOnlyUnretainableProposals(t *testing.T) {
	f := newDeletionImportFixture(t)
	evidence := writeFixtureEvidence(t, f, 2)
	missingRevision := "missing-revision"
	missing := evidence[1]
	missing.Ref.Revision = &missingRevision
	corrupt := evidence[1]
	corrupt.SizeBytes++
	unretainable := map[string]string{
		insertAuditChildReceiptWithEvidence(t, f, "missing-evidence", []ExactArtifact{evidence[0], missing}).ReceiptID: "finding-proposal-artifact-missing",
		insertAuditChildReceiptWithEvidence(t, f, "corrupt-evidence", []ExactArtifact{evidence[0], corrupt}).ReceiptID: "finding-proposal-artifact-invalid",
	}
	for i := range 3 {
		insertAuditChildReceiptWithEvidence(t, f, fmt.Sprintf("valid-candidate-%d", i), evidence)
	}
	if err := collectFixtureProposals(f.ctx, f, false); err != nil {
		t.Fatalf("collection with unretainable proposals: %v", err)
	}
	assertFixtureHolds(t, f, 4)
	rows, err := f.pool.Query(f.ctx, `
SELECT entity_id, summary ->> 'reason' FROM audit_events
 WHERE audit_id = $1 AND kind = 'finding.proposal_rejected'`, f.request.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	rejected := map[string]string{}
	for rows.Next() {
		var receiptID, reason string
		if err := rows.Scan(&receiptID, &reason); err != nil {
			t.Fatal(err)
		}
		rejected[receiptID] = reason
	}
	if err := rows.Err(); err != nil || len(rejected) != len(unretainable) {
		t.Fatalf("rejected proposals = %v, %v; want %v", rejected, err, unretainable)
	}
	for receiptID, reason := range unretainable {
		if rejected[receiptID] != reason {
			t.Fatalf("rejected proposals = %v, want %v", rejected, unretainable)
		}
	}
	page, err := f.intake.ListAuditCollection(f.ctx, f.request.OwnerID, f.request.AuditID, f.request.RunID, ListQuery{Limit: 200})
	if err != nil || len(page) != 6 {
		t.Fatalf("collection page = %d receipts, %v", len(page), err)
	}
	for _, candidate := range page {
		if candidate.NeedsRetention(false) {
			t.Fatalf("receipt %s still needs retention: %+v", candidate.Receipt.ReceiptID, candidate)
		}
	}
	// A replay over the settled page changes nothing.
	audit, err := auditstore.NewPostgresStore(f.pool).Get(f.ctx, f.request.OwnerID, f.request.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if err := collectFixtureProposals(f.ctx, f, false); err != nil {
		t.Fatal(err)
	}
	replayed, err := auditstore.NewPostgresStore(f.pool).Get(f.ctx, f.request.OwnerID, f.request.AuditID)
	if err != nil || replayed.Revision != audit.Revision {
		t.Fatalf("collection replay changed the Audit: %+v, %v", replayed, err)
	}
}

// Rejecting a proposal replays the transaction that failed on it. Direct
// assessments that fit the evidence budget in that rolled-back transaction
// must still be accepted by the replay.
func TestPostgresCollectionRejectionKeepsEarlierDirectAssessments(t *testing.T) {
	f := newDirectCollectionFixture(t)
	evidence := writeFixtureEvidence(t, f, 1)
	missingRevision := "missing-revision"
	missing := evidence[0]
	missing.Ref.Revision = &missingRevision
	keys := []string{"direct-0", "direct-1", "direct-2", "missing-evidence", "direct-3"}
	verifications := make([]auditdomain.DirectVerificationResult, len(keys))
	var rejectedID string
	for index, key := range keys {
		cited := []ExactArtifact{}
		if key == "missing-evidence" {
			cited = []ExactArtifact{missing}
		}
		receipt := insertAuditChildReceiptWithEvidence(t, f, key, cited)
		if key == "missing-evidence" {
			rejectedID = receipt.ReceiptID
		}
		verifications[index] = auditdomain.DirectVerificationResult{
			InvocationID: key + "-invocation", ClientKey: key,
			Assessment: "supported", Summary: "Verified exact candidate.", EvidenceIDs: []string{},
		}
	}
	sort.Slice(verifications, func(i, j int) bool { return verifications[i].ClientKey < verifications[j].ClientKey })
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
	output, err := runArtifacts.Write(f.ctx,
		artifacts.ArtifactRef{Namespace: "builder", Name: "direct-result"},
		artifacts.Payload{MediaType: auditdomain.DirectVerificationsMediaType, Data: encoded}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifactService.BindOutputExact(f.ctx, f.request.RunID, "result", output.Ref, nil); err != nil {
		t.Fatal(err)
	}
	if err := artifactService.FreezeRunOutputs(f.ctx, f.request.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := runstore.NewPostgresStore(f.pool).TransitionRun(f.ctx, f.request.RunID,
		runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "collection_fixture_succeeded"}); err != nil {
		t.Fatal(err)
	}
	// Admit exactly two direct assessments: the third verified proposal finds
	// the budget exhausted before the transaction fails on the missing evidence.
	base, err := readReceiptByID(f.ctx, f.pool, f.receiptID)
	if err != nil {
		t.Fatal(err)
	}
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := prepareDirectVerification(f.ctx, tx, artifacts.NewService(artifacts.NewPostgresRepository(tx)),
		directVerificationInput{
			AuditID: f.request.AuditID, ProjectID: "delete-project", RunID: f.request.RunID,
			WorkflowClosureDigest: base.Origin.Workflow.ClosureDigest,
		})
	if rollbackErr := tx.Rollback(f.ctx); err != nil || rollbackErr != nil || prepared == nil {
		t.Fatalf("prepare direct verification = (%+v, %v, %v)", prepared, err, rollbackErr)
	}
	charge := prepared.descriptor.Size + int64(len(prepared.contractBytes))
	if _, err := f.pool.Exec(f.ctx, `UPDATE audits SET max_evidence_bytes = $2 WHERE audit_id = $1`,
		f.request.AuditID, 2*charge+charge/2); err != nil {
		t.Fatal(err)
	}

	if err := collectFixtureProposals(f.ctx, f, true); err != nil {
		t.Fatalf("collection with one unretainable proposal: %v", err)
	}
	assertFixtureHolds(t, f, len(keys))
	rows, err := f.pool.Query(f.ctx, `
SELECT receipt.client_key
  FROM audit_finding_assessments AS assessment
  JOIN finding_proposal_receipts AS receipt USING (receipt_id)
 WHERE assessment.audit_id = $1 AND assessment.direct_verification
 ORDER BY receipt.client_key`, f.request.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	assessed := []string{}
	for rows.Next() {
		var clientKey string
		if err := rows.Scan(&clientKey); err != nil {
			t.Fatal(err)
		}
		assessed = append(assessed, clientKey)
	}
	if err := rows.Err(); err != nil || strings.Join(assessed, ",") != "direct-0,direct-1" {
		t.Fatalf("direct assessments = %v, %v; want direct-0 and direct-1", assessed, err)
	}
	var rejected bool
	if err := f.pool.QueryRow(f.ctx, `
SELECT EXISTS (SELECT 1 FROM audit_events
 WHERE audit_id = $1 AND kind = 'finding.proposal_rejected' AND entity_id = $2)`,
		f.request.AuditID, rejectedID).Scan(&rejected); err != nil || !rejected {
		t.Fatalf("missing-evidence proposal rejection = (%t, %v)", rejected, err)
	}
}
